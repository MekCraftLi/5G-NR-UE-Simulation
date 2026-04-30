import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.config import createSsbConfig
from common.cpManager import CpManager
from common.gscn import GscnRaster
from common.gscnRasterTable import buildGscnMatrix, findCasesByGscn
from pss.cpFreqOffsetEstimator import CpFreqOffsetEstimator
from pss.pssDetector import PssDetector
from pss.pssFreqOffsetEstimator import PssFreqOffsetEstimator
from pss.pssVisualizer import PssVisualizer
from sss.sssDetector import SssDetector
from sss.sssVisualizer import SssVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    sampleRate: float
    scs: int
    prbCount: int
    inputPath: str | None = None
    outputPrefix: str = ""
    pssWorkers: int | None = None
    pssParallelMode: str = "process"
    pssScoreMode: str = "raw"
    pssAdaptiveIterations: int = 5
    pssLsqMinR2: float = 0.8
    sssEnable: bool = True
    pssOnly: bool = False
    sssSearchStartSymbol: float = 0.5
    sssSearchEndSymbol: float = 2.5
    sssSearchStepSamples: int = 1
    progressMode: str = "auto"
    progressRefreshSec: float = 0.7


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5G NR PSS blind search with dynamic config loading."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sample rate in Hz")
    parser.add_argument("--scs", type=int, default=None, help="Subcarrier spacing in Hz")
    parser.add_argument("--prb-count", type=int, default=None, help="Number of PRBs")
    parser.add_argument("--input-path", type=str, default=None, help="Input .npy signal path")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output file prefix")
    parser.add_argument("--pss-workers", type=int, default=None, help="PSS detector worker count")
    parser.add_argument(
        "--pss-parallel-mode",
        type=str,
        default=None,
        choices=["process", "thread"],
        help="PSS detector parallel mode",
    )
    parser.add_argument(
        "--pss-score-mode",
        type=str,
        default=None,
        choices=["raw", "ncc"],
        help="PSS detector score mode",
    )
    parser.add_argument(
        "--pss-adaptive-iterations",
        type=int,
        default=None,
        help="Adaptive sub-fine refinement iterations",
    )
    parser.add_argument(
        "--pss-lsq-min-r2",
        type=float,
        default=None,
        help="Minimum R^2 to accept PSS-assisted CFO LS fit",
    )
    parser.add_argument(
        "--disable-sss",
        action="store_true",
        help="Disable SSS sliding search stage",
    )
    parser.add_argument(
        "--pss-only",
        action="store_true",
        help="Run PSS stage only and skip SSS stage",
    )
    parser.add_argument(
        "--sss-start-symbol",
        type=float,
        default=None,
        help="SSS sliding-search start (in OFDM-symbol units from PSS timing)",
    )
    parser.add_argument(
        "--sss-end-symbol",
        type=float,
        default=None,
        help="SSS sliding-search end (in OFDM-symbol units from PSS timing)",
    )
    parser.add_argument(
        "--sss-step-samples",
        type=int,
        default=None,
        help="SSS sliding-search step size in samples",
    )
    parser.add_argument(
        "--progress-mode",
        type=str,
        default=None,
        choices=["auto", "rich", "log"],
        help="Progress display mode",
    )
    parser.add_argument(
        "--progress-refresh",
        type=float,
        default=None,
        help="Progress refresh interval in seconds (log mode)",
    )
    return parser.parse_args(argv)


def _loadConfigFile(configPath: str) -> dict[str, Any]:
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML installed") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config format. Use .json/.yaml/.yml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file root must be an object/dict")
    return data


def _pick(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _buildRuntimeConfig(args: argparse.Namespace) -> RuntimeConfig:
    defaults = {
        "sampleRate": 30.72e6,
        "scs": 30000,
        "prbCount": 51,
        "inputPath": None,
        "outputPrefix": "",
        "pssWorkers": None,
        "pssParallelMode": "process",
        "pssScoreMode": "raw",
        "pssAdaptiveIterations": 5,
        "pssLsqMinR2": 0.8,
        "sssEnable": True,
        "pssOnly": False,
        "sssSearchStartSymbol": 0.5,
        "sssSearchEndSymbol": 2.5,
        "sssSearchStepSamples": 1,
        "progressMode": "auto",
        "progressRefreshSec": 0.7,
    }

    fileCfg: dict[str, Any] = {}
    if args.config:
        fileCfg = _loadConfigFile(args.config)

    sampleRate = float(
        args.sample_rate
        if args.sample_rate is not None
        else _pick(fileCfg, "sample_rate", "sampleRate", default=defaults["sampleRate"])
    )
    scs = int(
        args.scs
        if args.scs is not None
        else _pick(fileCfg, "scs", "subcarrier_spacing", "subcarrierSpacing", default=defaults["scs"])
    )
    prbCount = int(
        args.prb_count
        if args.prb_count is not None
        else _pick(fileCfg, "prb_count", "prbCount", "rb_count", default=defaults["prbCount"])
    )
    inputPath = (
        args.input_path
        if args.input_path is not None
        else _pick(fileCfg, "input_path", "inputPath", "signal_path", default=defaults["inputPath"])
    )
    outputPrefix = (
        args.output_prefix
        if args.output_prefix is not None
        else _pick(fileCfg, "output_prefix", "outputPrefix", default=defaults["outputPrefix"])
    )
    pssWorkersRaw = (
        args.pss_workers
        if args.pss_workers is not None
        else _pick(fileCfg, "pss_workers", "pssWorkers", default=defaults["pssWorkers"])
    )
    pssWorkers = int(pssWorkersRaw) if pssWorkersRaw is not None else None
    pssParallelMode = str(
        args.pss_parallel_mode
        if args.pss_parallel_mode is not None
        else _pick(fileCfg, "pss_parallel_mode", "pssParallelMode", default=defaults["pssParallelMode"])
    ).lower()
    pssScoreMode = str(
        args.pss_score_mode
        if args.pss_score_mode is not None
        else _pick(fileCfg, "pss_score_mode", "pssScoreMode", default=defaults["pssScoreMode"])
    ).lower()
    pssAdaptiveIterations = int(
        args.pss_adaptive_iterations
        if args.pss_adaptive_iterations is not None
        else _pick(
            fileCfg,
            "pss_adaptive_iterations",
            "pssAdaptiveIterations",
            default=defaults["pssAdaptiveIterations"],
        )
    )
    pssLsqMinR2 = float(
        args.pss_lsq_min_r2
        if args.pss_lsq_min_r2 is not None
        else _pick(fileCfg, "pss_lsq_min_r2", "pssLsqMinR2", default=defaults["pssLsqMinR2"])
    )
    sssEnableCfg = _pick(fileCfg, "sss_enable", "sssEnable", default=defaults["sssEnable"])
    sssEnable = bool(sssEnableCfg)
    if args.disable_sss:
        sssEnable = False
    pssOnlyCfg = _pick(fileCfg, "pss_only", "pssOnly", default=defaults["pssOnly"])
    pssOnly = bool(pssOnlyCfg) or bool(args.pss_only)
    if pssOnly:
        sssEnable = False
    sssSearchStartSymbol = float(
        args.sss_start_symbol
        if args.sss_start_symbol is not None
        else _pick(
            fileCfg,
            "sss_start_symbol",
            "sssSearchStartSymbol",
            default=defaults["sssSearchStartSymbol"],
        )
    )
    sssSearchEndSymbol = float(
        args.sss_end_symbol
        if args.sss_end_symbol is not None
        else _pick(
            fileCfg,
            "sss_end_symbol",
            "sssSearchEndSymbol",
            default=defaults["sssSearchEndSymbol"],
        )
    )
    sssSearchStepSamples = int(
        args.sss_step_samples
        if args.sss_step_samples is not None
        else _pick(
            fileCfg,
            "sss_step_samples",
            "sssSearchStepSamples",
            default=defaults["sssSearchStepSamples"],
        )
    )
    progressMode = str(
        args.progress_mode
        if args.progress_mode is not None
        else _pick(fileCfg, "progress_mode", "progressMode", default=defaults["progressMode"])
    ).lower()
    progressRefreshSec = float(
        args.progress_refresh
        if args.progress_refresh is not None
        else _pick(fileCfg, "progress_refresh", "progressRefreshSec", default=defaults["progressRefreshSec"])
    )

    if sampleRate <= 0:
        raise ValueError("sampleRate must be > 0")
    if scs <= 0:
        raise ValueError("scs must be > 0")
    if prbCount <= 0:
        raise ValueError("prbCount must be > 0")
    if pssWorkers is not None and pssWorkers <= 0:
        raise ValueError("pssWorkers must be > 0 when provided")
    if pssParallelMode not in ("process", "thread"):
        raise ValueError("pssParallelMode must be one of: process, thread")
    if pssScoreMode not in ("raw", "ncc"):
        raise ValueError("pssScoreMode must be one of: raw, ncc")
    if pssAdaptiveIterations < 0:
        raise ValueError("pssAdaptiveIterations must be >= 0")
    if pssLsqMinR2 < 0.0 or pssLsqMinR2 > 1.0:
        raise ValueError("pssLsqMinR2 must be in [0, 1]")
    if sssSearchEndSymbol <= sssSearchStartSymbol:
        raise ValueError("sssSearchEndSymbol must be > sssSearchStartSymbol")
    if sssSearchStepSamples <= 0:
        raise ValueError("sssSearchStepSamples must be > 0")
    if progressMode not in ("auto", "rich", "log"):
        raise ValueError("progressMode must be one of: auto, rich, log")
    if progressRefreshSec <= 0:
        raise ValueError("progressRefreshSec must be > 0")

    return RuntimeConfig(
        sampleRate=sampleRate,
        scs=scs,
        prbCount=prbCount,
        inputPath=inputPath,
        outputPrefix=outputPrefix or "",
        pssWorkers=pssWorkers,
        pssParallelMode=pssParallelMode,
        pssScoreMode=pssScoreMode,
        pssAdaptiveIterations=pssAdaptiveIterations,
        pssLsqMinR2=pssLsqMinR2,
        sssEnable=sssEnable,
        pssOnly=pssOnly,
        sssSearchStartSymbol=sssSearchStartSymbol,
        sssSearchEndSymbol=sssSearchEndSymbol,
        sssSearchStepSamples=sssSearchStepSamples,
        progressMode=progressMode,
        progressRefreshSec=progressRefreshSec,
    )


def _loadRxSignal(inputPath: str | None) -> np.ndarray:
    if inputPath:
        path = Path(inputPath)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input signal file not found: {path}")
        logger.info(f"Loading input signal: {path}")
        return np.load(str(path))

    dataDir = Path(__file__).resolve().parent.parent / "data"
    preferredPath = dataDir / "rxSignal.npy"
    fallbackPath = dataDir / "txs0.npy"

    if preferredPath.exists():
        logger.info(f"Loading input signal: {preferredPath}")
        return np.load(str(preferredPath))
    if fallbackPath.exists():
        logger.info(f"rxSignal.npy not found, fallback to: {fallbackPath}")
        return np.load(str(fallbackPath))

    raise FileNotFoundError("No input signal found: data/rxSignal.npy or data/txs0.npy")


def main(argv: list[str] | None = None):
    args = _parseArgs(argv)
    try:
        cfg = _buildRuntimeConfig(args)
    except Exception as e:
        logger.error(f"Config error: {e}")
        return

    logger.info(
        "Runtime config: "
        f"sampleRate={cfg.sampleRate}, scs={cfg.scs}, prbCount={cfg.prbCount}, "
        f"inputPath={cfg.inputPath}, outputPrefix={cfg.outputPrefix}, "
        f"pssWorkers={cfg.pssWorkers}, pssParallelMode={cfg.pssParallelMode}, "
        f"pssScoreMode={cfg.pssScoreMode}, pssAdaptiveIterations={cfg.pssAdaptiveIterations}, "
        f"pssLsqMinR2={cfg.pssLsqMinR2}, "
        f"sssEnable={cfg.sssEnable}, pssOnly={cfg.pssOnly}, sssSearchStartSymbol={cfg.sssSearchStartSymbol}, "
        f"sssSearchEndSymbol={cfg.sssSearchEndSymbol}, sssSearchStepSamples={cfg.sssSearchStepSamples}, "
        f"progressMode={cfg.progressMode}, progressRefreshSec={cfg.progressRefreshSec}"
    )

    logger.info("1) Load signal")
    try:
        rxSignal = _loadRxSignal(cfg.inputPath)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    logger.info(f"Signal length: {len(rxSignal)} samples")

    logger.info("2) Derive OFDM parameters")
    ssbConfig = createSsbConfig(sampleRate=cfg.sampleRate, subcarrierSpacing=cfg.scs)
    cpManager = CpManager(ssbConfig)
    logger.info(f"FFT: {ssbConfig.FftSize}, NormalCP: {ssbConfig.NormalCpLength}, LongCP: {cpManager.longCpLength}")

    logger.info("3) Build GSCN matrix from PRB+SCS")
    try:
        validGscnList, rasterEntries, meta = buildGscnMatrix(prbCount=cfg.prbCount, scsHz=cfg.scs)
    except ValueError as e:
        logger.error(str(e))
        return

    logger.info(f"Occupied bandwidth: {meta['occupiedBandwidthMHz']:.3f} MHz")
    if meta["inferredChannelBandwidthMHz"] is not None:
        logger.info(f"Inferred channel bandwidth: {meta['inferredChannelBandwidthMHz']:.1f} MHz")
    else:
        logger.info("Inferred channel bandwidth: Unknown (fallback to occupied-bandwidth threshold)")
    logger.info(f"GSCN table: {'Table 5.4.3.3-2 (3MHz)' if meta['bandwidthClass'] == 'bw3mhz' else 'Table 5.4.3.3-1 (>3MHz)'}")
    logger.info(f"SCS cases: {', '.join(meta['caseSet'])}")
    logger.info(f"GSCN candidates: {len(validGscnList)}")
    logger.info(f"GSCN range: {validGscnList[0][0]} ~ {validGscnList[-1][0]}")
    logger.info(
        f"Absolute frequency range: {validGscnList[0][1] / 1e6:.3f} MHz ~ {validGscnList[-1][1] / 1e6:.3f} MHz"
    )

    logger.info("4) Run PSS blind search")
    pssDetector = PssDetector(
        ssbConfig,
        scoreMode=cfg.pssScoreMode,
        adaptiveRefineIterations=cfg.pssAdaptiveIterations,
        maxWorkers=cfg.pssWorkers,
        parallelMode=cfg.pssParallelMode,
        progressMode=cfg.progressMode,
        progressRefreshSec=cfg.progressRefreshSec,
    )
    pssResult = pssDetector.detectPss(rxSignal, validGscnList)
    baseFreqForEstHz = float(pssResult.get("freqOffsetParabolic", pssResult["freqOffset"]))
    pssFreqEstimator = PssFreqOffsetEstimator(ssbConfig)
    pssLsqEstimation = pssFreqEstimator.estimate(
        rxSignal=rxSignal,
        nId2=int(pssResult["nId2"]),
        timingOffset=int(pssResult["timingOffset"]),
        baseFreqHz=baseFreqForEstHz,
    )
    if pssLsqEstimation is not None:
        pssResult["pssFreqOffsetEstimation"] = pssLsqEstimation

    cpFreqEstimator = CpFreqOffsetEstimator(ssbConfig)
    cpEstimation = cpFreqEstimator.estimate(
        rxSignal=rxSignal,
        timingOffset=int(pssResult["timingOffset"]),
        baseFreqHz=baseFreqForEstHz,
    )
    if cpEstimation is not None:
        pssResult["cpFreqOffsetEstimation"] = cpEstimation

    selectedEstimation = None
    useCpFallback = True
    if pssLsqEstimation is not None:
        rSquared = float(pssLsqEstimation.get("rSquared", 0.0))
        useCpFallback = rSquared < float(cfg.pssLsqMinR2)
    if useCpFallback and cpEstimation is not None:
        selectedEstimation = cpEstimation
    elif pssLsqEstimation is not None:
        selectedEstimation = pssLsqEstimation
    elif cpEstimation is not None:
        selectedEstimation = cpEstimation

    if selectedEstimation is not None:
        pssResult["freqOffsetEstimation"] = selectedEstimation

    bestGscn = int(pssResult["gscn"])
    bestAbsFreqHz = GscnRaster.getAbsoluteFrequency(bestGscn)
    bestCase = findCasesByGscn(bestGscn, rasterEntries)

    logger.info("5) Search result")
    logger.info(f"  Best GSCN: {bestGscn}")
    logger.info(f"  Best case: {bestCase}")
    logger.info(f"  Absolute frequency by GSCN: {bestAbsFreqHz / 1e6:.6f} MHz")
    logger.info(f"  Fine-search best offset (relative to DC): {pssResult['freqOffset'] / 1e6:.6f} MHz")
    if "freqOffsetParabolic" in pssResult:
        logger.info(f"  Parabolic-interp offset (relative to DC): {pssResult['freqOffsetParabolic'] / 1e6:.6f} MHz")
    adaptiveRefinement = pssResult.get("adaptiveRefinement")
    if adaptiveRefinement is not None:
        logger.info(
            "  Adaptive refinement: "
            f"maxIters={adaptiveRefinement.get('maxIterations', 0)}, "
            f"validIters={adaptiveRefinement.get('finalValidIterations', 0)}, "
            f"status={adaptiveRefinement.get('finalStatus', 'unknown')}"
        )
    logger.info(f"  Best N_ID_2: {pssResult['nId2']}")
    logger.info(f"  Best timing offset: {pssResult['timingOffset']}")
    logger.info(f"  Peak value: {pssResult['peakValue']:.6f}")
    if pssLsqEstimation is not None:
        logger.info(
            "  PSS-assisted CFO LS fit: "
            f"base={pssLsqEstimation['baseFreqHz']:.3f} Hz, "
            f"residual={pssLsqEstimation['residualFreqHz']:.3f} Hz, "
            f"refined={pssLsqEstimation['refinedFreqHz']:.3f} Hz, "
            f"R2={pssLsqEstimation['rSquared']:.6f}"
        )
        if float(pssLsqEstimation["rSquared"]) < float(cfg.pssLsqMinR2):
            logger.info(
                f"  PSS-assisted CFO LS rejected (R2={pssLsqEstimation['rSquared']:.6f} < {cfg.pssLsqMinR2:.3f}); fallback to CP."
            )

    selectedEstimation = pssResult.get("freqOffsetEstimation")
    if selectedEstimation is not None:
        method = str(selectedEstimation.get("method", "unknown"))
        if method == "cp_phase":
            logger.info(
                "  Selected CFO estimation (CP): "
                f"base={selectedEstimation['baseFreqHz']:.3f} Hz, "
                f"residual={selectedEstimation['residualFreqHz']:.3f} Hz, "
                f"refined={selectedEstimation['refinedFreqHz']:.3f} Hz, "
                f"std={selectedEstimation.get('residualStdHz', 0.0):.3f} Hz, "
                f"windows={selectedEstimation.get('symbolCountUsed', 0)}"
            )
        else:
            logger.info(
                "  Selected CFO estimation (PSS-LS): "
                f"base={selectedEstimation['baseFreqHz']:.3f} Hz, "
                f"residual={selectedEstimation['residualFreqHz']:.3f} Hz, "
                f"refined={selectedEstimation['refinedFreqHz']:.3f} Hz, "
                f"R2={selectedEstimation.get('rSquared', 0.0):.6f}"
            )
    else:
        logger.info(
            "  Selected CFO estimation: unavailable (both PSS-LS and CP estimation failed)."
        )

    sssResult = None
    if cfg.sssEnable:
        logger.info("6) Run SSS sliding search")
        sssCompFreqHz = baseFreqForEstHz
        if selectedEstimation is not None and "refinedFreqHz" in selectedEstimation:
            sssCompFreqHz = float(selectedEstimation["refinedFreqHz"])
        sssDetector = SssDetector(ssbConfig)
        try:
            sssResult = sssDetector.detectSss(
                rxSignal=rxSignal,
                pssResult=pssResult,
                freqCompHz=sssCompFreqHz,
                startSymbol=cfg.sssSearchStartSymbol,
                endSymbol=cfg.sssSearchEndSymbol,
                stepSamples=cfg.sssSearchStepSamples,
            )
            pssResult["sssResult"] = sssResult
            logger.info(
                "  SSS result: "
                f"N_ID_1={sssResult['nId1']}, N_ID_cell={sssResult['nIdCell']}, "
                f"bestOffset={sssResult['bestOffsetSamples']} samples, score={sssResult['bestScore']:.6f}"
            )
        except Exception as e:
            logger.warning(f"  SSS sliding search failed: {e}")

    logger.info("7) Save plots and result json")
    PssVisualizer.plotPssSearchValidation(
        pssResult,
        validGscnList,
        len(rxSignal),
        outputPrefix=cfg.outputPrefix,
    )
    if sssResult is not None:
        SssVisualizer.plotAndSave(sssResult, outputPrefix=cfg.outputPrefix)
    logger.info("Done")


if __name__ == "__main__":
    main()
