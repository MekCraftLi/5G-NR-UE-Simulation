import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.bandwidthEstimator import detectBandwidth, inferScs
from common.config import createSsbConfig
from common.cpManager import CpManager
from pss.pssBasebandSearcher import PssBasebandSearcher
from pss.cpFreqOffsetEstimator import CpFreqOffsetEstimator
from pss.pssFreqOffsetEstimator import PssFreqOffsetEstimator
from pss.pssTemplateFactory import generatePssSequence
from pss.pssVisualizer import PssVisualizer
from pss.scsDetectVisualizer import plotScsDetection
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
    pssCoarseStepHz: float = 1000.0
    pssMediumStepHz: float = 100.0
    pssFineStepHz: float = 15.0
    pssUseGpu: bool = True
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
        "--pss-coarse-step",
        type=float,
        default=None,
        help="PSS adaptive search coarse step (Hz), default 1000",
    )
    parser.add_argument(
        "--pss-medium-step",
        type=float,
        default=None,
        help="PSS adaptive search medium step (Hz), default 100",
    )
    parser.add_argument(
        "--pss-fine-step",
        type=float,
        default=None,
        help="PSS adaptive search fine step (Hz), default 15",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=None,
        help="Use GPU for PSS search (default: auto-detect)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", default=None,
        help="Force CPU for PSS search",
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
        "pssCoarseStepHz": 1000.0,
        "pssMediumStepHz": 100.0,
        "pssFineStepHz": 15.0,
        "pssUseGpu": True,
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
    pssCoarseStepHz = float(
        args.pss_coarse_step
        if args.pss_coarse_step is not None
        else _pick(fileCfg, "pss_coarse_step_hz", "pssCoarseStepHz", default=defaults["pssCoarseStepHz"])
    )
    pssMediumStepHz = float(
        args.pss_medium_step
        if args.pss_medium_step is not None
        else _pick(fileCfg, "pss_medium_step_hz", "pssMediumStepHz", default=defaults["pssMediumStepHz"])
    )
    pssFineStepHz = float(
        args.pss_fine_step
        if args.pss_fine_step is not None
        else _pick(fileCfg, "pss_fine_step_hz", "pssFineStepHz", default=defaults["pssFineStepHz"])
    )
    pssUseGpu = bool(
        _pick(fileCfg, "pss_use_gpu", "pssUseGpu", default=defaults["pssUseGpu"])
    )
    if args.use_gpu:
        pssUseGpu = True
    if args.no_gpu:
        pssUseGpu = False
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
    if pssCoarseStepHz <= 0:
        raise ValueError("pssCoarseStepHz must be > 0")
    if pssMediumStepHz <= 0:
        raise ValueError("pssMediumStepHz must be > 0")
    if pssFineStepHz <= 0:
        raise ValueError("pssFineStepHz must be > 0")
    if not (pssFineStepHz < pssMediumStepHz < pssCoarseStepHz):
        raise ValueError("step sizes must satisfy: fine < medium < coarse")
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
        pssCoarseStepHz=pssCoarseStepHz,
        pssMediumStepHz=pssMediumStepHz,
        pssFineStepHz=pssFineStepHz,
        pssUseGpu=pssUseGpu,
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
        f"pssCoarseStepHz={cfg.pssCoarseStepHz}, pssMediumStepHz={cfg.pssMediumStepHz}, "
        f"pssFineStepHz={cfg.pssFineStepHz}, pssUseGpu={cfg.pssUseGpu}, "
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

    logger.info("3) Detect SCS and signal bandwidth via CP autocorrelation")
    bwEstimate = detectBandwidth(rxSignal, cfg.sampleRate)
    scsCandidates = inferScs(bwEstimate)

    detectedScs = bwEstimate.scsHz
    scsMatch = (int(cfg.scs) == int(detectedScs))
    # 搜索范围: SSB 带宽 = 240 子载波 × SCS
    freqMinHz = -bwEstimate.bandwidthHz / 2
    freqMaxHz = +bwEstimate.bandwidthHz / 2

    logger.info(
        f"Detected: SCS={detectedScs/1000:.0f}kHz, FFT={bwEstimate.fftSize}, "
        f"CP={bwEstimate.cpLength}, SSB BW={bwEstimate.bandwidthHz/1e6:.3f}MHz, "
        f"confidence={bwEstimate.confidence:.2f}"
    )
    logger.info(f"Search range: [{freqMinHz/1e6:.3f}, {freqMaxHz/1e6:.3f}] MHz")

    def _richWarn(title, body, style="yellow"):
        try:
            from rich.console import Console
            from rich.panel import Panel
            Console(stderr=True).print(Panel.fit(body, title=title, border_style=style))
        except ImportError:
            logger.warning(f"{title}: {body}")

    if bwEstimate.confidence < 0.5:
        _richWarn(
            "SCS Detection",
            f"[dim]CP 自相关置信度仅 {bwEstimate.confidence:.2f}，"
            f"SCS 推断可能不准确[/dim]\n"
            f"检测值: SCS={detectedScs/1000:.0f}kHz, "
            f"SSB BW={bwEstimate.bandwidthHz/1e6:.2f}MHz"
        )

    if not scsMatch:
        _richWarn(
            "SCS Validation",
            f"检测 SCS: [bold green]{detectedScs/1000:.0f} kHz[/bold green] "
            f"(SSB BW={bwEstimate.bandwidthHz/1e6:.2f} MHz)\n"
            f"指定 SCS: [bold red]{cfg.scs/1000:.0f} kHz[/bold red] "
            f"— [bold yellow]不匹配![/bold yellow]",
        )
    else:
        logger.info(f"SCS {cfg.scs/1000:.0f}kHz: matches detected SCS")

    plotScsDetection(rxSignal, cfg.sampleRate, outputPrefix=cfg.outputPrefix)

    logger.info("4) Run PSS adaptive baseband frequency search")
    pssSearcher = PssBasebandSearcher(
        config=ssbConfig,
        freqMinHz=freqMinHz,
        freqMaxHz=freqMaxHz,
        workers=cfg.pssWorkers,
        useGpu=cfg.pssUseGpu,
    )
    pssResult = pssSearcher.searchAdaptive(
        signal=rxSignal,
        coarseStepHz=cfg.pssCoarseStepHz,
        mediumStepHz=cfg.pssMediumStepHz,
        fineStepHz=cfg.pssFineStepHz,
        subfineIterations=cfg.pssAdaptiveIterations,
    )
    baseFreqForEstHz = float(pssResult.get("freqOffsetParabolic", pssResult["freqOffset"]))

    # ── 论文 FFO 估计 (Tuninato 2023, Eq 15-17): PSS 半符号相关 ──
    nId2 = int(pssResult["nId2"])
    timingOffset = int(pssResult["timingOffset"])
    fftSize = int(ssbConfig.FftSize)
    cpLen = int(ssbConfig.NormalCpLength)
    halfLen = fftSize // 2
    sampleRate = float(ssbConfig.SampleRate)

    pssSeq = generatePssSequence(nId2)
    pssFreq = np.zeros(fftSize, dtype=np.complex64)
    pssFreq[fftSize // 2 - 63:fftSize // 2 + 64] = pssSeq
    pssTimeTemplate = np.fft.ifft(np.fft.ifftshift(pssFreq)).astype(np.complex64)
    pssTimeTemplate /= np.linalg.norm(pssTimeTemplate)

    usefulStart = timingOffset + cpLen
    globalIdx = np.arange(usefulStart, usefulStart + fftSize, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * baseFreqForEstHz * globalIdx / sampleRate).astype(np.complex64)
    pssRx = (np.asarray(rxSignal[usefulStart:usefulStart + fftSize], dtype=np.complex64) * phase).astype(np.complex64)

    C0 = np.vdot(pssRx[:halfLen], pssTimeTemplate[:halfLen])
    C1 = np.vdot(pssRx[halfLen:], pssTimeTemplate[halfLen:])
    deltaTheta = float(np.angle(C1 * np.conjugate(C0)))
    ffoHz = deltaTheta * sampleRate / (np.pi * fftSize)
    totalCfoHz = baseFreqForEstHz + ffoHz

    pssFfoResult = {
        "method": "pss_half_symbol_ffo",
        "C0": complex(C0),
        "C1": complex(C1),
        "deltaThetaRad": deltaTheta,
        "ffoHz": ffoHz,
        "totalCfoHz": totalCfoHz,
        "baseFreqHz": baseFreqForEstHz,
    }
    pssResult["pssFfoEstimation"] = pssFfoResult

    # ── PSS LS 相位拟合: 频偏精估 + R² 质量指标 ──
    lsEstimator = PssFreqOffsetEstimator(ssbConfig)
    lsResult = lsEstimator.estimate(
        rxSignal=rxSignal,
        nId2=nId2,
        timingOffset=timingOffset,
        baseFreqHz=baseFreqForEstHz,
    )
    if lsResult is not None:
        pssResult["pssFreqOffsetEstimation"] = lsResult

    # ── CP 相位频偏估计 ──
    cpEstimator = CpFreqOffsetEstimator(ssbConfig)
    cpResult = cpEstimator.estimate(
        rxSignal=rxSignal,
        timingOffset=timingOffset,
        baseFreqHz=baseFreqForEstHz,
    )
    if cpResult is not None:
        pssResult["cpFreqOffsetEstimation"] = cpResult
    else:
        logger.warning("CP frequency offset estimation returned None (no valid CP windows)")

    logger.info("5) Search result")
    logger.info(f"  Mode: adaptive baseband (no GSCN)")
    logger.info(f"  Best baseband frequency: {pssResult['freqOffsetParabolic'] / 1e3:.3f} kHz")
    freqSearch = pssResult.get("freqSearch", {})
    if isinstance(freqSearch, dict):
        for name in ["coarse", "medium", "fine"]:
            p = freqSearch.get("passes", {}).get(name, {})
            if p:
                logger.info(
                    f"  {name}: step={p.get('stepHz', 0):.0f} Hz, "
                    f"bestFreq={p.get('bestFreqHz', 0) / 1e3:.3f} kHz, "
                    f"bestNId2={p.get('bestNId2', '?')}"
                )
    ar = pssResult.get("adaptiveRefinement", {})
    if ar:
        logger.info(
            f"  Sub-fine: iters={ar.get('finalValidIterations', 0)}/{ar.get('maxIterations', 0)}, "
            f"status={ar.get('finalStatus', '?')}"
        )
    logger.info(f"  Best N_ID_2: {pssResult['nId2']}")
    logger.info(f"  Best timing offset: {pssResult['timingOffset']}")
    logger.info(f"  Peak value: {pssResult['peakValue']:.6f}")
    logger.info(
        f"  Half-symbol FFO: C0={np.abs(C0):.0f}, C1={np.abs(C1):.0f}, "
        f"Δθ={np.degrees(deltaTheta):.2f}°, "
        f"FFO={ffoHz:.2f} Hz, totalCFO={totalCfoHz:.2f} Hz"
    )
    if lsResult is not None:
        logger.info(
            f"  LS phase fit: R²={lsResult['rSquared']:.6f}, "
            f"residual={lsResult['residualFreqHz']:.2f} Hz, "
            f"refined={lsResult['refinedFreqHz']:.2f} Hz, "
            f"validPts={lsResult['validSampleCount']}/{lsResult['sampleCount']}"
        )
    if cpResult is not None:
        logger.info(
            f"  CP phase: residual={cpResult['residualFreqHz']:.2f} Hz, "
            f"std={cpResult['residualStdHz']:.2f} Hz, "
            f"windows={cpResult['symbolCountUsed']}, "
            f"coherence={np.mean(cpResult['coherence']):.3f}"
        )

    sssResult = None
    sssCompFreqHz = totalCfoHz
    if cfg.sssEnable:
        logger.info("6) Run SSS sliding search")
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
        [],  # validGscnList (unused, kept for API compatibility)
        len(rxSignal),
        outputPrefix=cfg.outputPrefix,
    )
    if sssResult is not None:
        SssVisualizer.plotAndSave(sssResult, outputPrefix=cfg.outputPrefix)
    logger.info("Done")


if __name__ == "__main__":
    main()
