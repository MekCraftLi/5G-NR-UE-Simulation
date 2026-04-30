import argparse
import json
import logging
from pathlib import Path

import numpy as np

from common.config import createSsbConfig
from sss.sssDetector import SssDetector
from sss.sssVisualizer import SssVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SSS-only sliding search from cached PSS result")
    parser.add_argument("--sample-rate", type=float, default=30.72e6, help="Sample rate in Hz")
    parser.add_argument("--scs", type=int, default=30000, help="Subcarrier spacing in Hz")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy", help="Input .npy signal path")
    parser.add_argument("--pss-result", type=str, required=True, help="Path to cached PSS result JSON")
    parser.add_argument("--output-prefix", type=str, default="", help="Output file prefix")
    parser.add_argument("--sss-start-symbol", type=float, default=0.5, help="SSS search start in OFDM-symbol units")
    parser.add_argument("--sss-end-symbol", type=float, default=2.5, help="SSS search end in OFDM-symbol units")
    parser.add_argument("--sss-step-samples", type=int, default=1, help="SSS search step in samples")
    return parser.parse_args(argv)


def _loadRxSignal(inputPath: str) -> np.ndarray:
    path = Path(inputPath)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input signal file not found: {path}")
    logger.info(f"Loading input signal: {path}")
    return np.load(str(path))


def _loadPssSnapshot(pssResultPath: str) -> dict:
    path = Path(pssResultPath)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"PSS result file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    finalBest = data.get("finalBest")
    if not isinstance(finalBest, dict):
        raise ValueError("Invalid PSS JSON: missing finalBest")

    pss = {
        "nId2": int(finalBest["nId2"]),
        "timingOffset": int(finalBest["timingOffset"]),
        "freqOffset": float(finalBest["freqOffsetHz"]),
        "freqOffsetParabolic": float(finalBest.get("freqOffsetParabolicHz", finalBest["freqOffsetHz"])),
    }

    estimation = data.get("selectedFreqOffsetEstimation")
    if not isinstance(estimation, dict):
        estimation = data.get("freqOffsetEstimation")
    if isinstance(estimation, dict) and "refinedFreqHz" in estimation:
        pss["freqOffsetEstimation"] = {
            "method": str(estimation.get("method", "unknown")),
            "refinedFreqHz": float(estimation["refinedFreqHz"]),
        }

    return pss


def _inferPrefix(pssResultPath: str) -> str:
    stem = Path(pssResultPath).stem
    suffix = "_pss_search_result"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return f"{stem}_sss"


def main(argv: list[str] | None = None):
    args = _parseArgs(argv)
    if args.sss_step_samples <= 0:
        raise ValueError("sss-step-samples must be > 0")
    if args.sss_end_symbol <= args.sss_start_symbol:
        raise ValueError("sss-end-symbol must be > sss-start-symbol")

    logger.info("1) Load input signal")
    rxSignal = _loadRxSignal(args.input_path)
    logger.info(f"Signal length: {len(rxSignal)} samples")

    logger.info("2) Load cached PSS result")
    pssResult = _loadPssSnapshot(args.pss_result)
    logger.info(
        "Cached PSS anchor: "
        f"N_ID_2={pssResult['nId2']}, timing={pssResult['timingOffset']}, "
        f"freq={pssResult.get('freqOffsetParabolic', pssResult['freqOffset']):.3f} Hz"
    )

    logger.info("3) Build SSB config and run SSS-only sliding search")
    ssbConfig = createSsbConfig(sampleRate=float(args.sample_rate), subcarrierSpacing=int(args.scs))
    sssDetector = SssDetector(ssbConfig)
    sssResult = sssDetector.detectSss(
        rxSignal=rxSignal,
        pssResult=pssResult,
        startSymbol=float(args.sss_start_symbol),
        endSymbol=float(args.sss_end_symbol),
        stepSamples=int(args.sss_step_samples),
    )

    logger.info(
        "SSS result: "
        f"N_ID_1={sssResult['nId1']}, N_ID_cell={sssResult['nIdCell']}, "
        f"bestOffset={sssResult['bestOffsetSamples']}, score={sssResult['bestScore']:.6f}"
    )

    logger.info("4) Save SSS outputs")
    prefix = str(args.output_prefix).strip() or _inferPrefix(args.pss_result)
    SssVisualizer.plotAndSave(sssResult, outputPrefix=prefix)
    logger.info("Done")


if __name__ == "__main__":
    main()
