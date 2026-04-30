import logging
import os
import argparse

import numpy as np

from common.config import createSsbConfig
from common.cpManager import CpManager
from common.gscn import GscnRaster
from common.gscnRasterTable import buildGscnMatrix, findCasesByGscn
from pss.cpFreqOffsetEstimator import CpFreqOffsetEstimator
from pss.pssDetector import PssDetector
from pss.pssFreqOffsetEstimator import PssFreqOffsetEstimator
from pss.pssVisualizer import PssVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TX reference PSS blind-search flow")
    parser.add_argument("--sample-rate", type=float, default=30.72e6, help="Sample rate in Hz")
    parser.add_argument("--scs", type=int, default=30000, help="Subcarrier spacing in Hz")
    parser.add_argument("--prb-count", type=int, default=51, help="PRB count used for GSCN matrix")
    parser.add_argument("--known-baseband", action="store_true", help="Use single candidate (gscn=0, freq=0)")
    parser.add_argument("--output-prefix", type=str, default="tx", help="Output prefix")
    parser.add_argument("--pss-workers", type=int, default=None, help="PSS detector worker count")
    parser.add_argument("--pss-parallel-mode", type=str, default="process", choices=["process", "thread"])
    parser.add_argument("--pss-lsq-min-r2", type=float, default=0.8, help="Minimum R^2 to accept PSS-assisted CFO LS fit")
    parser.add_argument("--progress-mode", type=str, default="auto", choices=["auto", "rich", "log"])
    parser.add_argument("--progress-refresh", type=float, default=0.7, help="Progress refresh interval in seconds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = _parseArgs(argv)
    sampleRate = float(args.sample_rate)
    scs = int(args.scs)

    logger.info("1) Initialize TX reference flow")
    txPath = os.path.join(os.path.dirname(__file__), "..", "data", "txs0.npy")
    try:
        txSignal = np.load(txPath)
    except FileNotFoundError:
        logger.error(f"TX reference signal not found: {txPath}")
        return
    logger.info(f"TX signal length: {len(txSignal)} samples")

    logger.info("2) Derive OFDM parameters")
    ssbConfig = createSsbConfig(sampleRate=sampleRate, subcarrierSpacing=scs)
    cpManager = CpManager(ssbConfig)
    logger.info(f"FFT: {ssbConfig.FftSize}")
    logger.info(f"Normal CP: {ssbConfig.NormalCpLength}")
    logger.info(f"Long CP: {cpManager.longCpLength}")

    logger.info("3) Build TX carrier candidates")
    rasterEntries = []
    if args.known_baseband:
        validFreqList = [(0, 0.0)]
        logger.info("Known baseband mode enabled: use single candidate (GSCN=0, freq=0)")
    else:
        validFreqList, rasterEntries, meta = buildGscnMatrix(prbCount=int(args.prb_count), scsHz=scs)
        logger.info(f"Occupied bandwidth: {meta['occupiedBandwidthMHz']:.3f} MHz")
        if meta["inferredChannelBandwidthMHz"] is not None:
            logger.info(f"Inferred channel bandwidth: {meta['inferredChannelBandwidthMHz']:.1f} MHz")
        logger.info(f"SCS cases: {', '.join(meta['caseSet'])}")
        logger.info(f"GSCN candidates: {len(validFreqList)}")
        logger.info(
            f"GSCN range: {validFreqList[0][0]} ~ {validFreqList[-1][0]}, "
            f"absolute freq: {validFreqList[0][1] / 1e6:.3f} ~ {validFreqList[-1][1] / 1e6:.3f} MHz"
        )

    logger.info("4) Run TX PSS search")
    pssDetector = PssDetector(
        ssbConfig,
        scoreMode="ncc",
        maxWorkers=args.pss_workers,
        parallelMode=str(args.pss_parallel_mode),
        progressMode=str(args.progress_mode),
        progressRefreshSec=float(args.progress_refresh),
    )
    pssResult = pssDetector.detectPss(txSignal, validFreqList)
    baseFreqForEstHz = float(pssResult.get("freqOffsetParabolic", pssResult["freqOffset"]))

    pssFreqEstimator = PssFreqOffsetEstimator(ssbConfig)
    pssLsqEstimation = pssFreqEstimator.estimate(
        rxSignal=txSignal,
        nId2=int(pssResult["nId2"]),
        timingOffset=int(pssResult["timingOffset"]),
        baseFreqHz=baseFreqForEstHz,
    )
    if pssLsqEstimation is not None:
        pssResult["pssFreqOffsetEstimation"] = pssLsqEstimation

    cpFreqEstimator = CpFreqOffsetEstimator(ssbConfig)
    cpEstimation = cpFreqEstimator.estimate(
        rxSignal=txSignal,
        timingOffset=int(pssResult["timingOffset"]),
        baseFreqHz=baseFreqForEstHz,
    )
    if cpEstimation is not None:
        pssResult["cpFreqOffsetEstimation"] = cpEstimation

    selectedEstimation = None
    useCpFallback = True
    if pssLsqEstimation is not None:
        useCpFallback = float(pssLsqEstimation.get("rSquared", 0.0)) < float(args.pss_lsq_min_r2)
    if useCpFallback and cpEstimation is not None:
        selectedEstimation = cpEstimation
    elif pssLsqEstimation is not None:
        selectedEstimation = pssLsqEstimation
    elif cpEstimation is not None:
        selectedEstimation = cpEstimation
    if selectedEstimation is not None:
        pssResult["freqOffsetEstimation"] = selectedEstimation

    logger.info("5) TX PSS result")
    if not args.known_baseband:
        bestGscn = int(pssResult["gscn"])
        bestAbsFreqHz = GscnRaster.getAbsoluteFrequency(bestGscn)
        bestCase = findCasesByGscn(bestGscn, rasterEntries)
        logger.info(f"  BestCase={bestCase}")
        logger.info(f"  AbsoluteFreqByGSCN={bestAbsFreqHz / 1e6:.6f} MHz")
    logger.info(f"  GSCN={pssResult['gscn']}")
    logger.info(f"  FreqOffset={pssResult['freqOffset']:.1f} Hz")
    if "freqOffsetParabolic" in pssResult:
        logger.info(f"  FreqOffsetParabolic={pssResult['freqOffsetParabolic']:.3f} Hz")
    logger.info(f"  N_ID_2={pssResult['nId2']}")
    logger.info(f"  TimingOffset={pssResult['timingOffset']}")
    logger.info(f"  PeakValue={pssResult['peakValue']:.6f}")
    if pssLsqEstimation is not None:
        logger.info(
            "  PSS CFO-LS: "
            f"base={pssLsqEstimation['baseFreqHz']:.1f} Hz, "
            f"residual={pssLsqEstimation['residualFreqHz']:.3f} Hz, "
            f"refined={pssLsqEstimation['refinedFreqHz']:.3f} Hz, "
            f"R2={pssLsqEstimation['rSquared']:.6f}"
        )
        if float(pssLsqEstimation["rSquared"]) < float(args.pss_lsq_min_r2):
            logger.info(
                f"  PSS CFO-LS rejected (R2={pssLsqEstimation['rSquared']:.6f} < {float(args.pss_lsq_min_r2):.3f}); fallback to CP."
            )
    selected = pssResult.get("freqOffsetEstimation")
    if selected is not None:
        method = str(selected.get("method", "unknown"))
        if method == "cp_phase":
            logger.info(
                "  Selected CFO (CP): "
                f"base={selected['baseFreqHz']:.1f} Hz, residual={selected['residualFreqHz']:.3f} Hz, "
                f"refined={selected['refinedFreqHz']:.3f} Hz, std={selected.get('residualStdHz', 0.0):.3f} Hz, "
                f"windows={selected.get('symbolCountUsed', 0)}"
            )
        else:
            logger.info(
                "  Selected CFO (PSS-LS): "
                f"base={selected['baseFreqHz']:.1f} Hz, residual={selected['residualFreqHz']:.3f} Hz, "
                f"refined={selected['refinedFreqHz']:.3f} Hz, R2={selected.get('rSquared', 0.0):.6f}"
            )
    for item in pssResult.get("nId2BestResults", []):
        logger.info(
            f"  [N_ID_2={item['nId2']}] "
            f"BestFreq={item['freqOffset']:.1f} Hz, "
            f"TimingOffset={item['timingOffset']}, "
            f"PeakValue={item['peakValue']:.6f}"
        )

    logger.info("6) Save plots and json")
    PssVisualizer.plotPssSearchValidation(
        pssResult,
        validFreqList,
        len(txSignal),
        outputPrefix=str(args.output_prefix),
    )
    logger.info("TX reference flow done")


if __name__ == "__main__":
    main()
