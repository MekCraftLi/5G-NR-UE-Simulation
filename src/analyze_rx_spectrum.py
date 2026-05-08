import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RX spectra and interference around +/-1.5 MHz")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--pattern", type=str, default="rx*.npy")
    parser.add_argument("--max-samples", type=int, default=614400)
    parser.add_argument("--nfft", type=int, default=262144)
    parser.add_argument("--target-mhz", type=float, default=1.5)
    parser.add_argument("--target-width-khz", type=float, default=200.0)
    parser.add_argument("--ssb-band-mhz", type=float, default=7.2)
    parser.add_argument("--output-prefix", type=str, default="rx_spectrum_analysis")
    return parser.parse_args(argv)


def _loadSignal(path: Path, maxSamples: int) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    if maxSamples > 0:
        x = x[:maxSamples]
    return x


def _welchPsd(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(int(nfft), len(x))
    if nfft < 1024:
        raise ValueError("nfft too small for spectrum analysis")
    hop = nfft // 2
    if len(x) < nfft:
        starts = [0]
        segLen = len(x)
        win = np.hanning(segLen).astype(np.float32)
        spec = np.fft.fftshift(np.fft.fft(x * win, nfft))
        psd = np.abs(spec) ** 2 / max(float(np.sum(win ** 2)), 1e-12)
    else:
        starts = list(range(0, len(x) - nfft + 1, hop))
        win = np.hanning(nfft).astype(np.float32)
        acc = np.zeros(nfft, dtype=np.float64)
        for start in starts:
            seg = x[start:start + nfft]
            spec = np.fft.fftshift(np.fft.fft(seg * win, nfft))
            acc += np.abs(spec) ** 2 / max(float(np.sum(win ** 2)), 1e-12)
        psd = acc / max(len(starts), 1)
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    return freq.astype(np.float64), np.asarray(psd, dtype=np.float64)


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def _bandStats(freq: np.ndarray, psd: np.ndarray, centerHz: float, widthHz: float) -> dict:
    mask = np.abs(freq - float(centerHz)) <= float(widthHz) / 2.0
    if not np.any(mask):
        return {"meanDb": None, "maxDb": None, "power": 0.0}
    band = psd[mask]
    return {
        "meanDb": float(_db(np.mean(band))),
        "maxDb": float(_db(np.max(band))),
        "power": float(np.mean(band)),
    }


def _analyzeOne(path: Path, fs: float, maxSamples: int, nfft: int, targetHz: float, targetWidthHz: float, ssbBandHz: float) -> tuple[dict, np.ndarray, np.ndarray]:
    x = _loadSignal(path, maxSamples)
    freq, psd = _welchPsd(x, fs, nfft)
    psdDb = _db(psd)

    exclude = np.abs(freq) < ssbBandHz / 2.0
    outer = ~exclude
    noiseFloorDb = float(np.median(psdDb[outer])) if np.any(outer) else float(np.median(psdDb))

    pos = _bandStats(freq, psd, targetHz, targetWidthHz)
    neg = _bandStats(freq, psd, -targetHz, targetWidthHz)
    center = _bandStats(freq, psd, 0.0, targetWidthHz)
    ssbMask = np.abs(freq) <= ssbBandHz / 2.0
    targetMask = (
        (np.abs(freq - targetHz) <= targetWidthHz / 2.0)
        | (np.abs(freq + targetHz) <= targetWidthHz / 2.0)
    )
    restInBand = ssbMask & ~targetMask
    ssbPower = float(np.mean(psd[ssbMask])) if np.any(ssbMask) else 0.0
    restPower = float(np.mean(psd[restInBand])) if np.any(restInBand) else 0.0
    targetPower = float(np.mean(psd[targetMask])) if np.any(targetMask) else 0.0
    peakIdx = int(np.argmax(psd))

    stats = {
        "file": str(path),
        "samplesUsed": int(len(x)),
        "timePowerDb": float(_db(np.mean(np.abs(x) ** 2))),
        "noiseFloorDb": noiseFloorDb,
        "peakFreqHz": float(freq[peakIdx]),
        "peakFreqMHz": float(freq[peakIdx] / 1e6),
        "peakDb": float(psdDb[peakIdx]),
        "peakAboveNoiseDb": float(psdDb[peakIdx] - noiseFloorDb),
        "targetPositive": pos,
        "targetNegative": neg,
        "targetPositiveAboveNoiseDb": None if pos["meanDb"] is None else float(pos["meanDb"] - noiseFloorDb),
        "targetNegativeAboveNoiseDb": None if neg["meanDb"] is None else float(neg["meanDb"] - noiseFloorDb),
        "centerMeanDb": center["meanDb"],
        "targetVsRestInBandDb": float(_db(targetPower / max(restPower, 1e-30))),
        "targetVsSsbBandAvgDb": float(_db(targetPower / max(ssbPower, 1e-30))),
        "ssbBandAvgDb": float(_db(ssbPower)),
        "restInBandAvgDb": float(_db(restPower)),
    }
    return stats, freq, psdDb


def _plotOverview(results: list[dict], spectra: dict[str, tuple[np.ndarray, np.ndarray]], outputPrefix: str) -> Path:
    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{outputPrefix}_overview.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), constrained_layout=True)

    for item in results[:6]:
        freq, psdDb = spectra[item["file"]]
        axes[0].plot(freq / 1e6, psdDb, linewidth=0.8, alpha=0.85, label=Path(item["file"]).name)
    axes[0].axvline(1.5, color="red", linestyle="--", linewidth=0.9)
    axes[0].axvline(-1.5, color="red", linestyle="--", linewidth=0.9)
    axes[0].set_xlim(-6, 6)
    axes[0].set_xlabel("Frequency (MHz)")
    axes[0].set_ylabel("PSD (dB, relative)")
    axes[0].set_title("RX Spectrum Comparison")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)

    names = [Path(x["file"]).name for x in results]
    values = [x["targetVsRestInBandDb"] for x in results]
    axes[1].bar(range(len(names)), values)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("+/-1.5 MHz vs rest in SSB band (dB)")
    axes[1].set_title("Interference Indicator")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.savefig(figPath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figPath


def main(argv: list[str] | None = None) -> None:
    args = _parseArgs(argv)
    dataDir = Path(args.data_dir)
    paths = sorted(dataDir.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No files match {dataDir / args.pattern}")

    results = []
    spectra = {}
    for path in paths:
        try:
            stats, freq, psdDb = _analyzeOne(
                path=path,
                fs=float(args.sample_rate),
                maxSamples=int(args.max_samples),
                nfft=int(args.nfft),
                targetHz=float(args.target_mhz) * 1e6,
                targetWidthHz=float(args.target_width_khz) * 1e3,
                ssbBandHz=float(args.ssb_band_mhz) * 1e6,
            )
        except Exception as exc:
            logger.warning("Skip %s: %s", path, exc)
            continue
        results.append(stats)
        spectra[stats["file"]] = (freq, psdDb)
        logger.info(
            "%s: peak=%.3f MHz %.1f dB above floor, +/-%.1fMHz vs rest=%.1f dB",
            path.name,
            stats["peakFreqMHz"],
            stats["peakAboveNoiseDb"],
            float(args.target_mhz),
            stats["targetVsRestInBandDb"],
        )

    results.sort(key=lambda x: x["targetVsRestInBandDb"])
    figPath = _plotOverview(results, spectra, args.output_prefix)

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    outPath = outDir / f"{args.output_prefix}_result.json"
    outPath.write_text(json.dumps({
        "sampleRate": float(args.sample_rate),
        "targetMHz": float(args.target_mhz),
        "targetWidthKHz": float(args.target_width_khz),
        "ssbBandMHz": float(args.ssb_band_mhz),
        "resultsSortedByTargetInterference": results,
        "overviewFigure": str(figPath.resolve()),
    }, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info("Saved result: %s", outPath.resolve())
    logger.info("Saved overview: %s", figPath.resolve())


if __name__ == "__main__":
    main()
