import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parseCsvFloats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parseCsvInts(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep sample-rate and plot-NFFT spectrum parameters")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rates-mhz", type=str, default="30.72,61.44,122.88,245.76")
    parser.add_argument("--nffts", type=str, default="65536,131072,262144,524288")
    parser.add_argument("--max-samples", type=int, default=614400)
    parser.add_argument("--segment-samples", type=int, default=65536, help="Fixed time-domain samples used for every NFFT")
    parser.add_argument("--target-mhz", type=float, default=1.5)
    parser.add_argument("--target-width-khz", type=float, default=200.0)
    parser.add_argument("--xlim-mhz", type=float, default=8.0)
    parser.add_argument("--output-prefix", type=str, default="rx_spectrum_param_sweep")
    return parser.parse_args()


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-30))


def _directFft(x: np.ndarray, fs: float, nfft: int, segmentSamples: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = int(nfft)
    segmentSamples = min(int(segmentSamples), len(x))
    if nfft < segmentSamples:
        raise ValueError(f"NFFT={nfft} is smaller than fixed segment length={segmentSamples}")
    segment = x[:segmentSamples]
    spec = np.fft.fftshift(np.fft.fft(segment, n=nfft))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / fs))
    magDb = _db(np.abs(spec))
    return freq, magDb - float(np.median(magDb))


def _bandMean(freq: np.ndarray, magRelDb: np.ndarray, centerHz: float, widthHz: float) -> float:
    mask = np.abs(freq - centerHz) <= widthHz / 2.0
    return float(np.mean(magRelDb[mask])) if np.any(mask) else float("nan")


def main() -> None:
    args = _parseArgs()
    path = Path(args.input_path)
    x = np.asarray(np.load(path), dtype=np.complex64).reshape(-1)
    if args.max_samples and args.max_samples > 0:
        x = x[: int(args.max_samples)]

    sampleRatesMHz = _parseCsvFloats(args.sample_rates_mhz)
    nffts = _parseCsvInts(args.nffts)
    targetHz = float(args.target_mhz) * 1e6
    widthHz = float(args.target_width_khz) * 1e3

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}_grid.png"
    rows = len(sampleRatesMHz)
    cols = len(nffts)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.2 * rows), sharey=True, constrained_layout=True)
    if rows == 1:
        axes = np.asarray([axes])
    if cols == 1:
        axes = axes[:, None]

    results = []
    for r, fsMHz in enumerate(sampleRatesMHz):
        fs = fsMHz * 1e6
        for c, nfft in enumerate(nffts):
            try:
                freq, magRelDb = _directFft(x, fs, nfft, int(args.segment_samples))
            except ValueError as exc:
                ax = axes[r, c]
                ax.text(0.5, 0.5, str(exc), transform=ax.transAxes, ha="center", va="center", wrap=True)
                ax.set_axis_off()
                continue
            ax = axes[r, c]
            ax.plot(freq / 1e6, magRelDb, linewidth=0.45)
            for center in (-float(args.target_mhz), float(args.target_mhz)):
                ax.axvline(center, color="red", linestyle="--", linewidth=0.7)
                ax.axvspan(center - widthHz / 2e6, center + widthHz / 2e6, color="red", alpha=0.12)
            ax.set_xlim(-float(args.xlim_mhz), float(args.xlim_mhz))
            ax.grid(True, alpha=0.25)
            ax.set_title(f"fs={fsMHz:g}MHz, NFFT={nfft}\nΔf={fs/nfft:.1f}Hz")
            if c == 0:
                ax.set_ylabel("20log10|FFT| rel median (dB)")
            if r == rows - 1:
                ax.set_xlabel("Frequency (MHz)")

            peakIdx = int(np.argmax(magRelDb))
            ssbMask = np.abs(freq) <= 3.6e6
            targetMask = (np.abs(freq - targetHz) <= widthHz / 2.0) | (np.abs(freq + targetHz) <= widthHz / 2.0)
            restMask = ssbMask & ~targetMask
            targetMean = float(np.mean(magRelDb[targetMask])) if np.any(targetMask) else float("nan")
            restMean = float(np.mean(magRelDb[restMask])) if np.any(restMask) else float("nan")
            results.append({
                "sampleRateMHz": float(fsMHz),
                "nfft": int(nfft),
                "binResolutionHz": float(fs / nfft),
                "peakFreqMHz": float(freq[peakIdx] / 1e6),
                "peakRelDb": float(magRelDb[peakIdx]),
                "negativeTargetMeanRelDb": _bandMean(freq, magRelDb, -targetHz, widthHz),
                "positiveTargetMeanRelDb": _bandMean(freq, magRelDb, targetHz, widthHz),
                "targetMeanRelDb": targetMean,
                "restSsbMeanRelDb": restMean,
                "targetVsRestDb": float(targetMean - restMean),
            })

    fig.suptitle(f"{path.name} direct FFT parameter sweep", fontsize=15)
    fig.savefig(figPath, dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Best plotting parameters: high resolution without excessive zero-padding ambiguity.
    ranked = sorted(results, key=lambda x: (abs(x["targetVsRestDb"]), x["binResolutionHz"]))
    outPath = outDir / f"{args.output_prefix}_result.json"
    outPath.write_text(json.dumps({
        "inputPath": str(path.resolve()),
        "samplesUsed": int(len(x)),
        "fixedSegmentSamples": int(min(int(args.segment_samples), len(x))),
        "figure": str(figPath.resolve()),
        "results": results,
        "note": "Sample rate changes the frequency-axis labeling; NFFT changes plot resolution. Spectrum alone cannot prove true sample rate.",
        "recommendedForPlotting": ranked[0],
    }, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps({
        "figure": str(figPath.resolve()),
        "result": str(outPath.resolve()),
        "recommendedForPlotting": ranked[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
