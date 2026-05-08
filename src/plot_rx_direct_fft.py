import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct single-FFT spectrum plot for RX signal")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--nfft", type=int, default=0, help="0 means use full signal length")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use full signal")
    parser.add_argument("--window", choices=["none", "hann"], default="none")
    parser.add_argument("--target-mhz", type=float, default=1.5)
    parser.add_argument("--target-width-khz", type=float, default=200.0)
    parser.add_argument("--output-prefix", type=str, default="rx_direct_fft")
    return parser.parse_args()


def _db20(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-30))


def _bandMean(freq: np.ndarray, mag: np.ndarray, centerHz: float, widthHz: float) -> float:
    mask = np.abs(freq - centerHz) <= widthHz / 2.0
    return float(np.mean(mag[mask])) if np.any(mask) else 0.0


def main() -> None:
    args = _parseArgs()
    path = Path(args.input_path)
    x = np.asarray(np.load(path), dtype=np.complex64).reshape(-1)
    if args.max_samples and args.max_samples > 0:
        x = x[:args.max_samples]
    nfft = int(args.nfft) if int(args.nfft) > 0 else len(x)
    if args.window == "hann":
        win = np.hanning(len(x)).astype(np.float32)
        xFft = x * win
    else:
        xFft = x

    spec = np.fft.fftshift(np.fft.fft(xFft, n=nfft))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(args.sample_rate)))
    mag = np.abs(spec).astype(np.float64)
    magDb = _db20(mag)
    magDbRel = magDb - float(np.median(magDb))

    targetHz = float(args.target_mhz) * 1e6
    widthHz = float(args.target_width_khz) * 1e3
    targetMask = (np.abs(freq - targetHz) <= widthHz / 2.0) | (np.abs(freq + targetHz) <= widthHz / 2.0)
    ssbMask = np.abs(freq) <= 3.6e6
    restMask = ssbMask & ~targetMask

    posMean = _bandMean(freq, mag, targetHz, widthHz)
    negMean = _bandMean(freq, mag, -targetHz, widthHz)
    targetMean = float(np.mean(mag[targetMask])) if np.any(targetMask) else 0.0
    restMean = float(np.mean(mag[restMask])) if np.any(restMask) else 0.0
    targetVsRestDb20 = float(_db20(targetMean / max(restMean, 1e-30)))
    peakIdx = int(np.argmax(mag))

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}.png"

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    for ax, y, ylabel, title, xlim in [
        (axes[0], mag / max(float(np.max(mag)), 1e-30), "Normalized |FFT|", "Direct FFT magnitude, full band", (-float(args.sample_rate) / 2e6, float(args.sample_rate) / 2e6)),
        (axes[1], magDbRel, "20log10(|FFT|) relative to median (dB)", "Direct FFT magnitude in dB, zoom +/-6 MHz", (-6, 6)),
    ]:
        ax.plot(freq / 1e6, y, linewidth=0.55)
        for center in (-float(args.target_mhz), float(args.target_mhz)):
            ax.axvspan(center - widthHz / 2e6, center + widthHz / 2e6, color="red", alpha=0.16)
            ax.axvline(center, color="red", linestyle="--", linewidth=0.9)
        ax.set_xlim(*xlim)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    labels = ["-1.5 MHz", "+1.5 MHz", "Both target bands", "Rest of +/-3.6 MHz"]
    values = [_db20(negMean), _db20(posMean), _db20(targetMean), _db20(restMean)]
    bars = axes[2].bar(labels, values, color=["#d1495b", "#d1495b", "#edae49", "#00798c"])
    axes[2].set_ylabel("20log10(mean |FFT|)")
    axes[2].set_title(f"Direct FFT band-average magnitude | target vs rest = {targetVsRestDb20:+.2f} dB")
    axes[2].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(f"{path.name} | direct FFT | window={args.window} | nfft={nfft}", fontsize=14)
    fig.savefig(figPath, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result = {
        "inputPath": str(path.resolve()),
        "sampleRate": float(args.sample_rate),
        "samplesUsed": int(len(x)),
        "nfft": int(nfft),
        "window": args.window,
        "peakFreqMHz": float(freq[peakIdx] / 1e6),
        "targetWidthKHz": float(args.target_width_khz),
        "negativeTargetMeanMagDb20": float(_db20(negMean)),
        "positiveTargetMeanMagDb20": float(_db20(posMean)),
        "combinedTargetMeanMagDb20": float(_db20(targetMean)),
        "restOfSsbBandMeanMagDb20": float(_db20(restMean)),
        "targetVsRestMagDb20": targetVsRestDb20,
        "figure": str(figPath.resolve()),
    }
    jsonPath = outDir / f"{args.output_prefix}.json"
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
