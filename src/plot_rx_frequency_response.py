import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RX frequency response with +/-1.5 MHz annotations")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--nfft", type=int, default=262144)
    parser.add_argument("--target-mhz", type=float, default=1.5)
    parser.add_argument("--target-width-khz", type=float, default=200.0)
    parser.add_argument("--smooth-bins", type=int, default=101)
    parser.add_argument("--output-prefix", type=str, default="rxsignal_frequency_response")
    return parser.parse_args()


def _welch(x: np.ndarray, fs: float, nfft: int) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(int(nfft), len(x))
    hop = nfft // 2
    starts = list(range(0, len(x) - nfft + 1, hop)) or [0]
    win = np.hanning(nfft).astype(np.float32)
    acc = np.zeros(nfft, dtype=np.float64)
    for start in starts:
        seg = x[start:start + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        spec = np.fft.fftshift(np.fft.fft(seg * win, nfft))
        acc += np.abs(spec) ** 2 / max(float(np.sum(win ** 2)), 1e-12)
    psd = acc / max(len(starts), 1)
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / fs))
    return freq, psd


def _db(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def _smooth(y: np.ndarray, bins: int) -> np.ndarray:
    bins = max(1, int(bins))
    if bins <= 1:
        return y
    kernel = np.ones(bins, dtype=np.float64) / bins
    return np.convolve(y, kernel, mode="same")


def _bandMean(freq: np.ndarray, psd: np.ndarray, center: float, width: float) -> float:
    mask = np.abs(freq - center) <= width / 2.0
    return float(np.mean(psd[mask]))


def main() -> None:
    args = _parseArgs()
    path = Path(args.input_path)
    x = np.asarray(np.load(path), dtype=np.complex64).reshape(-1)
    if args.max_samples and args.max_samples > 0:
        x = x[:args.max_samples]

    fs = float(args.sample_rate)
    target = float(args.target_mhz) * 1e6
    width = float(args.target_width_khz) * 1e3
    freq, psd = _welch(x, fs, int(args.nfft))
    psdDb = _db(psd)
    floorMask = np.abs(freq) > 3.6e6
    floorDb = float(np.median(psdDb[floorMask]))
    psdRelDb = psdDb - floorDb
    psdRelSmooth = _smooth(psdRelDb, int(args.smooth_bins))

    targetMask = (np.abs(freq - target) <= width / 2.0) | (np.abs(freq + target) <= width / 2.0)
    ssbMask = np.abs(freq) <= 3.6e6
    restMask = ssbMask & ~targetMask
    posMeanDb = float(_db(_bandMean(freq, psd, target, width)) - floorDb)
    negMeanDb = float(_db(_bandMean(freq, psd, -target, width)) - floorDb)
    targetMeanDb = float(_db(np.mean(psd[targetMask])) - floorDb)
    restMeanDb = float(_db(np.mean(psd[restMask])) - floorDb)
    targetVsRestDb = targetMeanDb - restMeanDb

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}.png"

    fig, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)

    for ax, xlim, title in [
        (axes[0], (-fs / 2 / 1e6, fs / 2 / 1e6), "Full RX frequency response"),
        (axes[1], (-6, 6), "Zoom: +/-6 MHz"),
    ]:
        ax.plot(freq / 1e6, psdRelDb, color="#8aa1c1", linewidth=0.35, alpha=0.5, label="Raw Welch PSD")
        ax.plot(freq / 1e6, psdRelSmooth, color="#183a59", linewidth=1.2, label=f"Smoothed ({args.smooth_bins} bins)")
        for center in (-target / 1e6, target / 1e6):
            ax.axvspan(center - width / 2 / 1e6, center + width / 2 / 1e6, color="#d1495b", alpha=0.18)
            ax.axvline(center, color="#d1495b", linestyle="--", linewidth=1.0)
        ax.axhline(0, color="black", linestyle=":", linewidth=0.8, label="Wideband median floor" if ax is axes[0] else None)
        ax.set_xlim(*xlim)
        ax.set_ylabel("PSD above floor (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    labels = ["-1.5MHz band", "+1.5MHz band", "Both target bands", "Rest of +/-3.6MHz"]
    vals = [negMeanDb, posMeanDb, targetMeanDb, restMeanDb]
    bars = axes[2].bar(labels, vals, color=["#d1495b", "#d1495b", "#edae49", "#00798c"])
    axes[2].axhline(0, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("Mean PSD above floor (dB)")
    axes[2].set_title(f"Band average comparison | target vs rest = {targetVsRestDb:+.2f} dB")
    axes[2].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, vals):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            val + (0.05 if val >= 0 else -0.12),
            f"{val:+.2f} dB",
            ha="center",
            va="bottom" if val >= 0 else "top",
            fontsize=10,
        )

    fig.suptitle(f"{path.name} | fs={fs/1e6:.2f} MHz | target width={width/1e3:.0f} kHz", fontsize=14)
    fig.savefig(figPath, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result = {
        "inputPath": str(path.resolve()),
        "sampleRate": fs,
        "samplesUsed": int(len(x)),
        "noiseFloorDb": floorDb,
        "targetWidthKHz": float(args.target_width_khz),
        "negativeTargetMeanAboveFloorDb": negMeanDb,
        "positiveTargetMeanAboveFloorDb": posMeanDb,
        "combinedTargetMeanAboveFloorDb": targetMeanDb,
        "restOfSsbBandMeanAboveFloorDb": restMeanDb,
        "targetVsRestDb": targetVsRestDb,
        "figure": str(figPath.resolve()),
    }
    jsonPath = outDir / f"{args.output_prefix}.json"
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
