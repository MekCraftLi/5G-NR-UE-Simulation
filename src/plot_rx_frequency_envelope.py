import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot RX direct-FFT frequency envelope")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--max-samples", type=int, default=65536)
    parser.add_argument("--nfft", type=int, default=262144)
    parser.add_argument("--window", choices=["none", "hann"], default="hann")
    parser.add_argument("--envelope-bins", type=int, default=801)
    parser.add_argument("--target-mhz", type=float, default=1.5)
    parser.add_argument("--target-width-khz", type=float, default=200.0)
    parser.add_argument("--signal-band-mhz", type=float, default=7.2)
    parser.add_argument("--output-prefix", type=str, default="rxSignal_frequency_envelope")
    return parser.parse_args()


def _db20(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-30))


def _rolling_percentile(x: np.ndarray, window: int, percentile: float) -> np.ndarray:
    window = max(3, int(window) | 1)
    half = window // 2
    padded = np.pad(x, (half, half), mode="edge")
    out = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        out[i] = np.percentile(padded[i:i + window], percentile)
    return out


def _band_mean(freq: np.ndarray, power: np.ndarray, center_hz: float, width_hz: float) -> float:
    mask = np.abs(freq - center_hz) <= width_hz / 2.0
    return float(np.mean(power[mask])) if np.any(mask) else 0.0


def main() -> None:
    args = _parseArgs()
    path = Path(args.input_path)
    x = np.asarray(np.load(path), dtype=np.complex64).reshape(-1)
    if int(args.max_samples) > 0:
        x = x[: int(args.max_samples)]

    nfft = int(args.nfft)
    if nfft < len(x):
        raise ValueError(f"nfft={nfft} must be >= samples used={len(x)} for comparable zero-padded envelope")

    if args.window == "hann":
        win = np.hanning(len(x)).astype(np.float32)
        xw = x * win
    else:
        xw = x

    spec = np.fft.fftshift(np.fft.fft(xw, n=nfft))
    freq = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / float(args.sample_rate)))
    mag = np.abs(spec).astype(np.float64)
    power = mag ** 2
    mag_db = _db20(mag)
    floor_db = float(np.median(mag_db[np.abs(freq) > 10e6]))
    mag_rel_db = mag_db - floor_db

    median_env = _rolling_percentile(mag_rel_db, int(args.envelope_bins), 50)
    upper_env = _rolling_percentile(mag_rel_db, int(args.envelope_bins), 95)

    target_hz = float(args.target_mhz) * 1e6
    target_width_hz = float(args.target_width_khz) * 1e3
    signal_band_hz = float(args.signal_band_mhz) * 1e6
    signal_mask = np.abs(freq) <= signal_band_hz / 2.0
    noise_mask = (np.abs(freq) >= 10e6) & (np.abs(freq) <= float(args.sample_rate) * 0.45)
    noise_power = float(np.median(power[noise_mask]))
    inband_power = float(np.mean(power[signal_mask]))
    snr_linear = max(inband_power - noise_power, 0.0) / max(noise_power, 1e-30)
    snr_db = float(10.0 * np.log10(max(snr_linear, 1e-30)))

    target_mask = (
        (np.abs(freq - target_hz) <= target_width_hz / 2.0)
        | (np.abs(freq + target_hz) <= target_width_hz / 2.0)
    )
    rest_mask = signal_mask & ~target_mask
    target_mean_power = float(np.mean(power[target_mask]))
    rest_mean_power = float(np.mean(power[rest_mask]))
    target_vs_rest_db = float(10.0 * np.log10(max(target_mean_power, 1e-30) / max(rest_mean_power, 1e-30)))

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / f"{args.output_prefix}.png"

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    axes[0].plot(freq / 1e6, mag_rel_db, color="#8aa1c1", linewidth=0.35, alpha=0.45, label="Direct FFT magnitude")
    axes[0].plot(freq / 1e6, median_env, color="#183a59", linewidth=1.3, label="Rolling median envelope")
    axes[0].plot(freq / 1e6, upper_env, color="#d1495b", linewidth=1.1, label="Rolling 95% envelope")
    axes[0].axhline(0, color="black", linestyle=":", linewidth=0.8, label="Far-out median floor")
    axes[0].axvspan(-signal_band_hz / 2e6, signal_band_hz / 2e6, color="#00798c", alpha=0.08, label=f"+/-{signal_band_hz/2e6:.1f} MHz band")
    for center in (-float(args.target_mhz), float(args.target_mhz)):
        axes[0].axvspan(center - target_width_hz / 2e6, center + target_width_hz / 2e6, color="#edae49", alpha=0.20)
        axes[0].axvline(center, color="#edae49", linestyle="--", linewidth=1.0)
    axes[0].set_xlim(-12, 12)
    axes[0].set_ylabel("Magnitude above floor (dB)")
    axes[0].set_title("Frequency-domain envelope, zoom +/-12 MHz")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].plot(freq / 1e6, median_env, color="#183a59", linewidth=1.4, label="Median envelope")
    axes[1].plot(freq / 1e6, upper_env, color="#d1495b", linewidth=1.2, label="95% envelope")
    axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
    for center in (-float(args.target_mhz), float(args.target_mhz)):
        axes[1].axvspan(center - target_width_hz / 2e6, center + target_width_hz / 2e6, color="#edae49", alpha=0.20)
        axes[1].axvline(center, color="#edae49", linestyle="--", linewidth=1.0)
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylabel("Magnitude above floor (dB)")
    axes[1].set_title("Envelope detail, zoom +/-4 MHz")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    labels = ["Signal band avg", "Noise floor", "+/-1.5MHz avg", "Rest in band avg"]
    values = [
        10.0 * np.log10(max(inband_power, 1e-30)),
        10.0 * np.log10(max(noise_power, 1e-30)),
        10.0 * np.log10(max(target_mean_power, 1e-30)),
        10.0 * np.log10(max(rest_mean_power, 1e-30)),
    ]
    bars = axes[2].bar(labels, values, color=["#00798c", "#555555", "#edae49", "#66a182"])
    axes[2].set_ylabel("Power estimate (dB, arbitrary)")
    axes[2].set_title(f"Band-power estimates | SNR ~= {snr_db:.2f} dB | +/-1.5MHz vs rest = {target_vs_rest_db:+.2f} dB")
    axes[2].grid(True, axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"{path.name} | samples={len(x)} | NFFT={nfft} | window={args.window} | envelope={args.envelope_bins} bins",
        fontsize=14,
    )
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result = {
        "inputPath": str(path.resolve()),
        "sampleRate": float(args.sample_rate),
        "samplesUsed": int(len(x)),
        "nfft": int(nfft),
        "window": args.window,
        "envelopeBins": int(args.envelope_bins),
        "farOutMedianFloorMagDb": floor_db,
        "signalBandMHz": float(args.signal_band_mhz),
        "targetMHz": float(args.target_mhz),
        "targetWidthKHz": float(args.target_width_khz),
        "snrDbByBandPower": snr_db,
        "signalBandMeanPowerDb": float(values[0]),
        "noiseFloorPowerDb": float(values[1]),
        "targetVsRestDb": target_vs_rest_db,
        "figure": str(fig_path.resolve()),
    }
    json_path = out_dir / f"{args.output_prefix}.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
