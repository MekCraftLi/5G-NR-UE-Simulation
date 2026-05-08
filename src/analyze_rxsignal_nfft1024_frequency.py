"""按 Fs=30.72 MHz、NFFT=1024 对 rxSignal.npy 做频域分析。

本脚本不改变原始 IQ 数据，只按命令行给定的采样率解释频率轴。数据流：
`data/rxSignal.npy` -> 1024 点分块 FFT -> 平均 PSD / FFT bin 功率 /
时频热图 -> PNG、JSON、NPZ 工件。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def _configure_matplotlib_font() -> None:
    """配置中文字体，保证图中的中文标题和坐标轴能正常显示。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="rxSignal.npy 的 NFFT=1024 频域分析")
    parser.add_argument("--input-path", default="data/rxSignal.npy", help="输入 IQ .npy 文件")
    parser.add_argument("--sample-rate", type=float, default=30.72e6, help="采样率，单位 Hz")
    parser.add_argument("--nfft", type=int, default=1024, help="FFT 点数")
    parser.add_argument("--output-prefix", default="rxSignal_fs30p72_nfft1024_frequency_analysis", help="输出前缀")
    parser.add_argument("--window", choices=["hann", "rect"], default="hann", help="分块 FFT 窗函数")
    parser.add_argument("--zoom-mhz", type=float, default=6.0, help="局部频谱显示范围，单位 MHz")
    parser.add_argument("--ssb-half-band-mhz", type=float, default=3.6, help="SSB 240 子载波半带宽，单位 MHz")
    parser.add_argument("--top-count", type=int, default=16, help="JSON 中保留的最强 FFT bin 数量")
    return parser.parse_args()


def _db10(value: np.ndarray | float) -> np.ndarray | float:
    """功率量转 dB。"""
    return 10.0 * np.log10(np.maximum(value, 1e-30))


def _load_signal(path_text: str) -> tuple[Path, np.ndarray]:
    """读取复基带 IQ，并压成一维 complex64。"""
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT_DIR / path
    signal = np.asarray(np.load(path), dtype=np.complex64).reshape(-1)
    return path, signal


def _block_fft(signal: np.ndarray, nfft: int, window_name: str) -> tuple[np.ndarray, np.ndarray]:
    """把输入按 NFFT 非重叠分块并计算 fftshift 后的功率谱。"""
    block_count = len(signal) // int(nfft)
    if block_count <= 0:
        raise ValueError(f"输入长度 {len(signal)} 小于 NFFT={nfft}")
    trimmed = signal[: block_count * int(nfft)]
    blocks = trimmed.reshape(block_count, int(nfft))
    if window_name == "hann":
        window = np.hanning(int(nfft)).astype(np.float32)
    else:
        window = np.ones(int(nfft), dtype=np.float32)

    window_power = max(float(np.sum(window**2)), 1e-12)
    spectrum = np.fft.fftshift(np.fft.fft(blocks * window[None, :], n=int(nfft), axis=1), axes=1)
    power = (np.abs(spectrum) ** 2 / window_power).astype(np.float64)
    return power, trimmed


def _band_summary(freq_hz: np.ndarray, psd: np.ndarray, center_hz: float, width_hz: float) -> dict:
    """统计给定频段内的平均、最大和积分功率。"""
    mask = np.abs(freq_hz - float(center_hz)) <= float(width_hz) / 2.0
    if not np.any(mask):
        return {"binCount": 0, "meanDbRel": None, "maxDbRel": None, "powerSum": 0.0}
    floor = float(np.median(psd))
    return {
        "binCount": int(np.sum(mask)),
        "meanDbRel": float(_db10(np.mean(psd[mask]) / max(floor, 1e-30))),
        "maxDbRel": float(_db10(np.max(psd[mask]) / max(floor, 1e-30))),
        "powerSum": float(np.sum(psd[mask])),
    }


def _make_figure(
    freq_hz: np.ndarray,
    mean_psd: np.ndarray,
    spectrogram_power: np.ndarray,
    result: dict,
    output_png: Path,
) -> None:
    """生成平均频谱、bin 功率和时频热图。"""
    freq_mhz = freq_hz / 1e6
    median_power = max(float(np.median(mean_psd)), 1e-30)
    psd_db_rel = np.asarray(_db10(mean_psd / median_power), dtype=np.float64)
    spec_db_rel = np.asarray(_db10(spectrogram_power / median_power), dtype=np.float64)

    fig, axes = plt.subplots(4, 1, figsize=(16, 18), constrained_layout=True)

    ax = axes[0]
    ax.plot(freq_mhz, psd_db_rel, linewidth=0.9, color="#1f77b4")
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.axvspan(-float(result["ssbHalfBandMHz"]), float(result["ssbHalfBandMHz"]), color="#d9ead3", alpha=0.22, label="±3.6 MHz / 240 子载波区域")
    ax.scatter([float(result["peakFreqHz"]) / 1e6], [float(result["peakDbRel"])], color="#d62728", s=75, zorder=4, label="最强 bin")
    ax.set_title("NFFT=1024 平均功率谱：全频带")
    ax.set_xlabel("频率 / MHz")
    ax.set_ylabel("PSD 相对中位数 / dB")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1]
    zoom = float(result["zoomMHz"])
    mask = np.abs(freq_mhz) <= zoom
    ax.plot(freq_mhz[mask], psd_db_rel[mask], linewidth=1.0, color="#1f77b4")
    ax.axvline(0.0, color="#444444", linestyle="--", linewidth=0.9)
    for edge in (-float(result["ssbHalfBandMHz"]), float(result["ssbHalfBandMHz"])):
        ax.axvline(edge, color="#2ca02c", linestyle=":", linewidth=1.2)
    ax.scatter([float(result["peakFreqHz"]) / 1e6], [float(result["peakDbRel"])], color="#d62728", s=70, zorder=4)
    ax.annotate(
        f"peak={float(result['peakFreqHz'])/1e6:.3f} MHz\n{float(result['peakDbRel']):.2f} dB",
        xy=(float(result["peakFreqHz"]) / 1e6, float(result["peakDbRel"])),
        xytext=(-zoom * 0.92, float(np.max(psd_db_rel[mask])) - 1.0),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.1},
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff5f5", "edgecolor": "#d62728", "alpha": 0.9},
    )
    ax.set_title(f"NFFT=1024 平均功率谱：±{zoom:.1f} MHz 局部")
    ax.set_xlabel("频率 / MHz")
    ax.set_ylabel("PSD 相对中位数 / dB")
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    bin_index = np.arange(len(mean_psd), dtype=np.int32) - len(mean_psd) // 2
    ax.bar(bin_index, psd_db_rel, width=0.85, color="#4f81bd")
    ax.axvline(0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlim(-180, 180)
    ax.set_title("1024 个 FFT bin 的平均功率分布（局部显示中心 360 个 bin）")
    ax.set_xlabel("FFT bin 索引，0 表示 DC，1 bin = 30 kHz")
    ax.set_ylabel("PSD 相对中位数 / dB")
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[3]
    block_count = int(result["blockCount"])
    block_time_ms = float(result["nfft"]) / float(result["sampleRateHz"]) * 1e3
    time_ms = np.arange(block_count, dtype=np.float64) * block_time_ms
    spec_plot = np.clip(spec_db_rel, np.percentile(spec_db_rel, 5), np.percentile(spec_db_rel, 99.5))
    im = ax.imshow(
        spec_plot.T,
        origin="lower",
        aspect="auto",
        extent=[time_ms[0], time_ms[-1] + block_time_ms, freq_mhz[0], freq_mhz[-1]],
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="功率相对全局中位数 / dB")
    ax.set_title("NFFT=1024 非重叠分块时频热图")
    ax.set_xlabel("时间 / ms")
    ax.set_ylabel("频率 / MHz")
    ax.set_ylim(-float(result["zoomMHz"]), float(result["zoomMHz"]))
    ax.grid(False)

    fig.suptitle(
        (
            f"{result['inputPath']} | Fs={float(result['sampleRateHz'])/1e6:.2f} MHz | "
            f"NFFT={int(result['nfft'])} | bin 间隔={float(result['binSpacingHz'])/1e3:.1f} kHz | "
            f"分块数={block_count}"
        ),
        fontsize=14,
    )
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _configure_matplotlib_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path, signal = _load_signal(str(args.input_path))
    nfft = int(args.nfft)
    sample_rate = float(args.sample_rate)
    power, trimmed = _block_fft(signal, nfft=nfft, window_name=str(args.window))
    mean_psd = np.mean(power, axis=0)
    freq_hz = np.fft.fftshift(np.fft.fftfreq(nfft, d=1.0 / sample_rate)).astype(np.float64)
    bin_spacing_hz = sample_rate / nfft
    median_power = max(float(np.median(mean_psd)), 1e-30)
    psd_db_rel = np.asarray(_db10(mean_psd / median_power), dtype=np.float64)
    peak_idx = int(np.argmax(mean_psd))

    ssb_half_band_hz = float(args.ssb_half_band_mhz) * 1e6
    inband_mask = np.abs(freq_hz) <= ssb_half_band_hz
    outband_mask = ~inband_mask
    top_indices = np.argsort(mean_psd)[::-1][: max(1, int(args.top_count))]
    top_bins = [
        {
            "fftShiftedIndex": int(idx),
            "centeredBinIndex": int(idx - nfft // 2),
            "freqHz": float(freq_hz[idx]),
            "freqMHz": float(freq_hz[idx] / 1e6),
            "psdDbRel": float(psd_db_rel[idx]),
        }
        for idx in top_indices
    ]

    result = {
        "method": "rxsignal_frequency_analysis_nfft1024",
        "inputPath": str(input_path.relative_to(ROOT_DIR) if input_path.is_relative_to(ROOT_DIR) else input_path),
        "sampleRateHz": sample_rate,
        "nfft": nfft,
        "binSpacingHz": float(bin_spacing_hz),
        "window": str(args.window),
        "originalSampleCount": int(len(signal)),
        "usedSampleCount": int(len(trimmed)),
        "droppedTailSamples": int(len(signal) - len(trimmed)),
        "blockCount": int(power.shape[0]),
        "blockDurationUs": float(nfft / sample_rate * 1e6),
        "zoomMHz": float(args.zoom_mhz),
        "ssbHalfBandMHz": float(args.ssb_half_band_mhz),
        "peakFreqHz": float(freq_hz[peak_idx]),
        "peakFreqMHz": float(freq_hz[peak_idx] / 1e6),
        "peakCenteredBinIndex": int(peak_idx - nfft // 2),
        "peakDbRel": float(psd_db_rel[peak_idx]),
        "medianPowerDb": float(_db10(median_power)),
        "inbandMeanDbRel": float(_db10(np.mean(mean_psd[inband_mask]) / median_power)) if np.any(inband_mask) else None,
        "outbandMeanDbRel": float(_db10(np.mean(mean_psd[outband_mask]) / median_power)) if np.any(outband_mask) else None,
        "inbandVsOutbandDb": (
            float(_db10(np.mean(mean_psd[inband_mask]) / max(float(np.mean(mean_psd[outband_mask])), 1e-30)))
            if np.any(inband_mask) and np.any(outband_mask)
            else None
        ),
        "bandSummary": {
            "dc_300kHz": _band_summary(freq_hz, mean_psd, 0.0, 300e3),
            "positive_1p5MHz_300kHz": _band_summary(freq_hz, mean_psd, 1.5e6, 300e3),
            "negative_1p5MHz_300kHz": _band_summary(freq_hz, mean_psd, -1.5e6, 300e3),
            "positive_3p6MHz_300kHz": _band_summary(freq_hz, mean_psd, 3.6e6, 300e3),
            "negative_3p6MHz_300kHz": _band_summary(freq_hz, mean_psd, -3.6e6, 300e3),
        },
        "topBins": top_bins,
    }

    output_png = OUTPUT_DIR / f"{args.output_prefix}.png"
    output_json = OUTPUT_DIR / f"{args.output_prefix}.json"
    output_npz = OUTPUT_DIR / f"{args.output_prefix}.npz"
    _make_figure(
        freq_hz=freq_hz,
        mean_psd=mean_psd,
        spectrogram_power=power,
        result=result,
        output_png=output_png,
    )
    result["figure"] = str(output_png.relative_to(ROOT_DIR))
    result["npz"] = str(output_npz.relative_to(ROOT_DIR))
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        freqHz=freq_hz,
        meanPsd=mean_psd,
        meanPsdDbRel=psd_db_rel,
        spectrogramPower=power.astype(np.float32),
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
