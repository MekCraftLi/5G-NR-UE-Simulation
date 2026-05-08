"""对 TX 波形执行 PSS 时域滑动互相关并绘图。

数据流：
1. 读取 `data/txs.npy`，如果该文件不存在且 `data/txs0.npy` 存在，则使用
   `data/txs0.npy` 作为后备输入。
2. 根据采样率、SCS、FFT 和 CP 构造本地 PSS 时域模板。
3. 对 `N_ID_2=0/1/2` 分别计算滑动互相关幅度：
   横轴为候选 PSS 起点采样，纵轴为 `|rx 与 PSS 时域模板互相关|`。
4. 保存图像、JSON 摘要和 NPZ 曲线，供报告或后续交接复用。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import createSsbConfigWithOverrides
from pss.pssBasebandSearcher import PssBasebandSearcher


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "output"


def _configure_matplotlib_font() -> None:
    """配置中文字体，避免图像标题和坐标轴中文变成方框。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 TX 波形执行 PSS 时域互相关并绘图")
    parser.add_argument("--input-path", default="data/txs.npy", help="输入 IQ 的 .npy 文件")
    parser.add_argument("--sample-rate", type=float, default=30.72e6, help="采样率，单位 Hz")
    parser.add_argument("--scs", type=int, default=30000, help="SSB 子载波间隔，单位 Hz")
    parser.add_argument("--fft-size", type=int, default=None, help="FFT 点数，默认由 sample-rate/scs 推导")
    parser.add_argument("--normal-cp", type=int, default=None, help="常规 CP 长度，默认按 NR 规则缩放")
    parser.add_argument("--ssb-subcarrier-offset", type=int, default=0, help="PSS 映射相对中心 240 子载波窗口的偏移")
    parser.add_argument("--freq-hz", type=float, default=0.0, help="互相关前使用的频偏补偿假设，单位 Hz")
    parser.add_argument("--output-prefix", default="", help="输出文件前缀，默认使用实际输入文件名")
    parser.add_argument("--top-count", type=int, default=12, help="JSON 和图中保留的强峰数量")
    parser.add_argument("--zoom-radius", type=int, default=5000, help="全局峰附近局部放大的左右采样半径")
    parser.add_argument("--max-plot-points", type=int, default=220000, help="全局曲线最多绘制的点数，用于避免图像过重")
    return parser.parse_args()


def _resolve_input_path(requested_path: str) -> tuple[Path, bool, str]:
    """解析输入路径；`data/txs.npy` 不存在时允许回退到 `data/txs0.npy`。"""
    requested = Path(requested_path)
    if not requested.is_absolute():
        requested = ROOT_DIR / requested
    if requested.exists():
        return requested, False, ""

    fallback = ROOT_DIR / "data" / "txs0.npy"
    if requested.name == "txs.npy" and fallback.exists():
        reason = f"请求的 {requested} 不存在，已回退到 {fallback}"
        return fallback, True, reason

    raise FileNotFoundError(f"输入文件不存在：{requested}")


def _top_separated_peaks(corr: np.ndarray, top_count: int, exclusion_radius: int) -> list[dict]:
    """从一条互相关曲线中提取相互间隔足够远的强峰。"""
    work = np.asarray(corr, dtype=np.float32).copy()
    peaks: list[dict] = []
    exclusion = max(1, int(exclusion_radius))
    for _ in range(max(1, int(top_count))):
        idx = int(np.argmax(work))
        value = float(work[idx])
        if not np.isfinite(value) or value <= 0.0:
            break
        peaks.append({"timingOffset": idx, "peakValue": value})
        lo = max(0, idx - exclusion)
        hi = min(len(work), idx + exclusion + 1)
        work[lo:hi] = -np.inf
    return peaks


def _make_plot(
    corr_by_nid2: dict[int, np.ndarray],
    result: dict,
    output_png: Path,
    zoom_radius: int,
    max_plot_points: int,
) -> None:
    """绘制 PSS 全局互相关、局部放大和 Top 强峰摘要。"""
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    best = result["best"]
    best_idx = int(best["timingOffset"])
    best_nid2 = int(best["nId2"])

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)

    corr_len = len(next(iter(corr_by_nid2.values())))
    step = max(1, int(np.ceil(corr_len / max(1, int(max_plot_points)))))
    x_full = np.arange(0, corr_len, step, dtype=np.int64)

    ax = axes[0]
    for nid2 in [0, 1, 2]:
        y = corr_by_nid2[nid2][::step]
        ax.plot(x_full, y, linewidth=0.8, color=colors[nid2], label=f"N_ID_2={nid2}")
    ax.axvline(best_idx, color="#d62728", linestyle="--", linewidth=1.4, label=f"全局峰 sample={best_idx}")
    ax.set_title("PSS 时域滑动互相关全局曲线")
    ax.set_xlabel("候选 PSS 起点采样")
    ax.set_ylabel("|rx 与 PSS 模板互相关|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=4)

    ax = axes[1]
    lo = max(0, best_idx - int(zoom_radius))
    hi = min(corr_len, best_idx + int(zoom_radius) + 1)
    x_zoom = np.arange(lo, hi, dtype=np.int64)
    for nid2 in [0, 1, 2]:
        ax.plot(x_zoom, corr_by_nid2[nid2][lo:hi], linewidth=1.0, color=colors[nid2], label=f"N_ID_2={nid2}")
    ax.scatter([best_idx], [float(best["peakValue"])], s=85, marker="^", color="#d62728", zorder=5)
    ax.annotate(
        f"全局峰\nN_ID_2={best_nid2}\nsample={best_idx}\nvalue={float(best['peakValue']):.3f}",
        xy=(best_idx, float(best["peakValue"])),
        xytext=(lo + max(1, (hi - lo) // 20), float(best["peakValue"]) * 0.82),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.2},
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff5f5", "edgecolor": "#d62728", "alpha": 0.92},
    )
    ax.set_title("PSS 时域滑动互相关局部放大")
    ax.set_xlabel("候选 PSS 起点采样")
    ax.set_ylabel("|rx 与 PSS 模板互相关|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[2]
    global_peaks = result["globalTopPeaks"][: int(result["topCount"])]
    labels = [f"N{item['nId2']}@{item['timingOffset']}" for item in global_peaks]
    values = [float(item["peakValue"]) for item in global_peaks]
    bar_colors = [colors[int(item["nId2"])] for item in global_peaks]
    bars = ax.bar(np.arange(len(values)), values, color=bar_colors)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("PSS 互相关 Top 强峰摘要")
    ax.set_xlabel("候选峰：N_ID_2@采样点")
    ax.set_ylabel("|rx 与 PSS 模板互相关|")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        (
            f"输入：{result['actualInputPath']} | Fs={float(result['sampleRateHz'])/1e6:.2f} MHz | "
            f"SCS={int(result['scsHz'])/1000} kHz | freqComp={float(result['freqCompHz']):.1f} Hz"
        ),
        fontsize=14,
    )
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    _configure_matplotlib_font()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_path, fallback_used, fallback_reason = _resolve_input_path(str(args.input_path))
    signal = np.load(input_path).astype(np.complex64).reshape(-1)

    config = createSsbConfigWithOverrides(
        sampleRate=float(args.sample_rate),
        subcarrierSpacing=int(args.scs),
        fftSize=args.fft_size,
        normalCpLength=args.normal_cp,
        ssbSubcarrierOffset=int(args.ssb_subcarrier_offset),
    )
    searcher = PssBasebandSearcher(
        config,
        freqMinHz=float(args.freq_hz),
        freqMaxHz=float(args.freq_hz),
        useGpu=False,
    )

    corr_by_nid2: dict[int, np.ndarray] = {}
    per_nid2: list[dict] = []
    all_top_peaks: list[dict] = []
    exclusion_radius = int(searcher.templateLength)
    for nid2 in [0, 1, 2]:
        corr = searcher.getCorrelationAtFreq(signal, float(args.freq_hz), nid2)
        corr_by_nid2[nid2] = corr
        peak_idx = int(np.argmax(corr))
        peak_value = float(corr[peak_idx])
        peaks = _top_separated_peaks(corr, int(args.top_count), exclusion_radius)
        for item in peaks:
            all_top_peaks.append(
                {
                    "nId2": nid2,
                    "timingOffset": int(item["timingOffset"]),
                    "peakValue": float(item["peakValue"]),
                }
            )
        per_nid2.append(
            {
                "nId2": nid2,
                "bestTimingOffset": peak_idx,
                "bestPeakValue": peak_value,
                "topPeaks": peaks,
            }
        )

    all_top_peaks = sorted(all_top_peaks, key=lambda item: float(item["peakValue"]), reverse=True)
    best = all_top_peaks[0]
    prefix = str(args.output_prefix).strip() or input_path.stem

    result = {
        "method": "pss_time_domain_sliding_cross_correlation",
        "requestedInputPath": str(args.input_path),
        "actualInputPath": str(input_path.relative_to(ROOT_DIR) if input_path.is_relative_to(ROOT_DIR) else input_path),
        "fallbackUsed": bool(fallback_used),
        "fallbackReason": fallback_reason,
        "sampleRateHz": float(args.sample_rate),
        "scsHz": int(args.scs),
        "fftSize": int(config.FftSize),
        "normalCpLength": int(config.NormalCpLength),
        "ssbSubcarrierOffset": int(config.SsbSubcarrierOffset),
        "freqCompHz": float(args.freq_hz),
        "templateLength": int(searcher.templateLength),
        "correlationLength": int(len(next(iter(corr_by_nid2.values())))),
        "correlationDefinition": "|sum(rx[i:i+templateLength] * conj(pssTemplate))|，模板已单位范数，接收窗口未做能量归一化",
        "topCount": int(args.top_count),
        "best": {
            "nId2": int(best["nId2"]),
            "timingOffset": int(best["timingOffset"]),
            "peakValue": float(best["peakValue"]),
        },
        "perNId2": per_nid2,
        "globalTopPeaks": all_top_peaks[: max(1, int(args.top_count))],
    }

    output_png = OUTPUT_DIR / f"{prefix}_pss_time_correlation.png"
    output_json = OUTPUT_DIR / f"{prefix}_pss_time_correlation.json"
    output_npz = OUTPUT_DIR / f"{prefix}_pss_time_correlation_curves.npz"

    _make_plot(
        corr_by_nid2=corr_by_nid2,
        result=result,
        output_png=output_png,
        zoom_radius=int(args.zoom_radius),
        max_plot_points=int(args.max_plot_points),
    )
    output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez_compressed(
        output_npz,
        corrNId2_0=corr_by_nid2[0],
        corrNId2_1=corr_by_nid2[1],
        corrNId2_2=corr_by_nid2[2],
    )

    print(f"PSS 互相关图已保存：{output_png}")
    print(f"PSS 互相关结果已保存：{output_json}")
    print(f"PSS 互相关曲线已保存：{output_npz}")
    print(f"最佳结果：N_ID_2={result['best']['nId2']}, timingOffset={result['best']['timingOffset']}, peak={result['best']['peakValue']:.6f}")
    if fallback_used:
        print(fallback_reason)


if __name__ == "__main__":
    main()
