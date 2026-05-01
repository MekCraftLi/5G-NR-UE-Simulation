"""
SCS 检测可视化

绘制 CP 自相关结果，直观证明检测到的 SCS
"""
import os

import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "output")


def plotScsDetection(
    signal: np.ndarray,
    sampleRate: float,
    outputPrefix: str = "",
):
    """
    CP 自相关 SCS 检测可视化

    对每种 SCS 假设绘制:
      - 折叠后的 CP 自相关（按符号长度）
      - 峰均比对比柱状图
    """
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    signalC64 = signal.astype(np.complex64)
    corrLen = min(50000, len(signalC64) - 3000)

    hypotheses = [
        (15000, 2048, 144, 2192, "15 kHz (FFT=2048, CP=144)"),
        (30000, 1024, 72, 1096, "30 kHz (FFT=1024, CP=72)"),
        (60000, 512, 36, 548, "60 kHz (FFT=512, CP=36)"),
    ]

    results = []
    for scs, fftN, cpLen, symLen, label in hypotheses:
        scale = sampleRate / 30.72e6
        fftN_scaled = int(fftN * scale)
        cpLen_scaled = int(cpLen * scale)
        symLen_scaled = int(symLen * scale)

        corr = np.zeros(corrLen, dtype=np.float64)
        for i in range(corrLen):
            a = signalC64[i:i + cpLen_scaled]
            b = signalC64[i + fftN_scaled:i + fftN_scaled + cpLen_scaled]
            corr[i] = np.abs(np.vdot(a, b))

        folded = np.zeros(symLen_scaled, dtype=np.float64)
        for i in range(0, corrLen - symLen_scaled, symLen_scaled):
            folded += corr[i:i + symLen_scaled]

        peak = float(np.max(folded))
        mean = float(np.mean(folded))
        score = peak / mean if mean > 0 else 1.0
        peakPos = int(np.argmax(folded))

        results.append({
            "scs": scs, "label": label,
            "folded": folded, "symLen": symLen_scaled,
            "peak": peak, "mean": mean, "score": score,
            "peakPos": peakPos,
        })

    best = max(results, key=lambda r: r["score"])

    # ── 图: 3×2 布局 ──
    fig = plt.figure(figsize=(18, 12))

    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    barColors = [
        "#2ca02c" if r["scs"] == best["scs"] else "#cccccc"
        for r in results
    ]

    # 左侧: 每个 SCS 假设的折叠 CP 自相关
    for i, r in enumerate(results):
        ax = fig.add_subplot(3, 2, i * 2 + 1)
        x = np.arange(r["symLen"])
        ax.plot(x, r["folded"], color=colors[i], linewidth=0.8)
        ax.axvline(r["peakPos"], color="red", linestyle="--", linewidth=1.2,
                   label=f"peak @ {r['peakPos']}")
        ax.axhline(r["mean"], color="gray", linestyle=":", linewidth=1.0,
                   label=f"mean={r['mean']:.1f}")
        ax.set_title(f"SCS={r['label']}\npeak/mean={r['score']:.2f}", fontsize=11)
        ax.set_xlabel("Sample offset within symbol")
        ax.set_ylabel("CP correlation")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 右侧: 峰均比对比
    axBar = fig.add_subplot(3, 2, 2)
    labels = [f"{r['scs']/1000:.0f} kHz" for r in results]
    scores = [r["score"] for r in results]
    bars = axBar.bar(labels, scores, color=barColors, edgecolor="black", linewidth=1.2)
    axBar.axhline(1.0, color="gray", linestyle="--", linewidth=1.0, label="noise level")
    axBar.set_ylabel("CP correlation peak / mean")
    axBar.set_title(f"SCS Detection Confidence\nBest: {best['scs']/1000:.0f} kHz "
                    f"(score={best['score']:.2f})", fontsize=12)
    axBar.set_ylim(0.9, max(scores) * 1.15)
    for bar, score in zip(bars, scores):
        axBar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{score:.2f}", ha="center", va="bottom", fontweight="bold", fontsize=12)
    axBar.legend(fontsize=8)
    axBar.grid(True, alpha=0.3, axis="y")

    # 右侧下方: 结论文本
    axInfo = fig.add_subplot(3, 2, 4)
    axInfo.axis("off")
    infoLines = [
        "SCS Detection Summary",
        "=" * 40,
        "",
        f"  Detected SCS:    {best['scs']/1000:.0f} kHz",
        f"  FFT size:        {best['symLen'] - int(best['scs']/1000*144/15)}",
        f"  CP length:       {int(best['scs']/1000*144/15)} samples",
        f"  Confidence:      {best['score']:.2f} (peak/mean ratio)",
        "",
        "  Evidence:",
    ]
    for r in sorted(results, key=lambda x: x["score"], reverse=True):
        marker = " <-- BEST" if r["scs"] == best["scs"] else ""
        infoLines.append(
            f"    SCS={r['scs']/1000:.0f}kHz:  score={r['score']:.2f}"
            f"{marker}"
        )
    infoLines += [
        "",
        "  Method: CP autocorrelation",
        "  corr[i] = |sum(x[i:i+CP] * conj(x[i+FFT:i+FFT+CP]))|",
        "  Folded by symbol length, peak/mean compared.",
    ]
    axInfo.text(0.05, 0.95, "\n".join(infoLines), transform=axInfo.transAxes,
                fontsize=10, fontfamily="monospace", verticalalignment="top")

    # 右下: CP 自相关原理示意
    axPrinciple = fig.add_subplot(3, 2, 6)
    axPrinciple.axis("off")
    principle = [
        "Principle: CP Autocorrelation",
        "=" * 40,
        "",
        "  OFDM Symbol Structure:",
        "  |<---- CP ---->|<-------- FFT / data -------->|",
        "  |  x[0]..x[CP] |  x[CP]..x[CP+FFT]            |",
        "  |      ^        |        ^                      |",
        "  |      |______ dot product _____|                |",
        "",
        "  CP is a copy of the END of the symbol.",
        f"  For SCS={best['scs']/1000:.0f}kHz, this repetition",
        f"  occurs every {best['symLen']} samples ({best['symLen'] - int(best['scs']/1000*144/15)} + {int(best['scs']/1000*144/15)}).",
        "",
        "  The correct SCS hypothesis produces the",
        "  strongest autocorrelation because the CP",
        "  length and symbol period align.",
    ]
    axPrinciple.text(0.05, 0.95, "\n".join(principle), transform=axPrinciple.transAxes,
                     fontsize=10, fontfamily="monospace", verticalalignment="top")

    plt.tight_layout()
    prefix = f"{outputPrefix}_" if outputPrefix else ""
    path = os.path.join(_OUTPUT_DIR, f"{prefix}scs_detection.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path
