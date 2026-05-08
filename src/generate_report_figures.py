import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import createSsbConfigWithOverrides
from common.gscn import GscnRaster
from pbch.pbchDecoder import PbchDecoder
from pss.pssBasebandSearcher import PssBasebandSearcher


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
ROOT_DIR = Path(__file__).resolve().parents[1]


def configure_matplotlib_font() -> None:
    """配置 Matplotlib 中文字体，避免报告图片标题出现方框。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(name: str) -> dict:
    path = OUTPUT_DIR / name
    return json.loads(path.read_text(encoding="utf-8"))


def pbch_dmrs_re_list(n_id_cell: int) -> list[tuple[int, int]]:
    v = int(n_id_cell) % 4
    items: list[tuple[int, int]] = []
    for l in (1, 3):
        for k in range(v, 240, 4):
            items.append((k, l))
    for k in list(range(v, 48, 4)) + list(range(192 + v, 240, 4)):
        items.append((k, 2))
    return sorted(items, key=lambda x: (x[1], x[0]))


def plot_ssb_resource_map() -> None:
    grid = np.zeros((4, 240), dtype=np.int8)

    grid[0, 56:183] = 1  # PSS
    grid[2, 56:183] = 2  # SSS

    for k in range(240):
        grid[1, k] = 3
        grid[3, k] = 3
    for k in list(range(0, 48)) + list(range(192, 240)):
        grid[2, k] = 3

    for k, l in pbch_dmrs_re_list(0):
        grid[l, k] = 4

    cmap = plt.matplotlib.colors.ListedColormap(
        ["#f5f5f5", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    )
    bounds = np.arange(-0.5, 5.5, 1.0)
    norm = plt.matplotlib.colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm)
    ax.set_title("SS/PBCH Block Resource Map (Example: N_ID_cell = 0)")
    ax.set_xlabel("Subcarrier index k within SS/PBCH block")
    ax.set_ylabel("OFDM symbol index l")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticks([0, 56, 120, 182, 239])
    ax.set_xticklabels(["0", "56", "120", "182", "239"])
    legend_labels = [
        ("Unused / Set to 0", "#f5f5f5"),
        ("PSS", "#1f77b4"),
        ("SSS", "#2ca02c"),
        ("PBCH", "#ff7f0e"),
        ("PBCH DM-RS", "#d62728"),
    ]
    handles = [
        plt.matplotlib.patches.Patch(color=color, label=label)
        for label, color in legend_labels
    ]
    ax.legend(handles=handles, loc="upper right", ncol=5, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_ssb_resource_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def pss_sequence(n_id_2: int) -> np.ndarray:
    x = np.zeros(127 + 7, dtype=np.int8)
    x[:7] = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.int8)
    for n in range(127):
        x[n + 7] = (x[n + 4] + x[n]) & 1
    n = np.arange(127, dtype=np.int32)
    return (1.0 - 2.0 * x[(n + 43 * int(n_id_2)) % 127]).astype(np.float32)


def plot_pss_sequences_and_mapping() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    n = np.arange(127)
    for n_id_2, color in zip([0, 1, 2], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        axes[0].step(n, pss_sequence(n_id_2), where="mid", label=f"N_ID2={n_id_2}", color=color)
    axes[0].set_title("PSS Sequences in BPSK Form")
    axes[0].set_xlabel("n")
    axes[0].set_ylabel("d_PSS(n)")
    axes[0].set_ylim(-1.4, 1.4)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    fft_size = 4096
    center = fft_size // 2
    freq_grid = np.zeros(fft_size, dtype=np.float32)
    freq_grid[center - 63:center + 64] = 1.0
    axes[1].plot(np.arange(fft_size), freq_grid, color="#d62728", linewidth=1.0)
    axes[1].axvline(center - 63, color="black", linestyle="--", linewidth=0.8)
    axes[1].axvline(center + 63, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_title("Example PSS Occupancy on 4096-point FFT Grid")
    axes[1].set_xlabel("FFT bin index")
    axes[1].set_ylabel("Occupied")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_pss_sequences_and_mapping.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_dmrs_pattern() -> None:
    dmrs = pbch_dmrs_re_list(0)
    x = np.array([item[0] for item in dmrs], dtype=np.int32)
    y = np.array([item[1] for item in dmrs], dtype=np.int32)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(x, y, s=18, c="#d62728")
    ax.set_title("PBCH DM-RS Positions for N_ID_cell = 0")
    ax.set_xlabel("Subcarrier index k within SS/PBCH block")
    ax.set_ylabel("OFDM symbol index l")
    ax.set_yticks([1, 2, 3])
    ax.set_xlim(-2, 242)
    ax.set_ylim(0.5, 3.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_pbch_dmrs_pattern.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cp_symbol_structure() -> None:
    normal_cp = 288
    long_cp = 352
    fft_size = 4096
    cp_lengths = [long_cp, normal_cp, normal_cp, normal_cp]
    useful = [fft_size] * 4
    x = np.arange(4)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, cp_lengths, color="#8dd3c7", label="CP length")
    ax.bar(x, useful, bottom=cp_lengths, color="#80b1d3", label="Useful FFT part")
    for i, cp in enumerate(cp_lengths):
        ax.text(i, cp / 2.0, f"CP={cp}", ha="center", va="center", fontsize=9)
        ax.text(i, cp + fft_size / 2.0, f"NFFT={fft_size}", ha="center", va="center", fontsize=9)
    ax.set_title("Four OFDM Symbols inside One SS/PBCH Block")
    ax.set_xlabel("SS/PBCH block OFDM symbol index")
    ax.set_ylabel("Samples")
    ax.set_xticks(x)
    ax.set_xticklabels(["l=0", "l=1", "l=2", "l=3"])
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_cp_symbol_structure.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sync_timeline() -> None:
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")

    pss_timing = int(pss["finalBest"]["timingOffset"])
    sss_start = int(sss["bestSymbolStart"])
    expected_offset = int(sss["expectedSssOffsetSamples"])
    symbol_len = int(sss["symbolLengthSamples"])

    fig, ax = plt.subplots(figsize=(14, 3.8))
    ax.axvline(pss_timing, color="#1f77b4", linewidth=2.0, label="PSS timing anchor")
    ax.axvline(pss_timing + expected_offset, color="#2ca02c", linestyle="--", linewidth=2.0, label="Expected SSS start")
    ax.axvline(sss_start, color="#d62728", linewidth=2.0, label="Detected SSS start")
    ax.axvspan(pss_timing, pss_timing + symbol_len, alpha=0.12, color="#1f77b4", label="One OFDM-symbol span")
    ax.set_title("Timing Relationship from PSS Anchor to SSS Window")
    ax.set_xlabel("Sample index")
    ax.set_yticks([])
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_sync_timeline.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pss_candidate_comparison() -> None:
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    items = pss["nId2BestResults"]
    n_id2 = [item["nId2"] for item in items]
    peaks = [item["peakValue"] for item in items]
    timings = [item["timingOffset"] for item in items]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    bars = ax1.bar(n_id2, peaks, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax1.set_xlabel("N_ID2 hypothesis")
    ax1.set_ylabel("Peak correlation")
    ax1.set_title("PSS Candidate Comparison across N_ID2")
    ax1.grid(True, axis="y", alpha=0.3)
    for bar, timing in zip(bars, timings):
        ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"tim={timing}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_pss_candidate_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pss_cfo_estimators() -> None:
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    base = float(pss["finalBest"]["freqOffsetParabolicHz"])
    ls = float(pss["pssFreqOffsetEstimation"]["refinedFreqHz"])
    cp = float(pss["cpFreqOffsetEstimation"]["refinedFreqHz"])
    residual_ls = float(pss["pssFreqOffsetEstimation"]["residualFreqHz"])
    residual_cp = float(pss["cpFreqOffsetEstimation"]["residualFreqHz"])

    labels = ["Base PSS", "LS refined", "CP refined"]
    values = [base, ls, cp]
    colors = ["#4daf4a", "#377eb8", "#e41a1c"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(labels, values, color=colors)
    axes[0].set_title("Carrier Frequency Estimates")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(["LS residual", "CP residual"], [residual_ls, residual_cp], color=["#377eb8", "#e41a1c"])
    axes[1].set_title("Residual CFO after Base PSS Estimate")
    axes[1].set_ylabel("Residual frequency (Hz)")
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_pss_cfo_estimators.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sss_offset_zoom_and_profile() -> None:
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    data = np.load(OUTPUT_DIR / "nfft4096_best_normalcp_sss_sliding_correlation.npz")
    offsets = data["offsetGridSamples"]
    corr_max = data["corrMaxByOffset"]
    corr_matrix = data["corrMatrix"]
    best_offset = int(sss["bestOffsetSamples"])
    expected_offset = int(sss["expectedSssOffsetSamples"])
    best_nid1 = int(sss["nId1"])

    mask = (offsets >= best_offset - 64) & (offsets <= best_offset + 64)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].plot(offsets[mask], corr_max[mask], color="#1f77b4", linewidth=1.5)
    axes[0].axvline(best_offset, color="#d62728", linewidth=1.5, label="Best offset")
    axes[0].axvline(expected_offset, color="#2ca02c", linestyle="--", linewidth=1.2, label="Expected offset")
    axes[0].set_title("SSS Offset Envelope near Best Region")
    axes[0].set_xlabel("Offset from PSS timing (samples)")
    axes[0].set_ylabel("Max normalized correlation")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    row = int(np.argmin(np.abs(offsets - best_offset)))
    profile = corr_matrix[row, :]
    axes[1].plot(np.arange(len(profile)), profile, color="#ff7f0e", linewidth=1.5)
    axes[1].axvline(best_nid1, color="#d62728", linewidth=1.5, label="Best N_ID1")
    axes[1].set_title("SSS N_ID1 Correlation Profile at Best Offset")
    axes[1].set_xlabel("N_ID1")
    axes[1].set_ylabel("Normalized correlation")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_sss_offset_zoom_and_profile.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sss_candidate_comparison() -> None:
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    top = sss["topCandidates"][:8]
    verified = sss["verifiedCandidates"][:8]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].bar(
        [str(item["offsetSamples"]) for item in top],
        [item["score"] for item in top],
        color="#1f77b4",
    )
    axes[0].set_title("Top SSS Candidates by Time-domain Score")
    axes[0].set_xlabel("Offset samples")
    axes[0].set_ylabel("TD score")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(
        [str(item["offsetSamples"]) for item in verified],
        [item["fdScore"] for item in verified],
        color="#d62728",
    )
    axes[1].set_title("Verified SSS Candidates by Frequency-domain Score")
    axes[1].set_xlabel("Offset samples")
    axes[1].set_ylabel("FD score")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_sss_candidate_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_top_candidates() -> None:
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    top = pbch["topCandidates"][:10]

    labels = [f'{item["freqCompHz"]:.0f}' for item in top]
    evm = [item["evmPercent"] for item in top]
    dmrs = [item["dmrsCorrNorm"] for item in top]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(labels, evm, color="#ff7f0e")
    axes[0].set_title("PBCH Top Candidates by EVM")
    axes[0].set_xlabel("Compensated frequency (Hz)")
    axes[0].set_ylabel("EVM (%)")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(evm, dmrs, "o", color="#1f77b4")
    for idx, item in enumerate(top):
        axes[1].annotate(str(idx + 1), (evm[idx], dmrs[idx]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1].set_title("PBCH Candidate Tradeoff: EVM vs DMRS Correlation")
    axes[1].set_xlabel("EVM (%)")
    axes[1].set_ylabel("DMRS correlation norm")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_pbch_top_candidates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bch_attempts_and_bits() -> None:
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    attempts = bch["attempts"]
    best = bch["bestAttempt"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    x = np.arange(len(attempts))
    mean_abs = [item["meanAbsSoft"] for item in attempts]
    labels = [f'L{item["lssb"]}-v{item["v"]}' for item in attempts]
    colors = ["#d62728" if item["crcOk"] else "#7f7f7f" for item in attempts]
    axes[0].bar(x, mean_abs, color=colors)
    axes[0].set_title("Official BCH Attempts")
    axes[0].set_xlabel("Attempt index")
    axes[0].set_ylabel("Mean |soft bit|")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].grid(True, axis="y", alpha=0.3)

    scrambled = np.array(best["scrambledBlockBits"], dtype=np.int32)
    mib = np.array(best["mibBits"], dtype=np.int32)
    bit_view = np.vstack([scrambled, np.pad(mib, (0, max(0, len(scrambled) - len(mib))), constant_values=-1)])
    cmap = plt.matplotlib.colors.ListedColormap(["#1f77b4", "#ff7f0e", "#f0f0f0"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
    axes[1].imshow(bit_view, aspect="auto", cmap=cmap, norm=norm)
    axes[1].set_title("Best BCH Attempt Bit View")
    axes[1].set_xlabel("Bit index")
    axes[1].set_yticks([0, 1])
    axes[1].set_yticklabels(["Scrambled block", "MIB bits"])
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_bch_attempts_and_bits.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bch_decode_status_dashboard() -> None:
    """生成 BCH/DBCH 官方译码阶段状态总览图。"""
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    best = bch.get("bestAttempt", {})
    attempts = bch.get("attempts", [])

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    ax = axes[0, 0]
    stages = ["PSS", "SSS", "PBCH\nEVM", "BCH\nCRC"]
    status_values = [1.0, 1.0, 0.55, 0.0]
    colors = ["#2ca02c", "#2ca02c", "#ffbf00", "#d62728"]
    ax.bar(stages, status_values, color=colors)
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("阶段状态")
    ax.set_title("PSS -> SSS -> PBCH -> BCH 状态总览")
    ax.text(0, 1.03, "N_ID_2=0", ha="center", fontsize=9)
    ax.text(1, 1.03, "PCI=0", ha="center", fontsize=9)
    ax.text(2, 0.63, f"EVM={pbch.get('evmPercent', 0.0):.2f}%", ha="center", fontsize=9)
    ax.text(3, 0.08, "CRC fail", ha="center", fontsize=9)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[0, 1]
    metrics = [
        ("PBCH EVM(%)", float(pbch.get("evmPercent", 0.0))),
        ("DMRS EVM(%) /100", float(pbch.get("dmrsEvmPercent", 0.0)) / 100.0),
        ("mean|soft|", float(best.get("meanAbsSoft", 0.0))),
        ("max|soft|", float(best.get("maxAbsSoft", 0.0))),
    ]
    ax.bar([m[0] for m in metrics], [m[1] for m in metrics], color=["#ff7f0e", "#9467bd", "#1f77b4", "#17becf"])
    ax.set_title("BCH 输入质量相关指标")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)

    ax = axes[1, 0]
    if attempts:
        labels = [f"L{item.get('lssb')}-v{item.get('v')}" for item in attempts]
        mean_abs = [float(item.get("meanAbsSoft", 0.0)) for item in attempts]
        max_abs = [float(item.get("maxAbsSoft", 0.0)) for item in attempts]
        x = np.arange(len(labels))
        ax.plot(x, mean_abs, marker="o", label="mean |soft|", color="#1f77b4")
        ax.plot(x, max_abs, marker="s", label="max |soft|", color="#d62728")
        best_label = f"L{best.get('lssb')}-v{best.get('v')}"
        if best_label in labels:
            best_idx = labels.index(best_label)
            ax.axvline(best_idx, linestyle="--", color="#2ca02c", label="best attempt")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_title("官方 BCH 尝试软信息强度")
        ax.set_ylabel("软信息幅度")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")

    ax = axes[1, 1]
    ax.axis("off")
    summary_rows = [
        ("officialChain", str(bch.get("officialChain"))),
        ("crcOk", str(bch.get("crcOk"))),
        ("failureStage", str(bch.get("failureStage"))),
        ("best lssb", str(best.get("lssb"))),
        ("best v", str(best.get("v"))),
        ("hrf", str(best.get("hrf"))),
        ("ssbIndexMsb3", str(best.get("mib", {}).get("ssbIndexMsb3"))),
        ("noiseVar", f"{float(bch.get('noiseVar', 0.0)):.6f}"),
    ]
    table = ax.table(
        cellText=[[k, v] for k, v in summary_rows],
        colLabels=["字段", "结果"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)
    ax.set_title("BCH/DBCH 译码结果摘要", pad=18)

    fig.savefig(OUTPUT_DIR / "report_bch_decode_status_dashboard.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bch_attempt_soft_metrics() -> None:
    """生成 BCH/DBCH 各个 Lssb/v 尝试的软信息和 CRC 状态图。"""
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    attempts = bch.get("attempts", [])
    if not attempts:
        return

    labels = [f"L{item.get('lssb')}-v{item.get('v')}" for item in attempts]
    mean_abs = np.asarray([float(item.get("meanAbsSoft", 0.0)) for item in attempts])
    max_abs = np.asarray([float(item.get("maxAbsSoft", 0.0)) for item in attempts])
    crc = np.asarray([1 if item.get("crcOk") else 0 for item in attempts], dtype=np.float32)
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True, constrained_layout=True)
    axes[0].bar(x - 0.2, mean_abs, width=0.4, color="#1f77b4", label="mean |soft|")
    axes[0].bar(x + 0.2, max_abs, width=0.4, color="#ff7f0e", label="max |soft|")
    axes[0].set_ylabel("软信息幅度")
    axes[0].set_title("BCH/DBCH 官方链路各候选尝试软信息")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].bar(x, crc, color=["#2ca02c" if val > 0 else "#d62728" for val in crc])
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_ylabel("CRC OK")
    axes[1].set_xlabel("候选尝试")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("CRC 判决：当前所有候选均未通过")
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.savefig(OUTPUT_DIR / "report_bch_attempt_soft_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bch_best_attempt_bits_detail() -> None:
    """生成 BCH/DBCH 最佳失败尝试的比特细节图。"""
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    best = bch.get("bestAttempt", {})
    scrambled = np.asarray(best.get("scrambledBlockBits", []), dtype=np.int32)
    mib = np.asarray(best.get("mibBits", []), dtype=np.int32)
    sfn = np.asarray(best.get("sfnLsb4", []), dtype=np.int32)
    hrf = np.asarray([int(best.get("hrf", 0))], dtype=np.int32)
    max_len = max(len(scrambled), len(mib), len(sfn), len(hrf), 1)

    def pad_bits(bits: np.ndarray) -> np.ndarray:
        if len(bits) >= max_len:
            return bits[:max_len]
        return np.pad(bits, (0, max_len - len(bits)), constant_values=-1)

    bit_matrix = np.vstack([pad_bits(scrambled), pad_bits(mib), pad_bits(sfn), pad_bits(hrf)])
    cmap = plt.matplotlib.colors.ListedColormap(["#f2f2f2", "#1f77b4", "#ff7f0e"])
    norm = plt.matplotlib.colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

    fig, ax = plt.subplots(figsize=(16, 4.8))
    ax.imshow(bit_matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_title(
        f"BCH/DBCH最佳失败尝试比特细节 | Lssb={best.get('lssb')} v={best.get('v')} CRC={best.get('crcOk')}"
    )
    ax.set_xlabel("比特索引")
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(["scrambledBlockBits", "MIB bits", "SFN LSB4", "hrf"])
    ax.set_xticks(np.arange(0, max_len, 4))
    ax.grid(False)
    fig.savefig(OUTPUT_DIR / "report_bch_best_attempt_bits_detail.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_bch_mib_field_map() -> None:
    """生成 BCH/DBCH 最佳尝试 MIB 字段诊断图。CRC 未过时字段只作为诊断。"""
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    best = bch.get("bestAttempt", {})
    mib = best.get("mib", {})
    rows = [
        ("CRC状态", str(best.get("crcOk")), "未通过时字段不可靠"),
        ("failureStage", str(best.get("failureStage")), "官方链路失败层级"),
        ("Lssb", str(best.get("lssb")), "候选 SSB 最大数假设"),
        ("v", str(best.get("v")), "候选 SSB 索引低位/DMRS序列索引"),
        ("hrf", str(best.get("hrf")), "半帧指示诊断值"),
        ("systemFrameNumber10", str(mib.get("systemFrameNumber10")), "CRC fail，仅诊断"),
        ("subCarrierSpacingCommon", str(mib.get("subCarrierSpacingCommon")), "CRC fail，仅诊断"),
        ("kSsb", str(mib.get("kSsb")), "CRC fail，仅诊断"),
        ("ssbIndexMsb3", str(mib.get("ssbIndexMsb3")), "CRC fail，因此为 null/不可靠"),
        ("dmrsTypeAPosition", str(mib.get("dmrsTypeAPosition")), "CRC fail，仅诊断"),
        ("pdcchConfigSIB1", str(mib.get("pdcchConfigSIB1")), "CRC fail，仅诊断"),
        ("cellBarred", str(mib.get("cellBarred")), "CRC fail，仅诊断"),
    ]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["字段", "当前输出", "可靠性说明"],
        loc="center",
        cellLoc="center",
        colWidths=[0.24, 0.24, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.55)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif col == 2:
            cell.set_facecolor("#fff2cc")
    ax.set_title("BCH/DBCH 最佳尝试 MIB 字段诊断图", fontsize=14, pad=18)
    fig.savefig(OUTPUT_DIR / "report_bch_mib_field_map.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_bch_pipeline_summary() -> None:
    """生成 PBCH 到 BCH/DBCH 的数据流和失败定位图。"""
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")

    fig, ax = plt.subplots(figsize=(16, 5.8))
    ax.axis("off")
    nodes = [
        ("PBCH均衡符号\npbchEq=432", 0.08, "#cfe2f3"),
        (f"软比特\nnoiseVar={pbch.get('noiseVarEstimate', 0.0):.4f}", 0.27, "#d9ead3"),
        ("nrPBCHDecode\n官方工具", 0.46, "#d9ead3"),
        ("nrBCHDecode\nlistLength=8", 0.65, "#fff2cc"),
        ("CRC终判\ncrcOk=false", 0.84, "#f4cccc"),
    ]
    y = 0.58
    for text, x, color in nodes:
        rect = plt.matplotlib.patches.FancyBboxPatch(
            (x - 0.075, y - 0.12),
            0.15,
            0.24,
            boxstyle="round,pad=0.02",
            fc=color,
            ec="#555555",
            linewidth=1.1,
        )
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center", fontsize=10)
    for i in range(len(nodes) - 1):
        x0 = nodes[i][1] + 0.08
        x1 = nodes[i + 1][1] - 0.08
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.4))

    notes = [
        f"输入参数：N_ID_cell={pbch.get('nIdCell')}，iSsbBar={pbch.get('iSsbBar')}，EVM={pbch.get('evmPercent', 0.0):.2f}%",
        f"最佳尝试：Lssb={bch.get('bestAttempt', {}).get('lssb')}，v={bch.get('bestAttempt', {}).get('v')}，hrf={bch.get('bestAttempt', {}).get('hrf')}",
        "结论：官方链路已执行到 CRC，失败定位在 BCH CRC 终判，MIB字段不能作为可靠广播结果。",
    ]
    for idx, note in enumerate(notes):
        ax.text(0.08, 0.22 - idx * 0.08, note, ha="left", va="center", fontsize=10)

    ax.set_title("PBCH -> BCH/DBCH 官方译码数据流与失败定位", fontsize=14)
    fig.savefig(OUTPUT_DIR / "report_pbch_bch_pipeline_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_rx_diag_43_constellation() -> None:
    """复现固定参数诊断中 EVM=43.31% 的 PBCH 星座图。"""
    diag = load_json("rx_diag_check_current_report.json")
    params = diag["parameters"]
    rx_path = Path(params.get("signalPath", "data/rxSignal.npy"))
    if not rx_path.is_absolute():
        rx_path = ROOT_DIR / "data" / "rxSignal.npy"
    rx_signal = np.load(rx_path).astype(np.complex64).reshape(-1)

    cfg = createSsbConfigWithOverrides(
        sampleRate=float(params["sampleRate"]),
        subcarrierSpacing=int(params["scs"]),
        fftSize=int(params["nfft"]),
        ssbSubcarrierOffset=int(params["ssbSubcarrierOffset"]),
    )
    decoder = PbchDecoder(cfg, ssbIndexCandidates=[int(params["iSsbBar"])])
    cp_lengths = [int(v) for v in params["cpLengths"]]
    result = decoder._evaluateCandidate(  # noqa: SLF001
        rxSignal=rx_signal,
        nIdCell=int(params["nIdCell"]),
        ssbStart=int(params["ssbStart"]),
        freqCompHz=float(params["freqCompHz"]),
        iSsbBar=int(params["iSsbBar"]),
        cpProfileName=str(params["cpProfile"]),
        cpLengths=cp_lengths,
    )

    eq = np.asarray(result["pbchEq"], dtype=np.complex64)
    qpsk = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    axes[0].scatter(np.real(eq), np.imag(eq), s=7, alpha=0.28, color="#1f77b4")
    axes[0].scatter(np.real(qpsk), np.imag(qpsk), marker="x", s=130, c="#d62728", linewidths=2)
    axes[0].set_title(f"固定参数诊断 PBCH 星座 | EVM={result['evmPercent']:.2f}%")
    axes[0].set_xlabel("I")
    axes[0].set_ylabel("Q")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(True, alpha=0.25)

    labels = ["当前诊断", "主流程 normalcp"]
    values = [
        float(result["evmPercent"]),
        float(load_json("nfft4096_best_normalcp_pbch_demod_result.json")["evmPercent"]),
    ]
    axes[1].bar(labels, values, color=["#2ca02c", "#ff7f0e"])
    axes[1].set_ylabel("EVM (%)")
    axes[1].set_title("43.31% 诊断结果与主流程 48.11% 对比")
    axes[1].grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        axes[1].text(idx, value + 0.8, f"{value:.2f}%", ha="center", va="bottom")

    fig.savefig(OUTPUT_DIR / "report_pbch_constellation_evm43_diag.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_standard_vs_paper_flow() -> None:
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    std_steps = [
        "3GPP UE blind receive flow",
        "1. Search PSS on sync raster",
        "2. Timing alignment from PSS peak",
        "3. Frequency offset estimation / compensation",
        "4. Search SSS near PSS timing",
        "5. Recover N_IDcell",
        "6. PBCH demodulation and MIB decoding",
    ]
    paper_steps = [
        "Paper flow (Tuninato 2023)",
        "1. Front-end LPF + downsample",
        "2. PSS exhaustive correlation",
        "3. IFO grid search",
        "4. FFO estimate from half-symbol",
        "5. CFO compensation",
        "6. SSS brute / FWHT / partial / coherent",
        "7. PBCH + DMRS + BLER evaluation",
    ]

    for i, text in enumerate(std_steps):
        y = 0.92 - i * 0.11
        rect = plt.matplotlib.patches.FancyBboxPatch(
            (0.05, y - 0.045), 0.38, 0.075,
            boxstyle="round,pad=0.02", fc="#e8f1fb" if i else "#cfe2f3", ec="#4f81bd"
        )
        ax.add_patch(rect)
        ax.text(0.24, y - 0.005, text, ha="center", va="center", fontsize=10)

    for i, text in enumerate(paper_steps):
        y = 0.92 - i * 0.095
        rect = plt.matplotlib.patches.FancyBboxPatch(
            (0.57, y - 0.045), 0.38, 0.075,
            boxstyle="round,pad=0.02", fc="#fbe9e7" if i else "#f4cccc", ec="#cc4125"
        )
        ax.add_patch(rect)
        ax.text(0.76, y - 0.005, text, ha="center", va="center", fontsize=10)

    ax.annotate("", xy=(0.57, 0.82), xytext=(0.43, 0.82), arrowprops=dict(arrowstyle="<->", lw=1.5))
    ax.text(0.50, 0.85, "Our code maps standard flow to paper-inspired implementation", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_standard_vs_paper_flow.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_other_rx_constellations() -> None:
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    n_id_cell = int(pbch["nIdCell"])
    i_ssb_bar = int(pbch["iSsbBar"])
    ssb_start = int(pbch["ssbStart"])
    freq_hz = float(pbch["freqCompHz"])
    sample_rate = 122880000.0
    scs = 30000
    nfft = 4096
    ssb_offset = -7
    cfg = createSsbConfigWithOverrides(
        sampleRate=sample_rate,
        subcarrierSpacing=scs,
        fftSize=nfft,
        ssbSubcarrierOffset=ssb_offset,
    )
    decoder = PbchDecoder(cfg, ssbIndexCandidates=[i_ssb_bar])
    cp_profiles = {name: vals for name, vals in decoder._cpLengthProfiles()}  # noqa: SLF001
    cp_lengths = [int(v) for v in cp_profiles["all_normal"]]

    rx_main = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)
    main_result = decoder._evaluateCandidate(  # noqa: SLF001
        rxSignal=rx_main,
        nIdCell=n_id_cell,
        ssbStart=ssb_start,
        freqCompHz=freq_hz,
        iSsbBar=i_ssb_bar,
        cpProfileName="all_normal",
        cpLengths=cp_lengths,
    )
    eq_main = np.asarray(main_result["pbchEq"], dtype=np.complex64)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(np.real(eq_main), np.imag(eq_main), s=6, alpha=0.28, color="#1f77b4")
    q = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
    ax.scatter(np.real(q), np.imag(q), marker="x", s=120, c="red", linewidths=2)
    ax.set_title(f"PBCH Equalized Constellation of rxSignal.npy | EVM={main_result['evmPercent']:.2f}%")
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_rxsignal_pbch_constellation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    frame_paths = sorted((ROOT_DIR / "data").glob("rxsig0_frame*.npy"))
    if frame_paths:
        fig, axes = plt.subplots(2, 5, figsize=(16, 7), constrained_layout=True)
        axes = axes.ravel()
        for ax, path in zip(axes, frame_paths[:10]):
            x = np.load(path).astype(np.complex64).reshape(-1)
            try:
                res = decoder._evaluateCandidate(  # noqa: SLF001
                    rxSignal=x,
                    nIdCell=n_id_cell,
                    ssbStart=ssb_start,
                    freqCompHz=freq_hz,
                    iSsbBar=i_ssb_bar,
                    cpProfileName="all_normal",
                    cpLengths=cp_lengths,
                )
                eq = np.asarray(res["pbchEq"], dtype=np.complex64)
                ax.scatter(np.real(eq), np.imag(eq), s=2.0, alpha=0.22, color="#d62728")
                ax.set_title(f"{path.stem}\nEVM={res['evmPercent']:.1f}%", fontsize=8)
            except Exception:
                ax.text(0.5, 0.5, f"{path.stem}\nfail", ha="center", va="center", fontsize=10)
                ax.set_title(path.stem, fontsize=8)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.2)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Other RX Signal PBCH Equalized Constellations (same demodulation pipeline)", fontsize=14)
        fig.savefig(OUTPUT_DIR / "report_other_rx_frames_pbch_constellations.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_pss_sss_fs_comparison() -> None:
    pss_base = load_json("nfft4096_best_normalcp_pss_search_result.json")
    pss_alt = load_json("fs3072_scs30_mainstyle_pss_search_result.json")
    sss_base = load_json("nfft4096_best_normalcp_sss_search_result.json")
    sss_alt = load_json("fs3072_scs30_mainstyle_sss_search_result.json")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)

    # PSS comparison: N_ID2 peak values
    base_items = sorted(pss_base["nId2BestResults"], key=lambda x: int(x["nId2"]))
    alt_items = sorted(pss_alt["nId2BestResults"], key=lambda x: int(x["nId2"]))
    nids = [0, 1, 2]
    base_vals = [float(item["peakValue"]) for item in base_items]
    alt_vals = [float(item["peakValue"]) for item in alt_items]
    width = 0.36
    x = np.arange(len(nids))

    ax = axes[0, 0]
    ax.bar(x - width / 2.0, base_vals, width=width, color="#1f77b4", label="FS=122.88M, SCS=30k")
    ax.bar(x + width / 2.0, alt_vals, width=width, color="#ff7f0e", label="FS=30.72M, SCS=30k")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in nids])
    ax.set_xlabel("N_ID2")
    ax.set_ylabel("Peak correlation")
    ax.set_title("PSS: N_ID2 Candidate Peak Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # PSS comparison: key metrics
    ax = axes[0, 1]
    pss_base_best = pss_base["finalBest"]
    pss_alt_best = pss_alt["finalBest"]
    labels = ["Best N_ID2", "Best timing", "Best freq(kHz)"]
    base_key = [
        float(pss_base_best["nId2"]),
        float(pss_base_best["timingOffset"]),
        float(pss_base_best["freqOffsetParabolicHz"]) / 1e3,
    ]
    alt_key = [
        float(pss_alt_best["nId2"]),
        float(pss_alt_best["timingOffset"]),
        float(pss_alt_best["freqOffsetParabolicHz"]) / 1e3,
    ]
    xx = np.arange(len(labels))
    ax.bar(xx - width / 2.0, base_key, width=width, color="#1f77b4", label="FS=122.88M")
    ax.bar(xx + width / 2.0, alt_key, width=width, color="#ff7f0e", label="FS=30.72M")
    ax.set_xticks(xx)
    ax.set_xticklabels(labels)
    ax.set_title("PSS: Key Selected Parameters")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # SSS comparison: score and FD score
    ax = axes[1, 0]
    sss_labels = ["bestScore", "verifiedFdScore"]
    sss_base_vals = [float(sss_base["bestScore"]), float(sss_base.get("verifiedFdScore", 0.0))]
    sss_alt_vals = [float(sss_alt["bestScore"]), float(sss_alt.get("verifiedFdScore", 0.0))]
    xx = np.arange(len(sss_labels))
    ax.bar(xx - width / 2.0, sss_base_vals, width=width, color="#2ca02c", label="FS=122.88M")
    ax.bar(xx + width / 2.0, sss_alt_vals, width=width, color="#d62728", label="FS=30.72M")
    ax.set_xticks(xx)
    ax.set_xticklabels(sss_labels)
    ax.set_ylim(bottom=0.0)
    ax.set_title("SSS: Time/Frequency Verification Scores")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    # SSS comparison: timing relation
    ax = axes[1, 1]
    base_expected_start = int(sss_base["timingBase"]) + int(sss_base["expectedSssOffsetSamples"])
    alt_expected_start = int(sss_alt["timingBase"]) + int(sss_alt["expectedSssOffsetSamples"])
    base_delta = int(sss_base["bestSymbolStart"]) - base_expected_start
    alt_delta = int(sss_alt["bestSymbolStart"]) - alt_expected_start
    timing_labels = ["bestSymbolStart", "expectedStart", "delta"]
    base_timing = [
        float(sss_base["bestSymbolStart"]),
        float(base_expected_start),
        float(base_delta),
    ]
    alt_timing = [
        float(sss_alt["bestSymbolStart"]),
        float(alt_expected_start),
        float(alt_delta),
    ]
    xx = np.arange(len(timing_labels))
    ax.bar(xx - width / 2.0, base_timing, width=width, color="#2ca02c", label="FS=122.88M")
    ax.bar(xx + width / 2.0, alt_timing, width=width, color="#d62728", label="FS=30.72M")
    ax.set_xticks(xx)
    ax.set_xticklabels(timing_labels)
    ax.set_title("SSS: Timing Alignment Comparison")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")

    fig.suptitle("PSS/SSS Search Comparison: FS=122.88M vs FS=30.72M (SCS=30kHz)", fontsize=14)
    fig.savefig(OUTPUT_DIR / "report_pss_sss_fs_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_raw_search_figure_comparison() -> None:
    # Raw PSS heatmap comparison
    pss_base_img = plt.imread(OUTPUT_DIR / "nfft4096_best_normalcp_pss_freq_scan_heatmap.png")
    pss_alt_img = plt.imread(OUTPUT_DIR / "fs3072_scs30_mainstyle_pss_freq_scan_heatmap.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].imshow(pss_base_img)
    axes[0].axis("off")
    axes[0].set_title("PSS Heatmap | FS=122.88M, SCS=30k")
    axes[1].imshow(pss_alt_img)
    axes[1].axis("off")
    axes[1].set_title("PSS Heatmap | FS=30.72M, SCS=30k")
    fig.suptitle("Raw PSS Search Figure Comparison", fontsize=14)
    fig.savefig(OUTPUT_DIR / "report_pss_heatmap_fs_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Raw PSS time-scan comparison
    pss_base_time_img = plt.imread(OUTPUT_DIR / "nfft4096_best_normalcp_pss_nid2_time_scan.png")
    pss_alt_time_img = plt.imread(OUTPUT_DIR / "fs3072_scs30_mainstyle_pss_nid2_time_scan.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    axes[0].imshow(pss_base_time_img)
    axes[0].axis("off")
    axes[0].set_title("PSS Time Scan | FS=122.88M, SCS=30k")
    axes[1].imshow(pss_alt_time_img)
    axes[1].axis("off")
    axes[1].set_title("PSS Time Scan | FS=30.72M, SCS=30k")
    fig.suptitle("Raw PSS N_ID2 Time-domain Scan Comparison", fontsize=14)
    fig.savefig(OUTPUT_DIR / "report_pss_time_scan_fs_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Raw SSS sliding-search comparison
    sss_base_img = plt.imread(OUTPUT_DIR / "nfft4096_best_normalcp_sss_sliding_search.png")
    sss_alt_img = plt.imread(OUTPUT_DIR / "fs3072_scs30_mainstyle_sss_sliding_search.png")
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), constrained_layout=True)
    axes[0].imshow(sss_base_img)
    axes[0].axis("off")
    axes[0].set_title("SSS Sliding Search | FS=122.88M, SCS=30k")
    axes[1].imshow(sss_alt_img)
    axes[1].axis("off")
    axes[1].set_title("SSS Sliding Search | FS=30.72M, SCS=30k")
    fig.suptitle("Raw SSS Search Figure Comparison", fontsize=14)
    fig.savefig(OUTPUT_DIR / "report_sss_sliding_fs_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_case_c_gscn_grid_on_pss_axis() -> None:
    """按主 PSS 频域搜索图的 x 轴范围绘制 Case C GSCN 相对频率网格。"""
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")

    scs_hz = 30000.0
    ssb_subcarriers = 240.0
    pss_axis_min_khz = -(ssb_subcarriers * scs_hz / 2.0) / 1e3
    pss_axis_max_khz = +(ssb_subcarriers * scs_hz / 2.0) / 1e3

    center_gscn = 7881
    center_freq_mhz = GscnRaster.getAbsoluteFrequency(center_gscn) / 1e6
    gscn_step_hz = (
        GscnRaster.getAbsoluteFrequency(center_gscn + 1)
        - GscnRaster.getAbsoluteFrequency(center_gscn)
    )
    gscn_step_khz = gscn_step_hz / 1e3
    max_grid_index = int(np.floor(max(abs(pss_axis_min_khz), abs(pss_axis_max_khz)) / gscn_step_khz))
    relative_gscn_indices = np.arange(-max_grid_index, max_grid_index + 1)
    gscn_offsets_khz = relative_gscn_indices.astype(np.float64) * gscn_step_khz
    gscn_numbers = center_gscn + relative_gscn_indices

    pss_freq_khz = float(pss["finalBest"]["freqOffsetParabolicHz"]) / 1e3
    sss_freq_khz = float(sss["verifiedFreqCompHz"]) / 1e3
    pbch_freq_khz = float(pbch["freqCompHz"]) / 1e3
    nearest_grid_khz = float(gscn_offsets_khz[np.argmin(np.abs(gscn_offsets_khz - pss_freq_khz))])
    pss_delta_khz = pss_freq_khz - nearest_grid_khz

    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(pss_axis_min_khz, pss_axis_max_khz)
    ax.set_ylim(-0.35, 2.65)
    ax.set_yticks([])
    ax.set_xlabel("基带频偏 / kHz（与主 PSS 频域搜索图 x 轴一致）")
    ax.set_title("Case C GSCN 搜索频率网格与 PSS 频域搜索结果同轴对应")
    ax.grid(True, axis="x", alpha=0.22)

    ax.axhline(1.0, color="#4f6f8f", linewidth=1.4, alpha=0.75)
    ax.scatter(gscn_offsets_khz, np.ones_like(gscn_offsets_khz), s=90, color="#1f77b4", zorder=3)
    for offset_khz, gscn in zip(gscn_offsets_khz, gscn_numbers):
        ax.axvline(offset_khz, color="#1f77b4", linestyle="-", linewidth=0.8, alpha=0.22)
        ax.text(
            offset_khz,
            1.13,
            f"GSCN {int(gscn)}",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#1f4e79",
        )

    ax.axvspan(pss_axis_min_khz, pss_axis_max_khz, color="#f3f6f8", alpha=0.45, zorder=0)
    ax.axvline(pss_freq_khz, color="#d62728", linewidth=2.3, label=f"PSS 粗锁: {pss_freq_khz:.3f} kHz")
    ax.axvline(sss_freq_khz, color="#2ca02c", linestyle="--", linewidth=2.0, label=f"SSS 复核: {sss_freq_khz:.3f} kHz")
    ax.axvline(pbch_freq_khz, color="#9467bd", linestyle=":", linewidth=2.4, label=f"PBCH 候选: {pbch_freq_khz:.3f} kHz")

    ax.annotate(
        f"PSS 最终频偏 {pss_freq_khz:.3f} kHz\n相对最近示意 GSCN 栅格点偏差 {pss_delta_khz:.3f} kHz",
        xy=(pss_freq_khz, 1.0),
        xytext=(-1480, 2.18),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.4},
        fontsize=10,
        color="#7f1d1d",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff5f5", "edgecolor": "#d62728", "alpha": 0.9},
    )
    ax.annotate(
        f"主 PSS 频域搜索图数据范围\n{pss_axis_min_khz:.0f} kHz 到 {pss_axis_max_khz:.0f} kHz",
        xy=(pss_axis_max_khz, 0.18),
        xytext=(1200, 0.18),
        fontsize=10,
        ha="left",
        va="center",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#ffffff", "edgecolor": "#7f7f7f", "alpha": 0.9},
    )
    ax.text(
        pss_axis_min_khz,
        -0.18,
        (
            "说明：这里以 n78 Case C 的一个示意中心 GSCN 7881 作为 0 kHz 参考，"
            f"相邻 GSCN 间隔为 {gscn_step_khz:.0f} kHz。当前数据缺少 RF 中心频率，"
            "因此本图只展示基带 PSS 搜索频偏如何落到 Case C 栅格坐标上，"
            "不把 gscn=0 解释成真实 3GPP GSCN 编号。"
        ),
        ha="left",
        va="bottom",
        fontsize=10,
        color="#333333",
    )

    def rel_khz_to_abs_mhz(x):
        return center_freq_mhz + np.asarray(x, dtype=np.float64) / 1000.0

    def abs_mhz_to_rel_khz(x):
        return (np.asarray(x, dtype=np.float64) - center_freq_mhz) * 1000.0

    secax = ax.secondary_xaxis("top", functions=(rel_khz_to_abs_mhz, abs_mhz_to_rel_khz))
    secax.set_xlabel("示意 SSREF 绝对频率 / MHz（假设 0 kHz = n78 Case C GSCN 7881）")
    secax.set_xticks(center_freq_mhz + gscn_offsets_khz / 1000.0)
    secax.set_xticklabels([f"{center_freq_mhz + off / 1000.0:.2f}" for off in gscn_offsets_khz], rotation=0)

    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "report_case_c_gscn_grid_on_pss_axis.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_diag_pbch_decoder(ssb_subcarrier_offset: int) -> PbchDecoder:
    cfg = createSsbConfigWithOverrides(
        sampleRate=122880000.0,
        subcarrierSpacing=30000,
        fftSize=4096,
        ssbSubcarrierOffset=int(ssb_subcarrier_offset),
    )
    return PbchDecoder(cfg, ssbIndexCandidates=list(range(8)))


def _pbch_dmrs_metric(
    decoder: PbchDecoder,
    rx_signal: np.ndarray,
    n_id_cell: int,
    i_ssb_bar: int,
    ssb_start: int,
    freq_comp_hz: float,
    cp_lengths: list[int],
) -> dict:
    grid = decoder._extractSsbGrid(  # noqa: SLF001
        rx_signal,
        ssbStart=int(ssb_start),
        freqCompHz=float(freq_comp_hz),
        cpLengths=[int(v) for v in cp_lengths],
    )
    dmrs_items = decoder._dmrsReList(int(n_id_cell))  # noqa: SLF001
    dmrs_rx = decoder._extract(grid, dmrs_items)  # noqa: SLF001
    dmrs_ref = decoder.generatePbchDmrs(int(n_id_cell), int(i_ssb_bar))
    corr_norm = float(
        np.abs(np.vdot(dmrs_ref, dmrs_rx))
        / max(float(np.linalg.norm(dmrs_ref) * np.linalg.norm(dmrs_rx)), 1e-12)
    )
    h_mean = np.mean(dmrs_rx / dmrs_ref)
    if np.abs(h_mean) > 1e-9:
        dmrs_evm = float(
            100.0
            * np.sqrt(
                np.mean(np.abs(dmrs_rx / h_mean - dmrs_ref) ** 2)
                / max(float(np.mean(np.abs(dmrs_ref) ** 2)), 1e-12)
            )
        )
    else:
        dmrs_evm = 1e9
    return {"dmrsCorrNorm": corr_norm, "dmrsEvmPercent": dmrs_evm}


def plot_pbch_quality_breakpoint_metrics() -> None:
    """画出 PSS/SSS 很强，但 PBCH-DMRS/BCH 开始变差的阶段断点。"""
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    diag = load_json("rx_diag_check_current_report.json")

    pss_peaks = sorted(
        [float(item["peakValue"]) for item in pss["nId2BestResults"]],
        reverse=True,
    )
    pss_ratio = pss_peaks[0] / max(pss_peaks[1], 1e-12)
    sss_fd = float(sss.get("verifiedFdScore", 0.0))
    diag_dmrs = float(diag["channel"]["dmrsCorrNorm"])
    main_dmrs = float(pbch.get("dmrsCorrNorm", 0.0))
    pbch_evm = float(pbch.get("evmPercent", 0.0))
    crc_ok = 1.0 if pbch.get("bchDecode", {}).get("crcOk", False) else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    ax = axes[0]
    labels = ["PSS峰值比\nbest/second", "SSS频域复核", "固定诊断\nDMRS相关", "主流程\nPBCH-DMRS相关", "BCH CRC"]
    values = [pss_ratio, sss_fd, diag_dmrs, main_dmrs, crc_ok]
    colors = ["#2ca02c", "#2ca02c", "#ffbf00", "#d62728", "#d62728"]
    ax.bar(labels, values, color=colors)
    ax.set_title("同步链路质量断点：PBCH-DMRS 入口开始明显变差")
    ax.set_ylabel("指标值（按各自含义读取）")
    ax.grid(True, axis="y", alpha=0.25)
    for i, value in enumerate(values):
        ax.text(i, value + max(values) * 0.025, f"{value:.4g}", ha="center", va="bottom", fontsize=9)

    ax = axes[1]
    bad_labels = ["主流程PBCH EVM(%)", "主流程DMRS EVM(%) /100", "固定诊断PBCH EVM(%)"]
    bad_values = [
        pbch_evm,
        float(pbch.get("dmrsEvmPercent", 0.0)) / 100.0,
        float(diag["pbch"]["evmVariants"]["currentDecoder"]["evmPercent"]),
    ]
    ax.bar(bad_labels, bad_values, color=["#ff7f0e", "#9467bd", "#17becf"])
    ax.set_title("PBCH 之后的质量指标已经不健康")
    ax.set_ylabel("百分比或缩放百分比")
    ax.grid(True, axis="y", alpha=0.25)
    for i, value in enumerate(bad_values):
        ax.text(i, value + max(bad_values) * 0.025, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.savefig(OUTPUT_DIR / "report_pbch_quality_breakpoint_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_dmrs_frequency_mismatch() -> None:
    """在同一个 PBCH/DMRS 提取口径下扫频，显示主流程频偏附近没有好 DMRS 峰。"""
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    diag = load_json("rx_diag_check_current_report.json")
    params = diag["parameters"]

    rx_signal = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)
    decoder = _make_diag_pbch_decoder(int(params["ssbSubcarrierOffset"]))
    cp_lengths = [int(v) for v in params["cpLengths"]]
    n_id_cell = int(params["nIdCell"])
    i_ssb_bar = int(params["iSsbBar"])
    ssb_start = int(params["ssbStart"])

    freq_grid = np.arange(-160000.0, 80000.0 + 2500.0, 2500.0, dtype=np.float64)
    corr = []
    dmrs_evm = []
    for freq_hz in freq_grid:
        metric = _pbch_dmrs_metric(
            decoder,
            rx_signal,
            n_id_cell=n_id_cell,
            i_ssb_bar=i_ssb_bar,
            ssb_start=ssb_start,
            freq_comp_hz=float(freq_hz),
            cp_lengths=cp_lengths,
        )
        corr.append(metric["dmrsCorrNorm"])
        dmrs_evm.append(metric["dmrsEvmPercent"])
    corr = np.asarray(corr, dtype=np.float64)
    dmrs_evm = np.asarray(dmrs_evm, dtype=np.float64)
    best_idx = int(np.argmax(corr))

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(freq_grid / 1e3, corr, marker=".", linewidth=1.4, color="#1f77b4")
    ax.axvline(float(pss["finalBest"]["freqOffsetParabolicHz"]) / 1e3, color="#d62728", linestyle="--", label="PSS频偏")
    ax.axvline(float(sss["verifiedFreqCompHz"]) / 1e3, color="#2ca02c", linestyle="--", label="SSS复核频偏")
    ax.axvline(float(pbch["freqCompHz"]) / 1e3, color="#9467bd", linestyle=":", linewidth=2.0, label="主流程PBCH频偏")
    ax.axvline(float(params["freqCompHz"]) / 1e3, color="#ff7f0e", linestyle="-", linewidth=2.0, label="固定诊断频偏")
    ax.scatter(freq_grid[best_idx] / 1e3, corr[best_idx], s=90, color="#111111", zorder=4, label=f"本扫描最高 {corr[best_idx]:.3f}")
    ax.set_ylabel("PBCH DMRS 归一化相关")
    ax.set_title("PBCH-DMRS 对频偏补偿很敏感：主流程频偏附近相关明显偏低")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=3, fontsize=9)

    ax = axes[1]
    ax.plot(freq_grid / 1e3, dmrs_evm, marker=".", linewidth=1.2, color="#9467bd")
    ax.axvline(float(pbch["freqCompHz"]) / 1e3, color="#9467bd", linestyle=":", linewidth=2.0, label="主流程PBCH频偏")
    ax.axvline(float(params["freqCompHz"]) / 1e3, color="#ff7f0e", linestyle="-", linewidth=2.0, label="固定诊断频偏")
    ax.set_xlabel("频偏补偿 / kHz")
    ax.set_ylabel("DMRS EVM (%)")
    ax.set_ylim(0, min(float(np.nanmax(dmrs_evm)), 1200.0))
    ax.set_title("同一扫描下的 DMRS EVM：低相关位置对应导频补偿残差大")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.savefig(OUTPUT_DIR / "report_pbch_dmrs_frequency_mismatch.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_dmrs_local_search_heatmap() -> None:
    """围绕 SSS 交接给 PBCH 的频偏/定时局部搜索，展示没有强 DMRS 峰。"""
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    rx_signal = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)
    decoder = _make_diag_pbch_decoder(-7)
    cp_profiles = {name: vals for name, vals in decoder._cpLengthProfiles()}  # noqa: SLF001

    n_id_cell = int(sss["nIdCell"])
    f0 = float(sss["verifiedFreqCompHz"])
    anchor_start = int(sss["verifiedSymbolStart"]) - (4096 + 288 + 4096 + 288)
    starts = np.arange(anchor_start - 8, anchor_start + 8 + 1, dtype=np.int32)
    freqs = f0 + np.arange(-800.0, 800.0 + 100.0, 100.0, dtype=np.float64)
    heat = np.zeros((len(starts), len(freqs)), dtype=np.float64)
    best_ib = np.zeros_like(heat, dtype=np.int32)

    for i, start in enumerate(starts):
        for j, freq_hz in enumerate(freqs):
            best_corr = -1.0
            best_i = 0
            for cp_lengths in cp_profiles.values():
                for i_ssb_bar in range(8):
                    metric = _pbch_dmrs_metric(
                        decoder,
                        rx_signal,
                        n_id_cell=n_id_cell,
                        i_ssb_bar=i_ssb_bar,
                        ssb_start=int(start),
                        freq_comp_hz=float(freq_hz),
                        cp_lengths=[int(v) for v in cp_lengths],
                    )
                    if metric["dmrsCorrNorm"] > best_corr:
                        best_corr = metric["dmrsCorrNorm"]
                        best_i = int(i_ssb_bar)
            heat[i, j] = best_corr
            best_ib[i, j] = best_i

    best_pos = np.unravel_index(int(np.argmax(heat)), heat.shape)
    best_start = int(starts[best_pos[0]])
    best_freq = float(freqs[best_pos[1]])
    best_corr = float(heat[best_pos])

    fig, ax = plt.subplots(figsize=(16, 7))
    im = ax.imshow(
        heat,
        origin="lower",
        aspect="auto",
        extent=[freqs[0] / 1e3, freqs[-1] / 1e3, starts[0], starts[-1]],
        cmap="viridis",
        vmin=0.0,
        vmax=max(0.5, float(np.max(heat))),
    )
    fig.colorbar(im, ax=ax, label="max DMRS corrNorm over iSsbBar and CP profile")
    ax.scatter(best_freq / 1e3, best_start, s=90, c="red", marker="x", linewidths=2.0)
    ax.annotate(
        f"局部最高 corr={best_corr:.3f}\nstart={best_start}, freq={best_freq/1e3:.3f} kHz\niSsbBar={int(best_ib[best_pos])}",
        xy=(best_freq / 1e3, best_start),
        xytext=(freqs[0] / 1e3 + 0.08, starts[-1] - 2),
        arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1.4},
        color="white",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#000000", "edgecolor": "white", "alpha": 0.65},
    )
    ax.set_xlabel("围绕 SSS 复核频偏的 PBCH 频偏补偿 / kHz")
    ax.set_ylabel("候选 SSB 起点采样")
    ax.set_title("PBCH-DMRS 局部搜索热图：SSS 交接频偏附近没有形成强相关峰")
    ax.grid(False)
    fig.savefig(OUTPUT_DIR / "report_pbch_dmrs_local_search_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_issb_candidate_contrast() -> None:
    """比较主流程频偏口径和固定诊断频偏口径下的 iSsbBar DMRS 相关。"""
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    diag = load_json("rx_diag_check_current_report.json")
    params = diag["parameters"]
    rx_signal = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)

    decoder = _make_diag_pbch_decoder(int(params["ssbSubcarrierOffset"]))
    n_id_cell = int(params["nIdCell"])
    cp_slot = [int(v) for v in params["cpLengths"]]
    cp_all = [288, 288, 288, 288]

    diag_corr = []
    main_corr = []
    for i_ssb_bar in range(8):
        diag_corr.append(
            _pbch_dmrs_metric(
                decoder,
                rx_signal,
                n_id_cell=n_id_cell,
                i_ssb_bar=i_ssb_bar,
                ssb_start=int(params["ssbStart"]),
                freq_comp_hz=float(params["freqCompHz"]),
                cp_lengths=cp_slot,
            )["dmrsCorrNorm"]
        )
        main_corr.append(
            _pbch_dmrs_metric(
                decoder,
                rx_signal,
                n_id_cell=n_id_cell,
                i_ssb_bar=i_ssb_bar,
                ssb_start=int(pbch["ssbStart"]),
                freq_comp_hz=float(pbch["freqCompHz"]),
                cp_lengths=cp_all,
            )["dmrsCorrNorm"]
        )

    x = np.arange(8)
    width = 0.38
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, main_corr, width=width, label="主流程PBCH频偏口径", color="#d62728")
    ax.bar(x + width / 2, diag_corr, width=width, label="固定诊断频偏口径", color="#2ca02c")
    ax.axhline(0.3, color="#666666", linestyle="--", linewidth=1.0, label="直观可用参考线 corr=0.3")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("iSsbBar / PBCH DMRS 序列索引候选")
    ax.set_ylabel("DMRS 归一化相关")
    ax.set_title("iSsbBar 搜索对比：主流程频偏口径下所有 DMRS 序列都不够强")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    for i, value in enumerate(main_corr):
        ax.text(i - width / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(diag_corr):
        ax.text(i + width / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(OUTPUT_DIR / "report_pbch_issb_candidate_contrast.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_time_domain_correlation_searches() -> None:
    """生成真正以采样点/候选起点为横轴的互相关搜索图。"""
    pss = load_json("nfft4096_best_normalcp_pss_search_result.json")
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    diag = load_json("rx_diag_check_current_report.json")
    params = diag["parameters"]

    rx_signal = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)

    fig, axes = plt.subplots(3, 1, figsize=(16, 13), constrained_layout=True)

    pss_cfg = createSsbConfigWithOverrides(
        sampleRate=122880000.0,
        subcarrierSpacing=30000,
        fftSize=4096,
        ssbSubcarrierOffset=0,
    )
    pss_searcher = PssBasebandSearcher(pss_cfg, freqMinHz=-3.6e6, freqMaxHz=3.6e6, useGpu=False)
    pss_freq_hz = float(pss["finalBest"]["freqOffsetParabolicHz"])
    pss_timing = int(pss["finalBest"]["timingOffset"])
    pss_plot_len = int(pss.get("overlayPeakInFirstSamples", {}).get("sampleCount", 15000))
    pss_x = np.arange(pss_plot_len, dtype=np.int32)
    for n_id_2, color in zip([0, 1, 2], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        corr = pss_searcher.getCorrelationAtFreq(rx_signal, pss_freq_hz, n_id_2)[:pss_plot_len]
        axes[0].plot(pss_x, corr, linewidth=1.0, color=color, label=f"N_ID_2={n_id_2}")
    axes[0].axvline(pss_timing, color="#d62728", linestyle="--", linewidth=1.6, label=f"PSS峰值 sample={pss_timing}")
    axes[0].set_title("PSS时域滑动互相关：横轴=候选PSS起点采样，纵轴=|rx与PSS时域模板互相关|")
    axes[0].set_xlabel("候选 PSS 起点采样")
    axes[0].set_ylabel("PSS原始互相关幅度")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=4)

    sss_npz = np.load(OUTPUT_DIR / "nfft4096_best_normalcp_sss_sliding_correlation.npz")
    sss_offsets = np.asarray(sss_npz["offsetGridSamples"], dtype=np.int32)
    sss_corr = np.asarray(sss_npz["corrMaxByOffset"], dtype=np.float32)
    sss_abs_start = int(sss["timingBase"]) + sss_offsets
    sss_best_start = int(sss["verifiedSymbolStart"])
    sss_expected_start = int(sss["timingBase"]) + int(sss["expectedSssOffsetSamples"])
    axes[1].plot(sss_abs_start, sss_corr, color="#1f77b4", linewidth=1.2, label="max over N_ID_1")
    axes[1].axvline(sss_best_start, color="#d62728", linestyle="--", linewidth=1.6, label=f"SSS复核起点={sss_best_start}")
    axes[1].axvline(sss_expected_start, color="#2ca02c", linestyle=":", linewidth=1.6, label=f"理论起点={sss_expected_start}")
    axes[1].set_xlim(sss_best_start - 450, sss_best_start + 450)
    axes[1].set_title("SSS时域滑动互相关：横轴=候选SSS符号起点采样，纵轴=max_N_ID1 归一化互相关")
    axes[1].set_xlabel("候选 SSS 符号起点采样")
    axes[1].set_ylabel("SSS归一化互相关")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    decoder = _make_diag_pbch_decoder(int(params["ssbSubcarrierOffset"]))
    starts = np.arange(0, 91, dtype=np.int32)
    main_corr = []
    diag_corr = []
    main_cp = [288, 288, 288, 288]
    diag_cp = [int(v) for v in params["cpLengths"]]
    for start in starts:
        best_main = 0.0
        for i_ssb_bar in range(8):
            metric = _pbch_dmrs_metric(
                decoder,
                rx_signal,
                n_id_cell=int(params["nIdCell"]),
                i_ssb_bar=i_ssb_bar,
                ssb_start=int(start),
                freq_comp_hz=float(pbch["freqCompHz"]),
                cp_lengths=main_cp,
            )
            best_main = max(best_main, float(metric["dmrsCorrNorm"]))
        main_corr.append(best_main)
        diag_corr.append(
            _pbch_dmrs_metric(
                decoder,
                rx_signal,
                n_id_cell=int(params["nIdCell"]),
                i_ssb_bar=int(params["iSsbBar"]),
                ssb_start=int(start),
                freq_comp_hz=float(params["freqCompHz"]),
                cp_lengths=diag_cp,
            )["dmrsCorrNorm"]
        )
    main_corr = np.asarray(main_corr, dtype=np.float64)
    diag_corr = np.asarray(diag_corr, dtype=np.float64)
    axes[2].plot(
        starts,
        main_corr,
        color="#d62728",
        linewidth=1.6,
        marker=".",
        label=f"主流程PBCH频偏 {float(pbch['freqCompHz'])/1e3:.3f} kHz，max over iSsbBar",
    )
    axes[2].plot(
        starts,
        diag_corr,
        color="#2ca02c",
        linewidth=1.6,
        marker=".",
        label=f"固定诊断频偏 {float(params['freqCompHz'])/1e3:.3f} kHz，iSsbBar={int(params['iSsbBar'])}",
    )
    axes[2].axvline(int(pbch["ssbStart"]), color="#d62728", linestyle="--", linewidth=1.4, label=f"主流程start={int(pbch['ssbStart'])}")
    axes[2].axvline(int(params["ssbStart"]), color="#2ca02c", linestyle=":", linewidth=1.4, label=f"固定诊断start={int(params['ssbStart'])}")
    axes[2].set_title("PBCH-DMRS时域候选起点互相关：横轴=候选SSB起点采样，纵轴=DMRS归一化互相关")
    axes[2].set_xlabel("候选 SS/PBCH block 起点采样")
    axes[2].set_ylabel("PBCH-DMRS归一化互相关")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.savefig(OUTPUT_DIR / "report_time_domain_correlation_searches.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_domain_search_curves() -> None:
    """生成真正以频偏为横轴的频率搜索图。"""
    sss = load_json("nfft4096_best_normalcp_sss_search_result.json")
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    diag = load_json("rx_diag_check_current_report.json")
    params = diag["parameters"]

    rx_signal = np.load(ROOT_DIR / "data" / "rxSignal.npy").astype(np.complex64).reshape(-1)
    decoder = _make_diag_pbch_decoder(int(params["ssbSubcarrierOffset"]))

    fig, axes = plt.subplots(3, 1, figsize=(16, 15), constrained_layout=True)

    pss_img = plt.imread(OUTPUT_DIR / "nfft4096_best_normalcp_pss_freq_scan_heatmap.png")
    axes[0].imshow(pss_img)
    axes[0].axis("off")
    axes[0].set_title("PSS频率搜索原图：横轴=频偏kHz，纵轴=PSS互相关峰值")

    n_id_cell = int(params["nIdCell"])
    i_ssb_bar = int(params["iSsbBar"])
    ssb_start = int(params["ssbStart"])
    cp_lengths = [int(v) for v in params["cpLengths"]]
    broad_freqs = np.arange(-160000.0, 80000.0 + 2500.0, 2500.0, dtype=np.float64)
    broad_corr = []
    for freq_hz in broad_freqs:
        broad_corr.append(
            _pbch_dmrs_metric(
                decoder,
                rx_signal,
                n_id_cell=n_id_cell,
                i_ssb_bar=i_ssb_bar,
                ssb_start=ssb_start,
                freq_comp_hz=float(freq_hz),
                cp_lengths=cp_lengths,
            )["dmrsCorrNorm"]
        )
    broad_corr = np.asarray(broad_corr, dtype=np.float64)
    best_broad = int(np.argmax(broad_corr))
    axes[1].plot(broad_freqs / 1e3, broad_corr, color="#1f77b4", linewidth=1.5, marker=".")
    axes[1].scatter(broad_freqs[best_broad] / 1e3, broad_corr[best_broad], color="#111111", s=80, zorder=4, label=f"本扫描最高={broad_corr[best_broad]:.3f}")
    axes[1].axvline(float(pbch["freqCompHz"]) / 1e3, color="#d62728", linestyle="--", linewidth=1.6, label=f"主流程PBCH频偏={float(pbch['freqCompHz'])/1e3:.3f} kHz")
    axes[1].axvline(float(params["freqCompHz"]) / 1e3, color="#2ca02c", linestyle=":", linewidth=1.8, label=f"固定诊断频偏={float(params['freqCompHz'])/1e3:.3f} kHz")
    axes[1].set_title("PBCH-DMRS宽频偏搜索：横轴=频偏补偿kHz，纵轴=固定DMRS序列归一化互相关")
    axes[1].set_xlabel("频偏补偿 / kHz")
    axes[1].set_ylabel("PBCH-DMRS归一化互相关")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper left")

    f0 = float(sss["verifiedFreqCompHz"])
    anchor_start = int(sss["verifiedSymbolStart"]) - (4096 + 288 + 4096 + 288)
    local_starts = np.arange(anchor_start - 8, anchor_start + 8 + 1, dtype=np.int32)
    local_freqs = f0 + np.arange(-800.0, 800.0 + 100.0, 100.0, dtype=np.float64)
    cp_profiles = {name: vals for name, vals in decoder._cpLengthProfiles()}  # noqa: SLF001
    local_best_corr = []
    for freq_hz in local_freqs:
        best_corr = 0.0
        for start in local_starts:
            for cp in cp_profiles.values():
                for i_ssb in range(8):
                    metric = _pbch_dmrs_metric(
                        decoder,
                        rx_signal,
                        n_id_cell=n_id_cell,
                        i_ssb_bar=i_ssb,
                        ssb_start=int(start),
                        freq_comp_hz=float(freq_hz),
                        cp_lengths=[int(v) for v in cp],
                    )
                    best_corr = max(best_corr, float(metric["dmrsCorrNorm"]))
        local_best_corr.append(best_corr)
    local_best_corr = np.asarray(local_best_corr, dtype=np.float64)
    best_local = int(np.argmax(local_best_corr))
    axes[2].plot(local_freqs / 1e3, local_best_corr, color="#9467bd", linewidth=1.6, marker=".")
    axes[2].scatter(local_freqs[best_local] / 1e3, local_best_corr[best_local], color="#111111", s=80, zorder=4, label=f"局部最高={local_best_corr[best_local]:.3f}")
    axes[2].axvline(f0 / 1e3, color="#2ca02c", linestyle="--", linewidth=1.6, label=f"SSS交接频偏={f0/1e3:.3f} kHz")
    axes[2].axvline(float(pbch["freqCompHz"]) / 1e3, color="#d62728", linestyle=":", linewidth=1.6, label=f"主流程PBCH频偏={float(pbch['freqCompHz'])/1e3:.3f} kHz")
    axes[2].set_title("PBCH-DMRS局部频偏搜索：横轴=SSS频偏附近候选kHz，纵轴=max_{start,iSsbBar,CP} DMRS互相关")
    axes[2].set_xlabel("频偏补偿 / kHz")
    axes[2].set_ylabel("PBCH-DMRS归一化互相关")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper left")

    fig.savefig(OUTPUT_DIR / "report_frequency_domain_search_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _evm_percent(eq: np.ndarray, ref: np.ndarray) -> float:
    """按保存的均衡符号和硬判决参考符号复算 EVM 百分比。"""
    numerator = float(np.mean(np.abs(eq - ref) ** 2))
    denominator = max(float(np.mean(np.abs(ref) ** 2)), 1e-12)
    return float(100.0 * np.sqrt(numerator / denominator))


def _load_saved48_pbch_artifact() -> tuple[dict, dict, np.lib.npyio.NpzFile]:
    """读取主流程 48% EVM 的历史保存 PBCH 工件，不调用当前解码器重算。"""
    pbch = load_json("nfft4096_best_normalcp_pbch_demod_result.json")
    bch = load_json("nfft4096_best_normalcp_bch_official_result.json")
    symbols = np.load(OUTPUT_DIR / "nfft4096_best_normalcp_pbch_demod_symbols.npz")
    return pbch, bch, symbols


def plot_pbch_saved48_constellation_and_metrics() -> None:
    """展示主流程保存工件中的 48.11% PBCH 星座和关键指标。"""
    pbch, bch, symbols = _load_saved48_pbch_artifact()
    eq = np.asarray(symbols["pbchEq"], dtype=np.complex64)
    ref = np.asarray(symbols["pbchHardRef"], dtype=np.complex64)
    evm_from_npz = _evm_percent(eq, ref)
    qpsk = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

    ax = axes[0]
    ax.scatter(eq.real, eq.imag, s=9, alpha=0.32, color="#1f77b4", label="保存的 pbchEq")
    ax.scatter(qpsk.real, qpsk.imag, marker="x", s=150, c="#d62728", linewidths=2.2, label="理想 QPSK 点")
    ax.set_title(f"主流程保存 PBCH 星座：EVM={float(pbch['evmPercent']):.2f}%")
    ax.set_xlabel("同相分量 I")
    ax.set_ylabel("正交分量 Q")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    ax = axes[1]
    ax.axis("off")
    crc_ok = bool(bch.get("crcOk", False))
    rows = [
        ("保存 JSON EVM", f"{float(pbch['evmPercent']):.6f}%"),
        ("NPZ 符号复算 EVM", f"{evm_from_npz:.6f}%"),
        ("N_ID_cell / PCI", str(pbch.get("nIdCell"))),
        ("iSsbBar", str(pbch.get("iSsbBar"))),
        ("SSB 起点", f"{int(pbch.get('ssbStart'))} samples"),
        ("频偏补偿", f"{float(pbch.get('freqCompHz')):.6f} Hz"),
        ("CP profile", str(pbch.get("cpProfile"))),
        ("DMRS 归一化互相关", f"{float(pbch.get('dmrsCorrNorm')):.8f}"),
        ("DMRS EVM", f"{float(pbch.get('dmrsEvmPercent')):.4f}%"),
        ("BCH CRC", "通过" if crc_ok else "未通过"),
        ("图像口径", "历史保存工件，未重新解码"),
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["字段", "结果"],
        cellLoc="left",
        colLoc="center",
        bbox=[0.02, 0.04, 0.96, 0.88],
        colWidths=[0.42, 0.54],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#d9eaf7")
            cell.set_text_props(weight="bold")
        elif row in (1, 2):
            cell.set_facecolor("#fff2cc")
        elif row in (8, 9, 10):
            cell.set_facecolor("#f4cccc")
    ax.set_title("48% 主流程 PBCH 工件关键指标", fontsize=14, pad=14)

    fig.savefig(OUTPUT_DIR / "report_pbch_saved48_constellation_and_metrics.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_saved48_evm_by_symbol() -> None:
    """按 PBCH OFDM 符号拆分保存工件，观察 48% EVM 从哪些符号贡献而来。"""
    pbch, _, symbols = _load_saved48_pbch_artifact()
    eq = np.asarray(symbols["pbchEq"], dtype=np.complex64)
    ref = np.asarray(symbols["pbchHardRef"], dtype=np.complex64)
    data_re = np.asarray(symbols["dataRe"], dtype=np.int16)
    qpsk = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)

    symbol_ids = [1, 2, 3]
    evm_by_symbol = []
    count_by_symbol = []
    for symbol_id in symbol_ids:
        mask = data_re[:, 1] == symbol_id
        evm_by_symbol.append(_evm_percent(eq[mask], ref[mask]))
        count_by_symbol.append(int(np.sum(mask)))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    bar_ax = axes[0, 0]
    x = np.arange(len(symbol_ids))
    bars = bar_ax.bar(x, evm_by_symbol, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    bar_ax.axhline(float(pbch["evmPercent"]), color="#d62728", linestyle="--", linewidth=1.4, label=f"整体 EVM={float(pbch['evmPercent']):.2f}%")
    bar_ax.set_xticks(x)
    bar_ax.set_xticklabels([f"l={value}" for value in symbol_ids])
    bar_ax.set_title("主流程 48% 工件：按 PBCH 符号拆分 EVM")
    bar_ax.set_xlabel("SS/PBCH block 内 OFDM 符号索引")
    bar_ax.set_ylabel("EVM (%)")
    bar_ax.grid(True, axis="y", alpha=0.25)
    bar_ax.legend(loc="upper left")
    for bar, evm_value, count in zip(bars, evm_by_symbol, count_by_symbol):
        bar_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{evm_value:.2f}%\n{count} RE",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    max_abs = float(np.max(np.abs(eq))) * 1.12
    for ax, symbol_id, evm_value, count in zip([axes[0, 1], axes[1, 0], axes[1, 1]], symbol_ids, evm_by_symbol, count_by_symbol):
        mask = data_re[:, 1] == symbol_id
        ax.scatter(eq[mask].real, eq[mask].imag, s=12, alpha=0.38, color="#1f77b4")
        ax.scatter(qpsk.real, qpsk.imag, marker="x", s=130, c="#d62728", linewidths=2)
        ax.set_title(f"l={symbol_id} 星座：EVM={evm_value:.2f}%，{count} 个 PBCH RE")
        ax.set_xlabel("同相分量 I")
        ax.set_ylabel("正交分量 Q")
        ax.set_xlim(-max_abs, max_abs)
        ax.set_ylim(-max_abs, max_abs)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

    fig.savefig(OUTPUT_DIR / "report_pbch_saved48_evm_by_symbol.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_saved48_error_vector_by_subcarrier() -> None:
    """按 SSB 内子载波展示 48% 主流程工件的 PBCH 误差向量幅度。"""
    pbch, _, symbols = _load_saved48_pbch_artifact()
    eq = np.asarray(symbols["pbchEq"], dtype=np.complex64)
    ref = np.asarray(symbols["pbchHardRef"], dtype=np.complex64)
    data_re = np.asarray(symbols["dataRe"], dtype=np.int16)
    error_mag = np.abs(eq - ref)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)

    ax = axes[0]
    colors = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
    for symbol_id in [1, 2, 3]:
        mask = data_re[:, 1] == symbol_id
        order = np.argsort(data_re[mask, 0])
        k = data_re[mask, 0][order]
        err = error_mag[mask][order]
        ax.plot(k, err, marker=".", linewidth=1.1, alpha=0.85, color=colors[symbol_id], label=f"l={symbol_id}")
    ax.axhline(float(np.median(error_mag)), color="#d62728", linestyle="--", linewidth=1.2, label=f"全体中位数={float(np.median(error_mag)):.3f}")
    ax.set_title("主流程 48% 工件：PBCH 误差向量幅度频域分布")
    ax.set_xlabel("SSB 内子载波索引 k")
    ax.set_ylabel("|pbchEq - pbchHardRef|")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", ncol=4)

    ax = axes[1]
    unique_k = np.unique(data_re[:, 0])
    mean_err_by_k = np.zeros_like(unique_k, dtype=np.float64)
    max_err_by_k = np.zeros_like(unique_k, dtype=np.float64)
    for idx, k_value in enumerate(unique_k):
        mask = data_re[:, 0] == k_value
        mean_err_by_k[idx] = float(np.mean(error_mag[mask]))
        max_err_by_k[idx] = float(np.max(error_mag[mask]))
    ax.plot(unique_k, mean_err_by_k, color="#9467bd", linewidth=1.5, label="同一子载波上的平均误差")
    ax.fill_between(unique_k, mean_err_by_k, max_err_by_k, color="#c5b0d5", alpha=0.35, label="平均误差到最大误差范围")
    ax.axhline(float(np.mean(error_mag)), color="#d62728", linestyle="--", linewidth=1.2, label=f"全体均值={float(np.mean(error_mag)):.3f}")
    ax.set_title("按子载波聚合后的 PBCH 误差向量：显示频域上误差并非集中在单一点")
    ax.set_xlabel("SSB 内子载波索引 k")
    ax.set_ylabel("误差向量幅度")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")

    fig.suptitle(
        f"数据来源：nfft4096_best_normalcp 保存工件，整体 EVM={float(pbch['evmPercent']):.2f}%，不使用当前代码复算",
        fontsize=13,
    )
    fig.savefig(OUTPUT_DIR / "report_pbch_saved48_error_vector_by_subcarrier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pbch_saved48_top_candidate_trace() -> None:
    """只使用保存 JSON 中的 topCandidates 绘制 48% 主流程 PBCH 搜索结果摘要。"""
    pbch, _, _ = _load_saved48_pbch_artifact()
    top = pbch.get("topCandidates", [])
    candidate_index = np.arange(1, len(top) + 1, dtype=np.int32)
    freq_khz = np.asarray([float(item["freqCompHz"]) / 1e3 for item in top], dtype=np.float64)
    evm = np.asarray([float(item["evmPercent"]) for item in top], dtype=np.float64)
    dmrs_corr = np.asarray([float(item["dmrsCorrNorm"]) for item in top], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    ax = axes[0]
    ax.plot(candidate_index, evm, marker="o", color="#ff7f0e", linewidth=1.6, label="PBCH EVM")
    ax.set_title("主流程 48% 工件保存的 Top 候选 EVM")
    ax.set_xlabel("保存的 Top 候选序号")
    ax.set_ylabel("PBCH EVM (%)")
    ax.grid(True, alpha=0.25)
    best_text = (
        f"最终候选\n"
        f"iSsbBar={pbch.get('iSsbBar')}\n"
        f"start={pbch.get('ssbStart')}\n"
        f"freq={float(pbch.get('freqCompHz'))/1e3:.3f} kHz"
    )
    ax.annotate(
        best_text,
        xy=(candidate_index[0], evm[0]),
        xytext=(candidate_index[min(4, len(candidate_index) - 1)], float(np.max(evm)) + 0.002),
        arrowprops={"arrowstyle": "->", "color": "#d62728", "linewidth": 1.2},
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#fff5f5", "edgecolor": "#d62728", "alpha": 0.92},
    )

    ax = axes[1]
    scatter = ax.scatter(freq_khz, dmrs_corr, c=evm, cmap="viridis_r", s=90, edgecolors="#333333")
    ax.plot(freq_khz, dmrs_corr, color="#777777", linewidth=0.8, alpha=0.7)
    ax.axvline(float(pbch.get("freqCompHz")) / 1e3, color="#d62728", linestyle="--", linewidth=1.4, label="最终候选频偏")
    ax.set_title("主流程 48% 工件保存的 Top 候选 DMRS 相关")
    ax.set_xlabel("频偏补偿 / kHz")
    ax.set_ylabel("DMRS 归一化互相关")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("PBCH EVM (%)")

    fig.savefig(OUTPUT_DIR / "report_pbch_saved48_top_candidate_trace.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_matplotlib_font()
    ensure_output_dir()
    plot_standard_vs_paper_flow()
    plot_ssb_resource_map()
    plot_pss_sequences_and_mapping()
    plot_pbch_dmrs_pattern()
    plot_cp_symbol_structure()
    plot_sync_timeline()
    plot_pss_candidate_comparison()
    plot_pss_cfo_estimators()
    plot_sss_offset_zoom_and_profile()
    plot_sss_candidate_comparison()
    plot_pbch_top_candidates()
    plot_bch_attempts_and_bits()
    plot_bch_decode_status_dashboard()
    plot_bch_attempt_soft_metrics()
    plot_bch_best_attempt_bits_detail()
    plot_bch_mib_field_map()
    plot_pbch_bch_pipeline_summary()
    plot_rx_diag_43_constellation()
    plot_other_rx_constellations()
    plot_pss_sss_fs_comparison()
    plot_raw_search_figure_comparison()
    plot_case_c_gscn_grid_on_pss_axis()
    plot_pbch_quality_breakpoint_metrics()
    plot_pbch_dmrs_frequency_mismatch()
    plot_pbch_dmrs_local_search_heatmap()
    plot_pbch_issb_candidate_contrast()
    plot_time_domain_correlation_searches()
    plot_frequency_domain_search_curves()
    plot_pbch_saved48_constellation_and_metrics()
    plot_pbch_saved48_evm_by_symbol()
    plot_pbch_saved48_error_vector_by_subcarrier()
    plot_pbch_saved48_top_candidate_trace()


if __name__ == "__main__":
    main()
