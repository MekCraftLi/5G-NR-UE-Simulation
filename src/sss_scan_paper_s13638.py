import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SSS scan following methods discussed in DOI:10.1186/s13638-023-02317-5"
    )
    p.add_argument("--input-path", type=str, required=True)
    p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--scs", type=int, required=True)
    p.add_argument("--nfft", type=int, required=True)
    p.add_argument("--timing-offset", type=int, required=True, help="PSS symbol-0 start index (CP start)")
    p.add_argument("--ssb-subcarrier-offset", type=int, required=True)
    p.add_argument("--nid2", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--scan-radius", type=int, default=6000)
    p.add_argument("--scan-step", type=int, default=2)
    p.add_argument("--freq-center-hz", type=float, default=0.0)
    p.add_argument("--freq-search-hz", type=float, default=0.0)
    p.add_argument("--freq-step-hz", type=float, default=1000.0)
    p.add_argument("--partial-blocks", type=int, default=3, help="M in Eq.(23)")
    p.add_argument("--unique-guard", type=int, default=160)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--output-prefix", type=str, default="sss_paper_s13638_scan")
    return p.parse_args()


def standard_cp(fs: float, scs: int, nfft: int) -> tuple[int, int, int]:
    mu_float = np.log2(float(scs) / 15000.0)
    if abs(mu_float - round(mu_float)) > 1e-9:
        raise ValueError(f"Invalid SCS for NR numerology: {scs}")
    mu = int(round(mu_float))
    cp_other = int(round(144 * int(nfft) / 2048))
    slot_samples = int(round(float(fs) * (1e-3 / (2 ** mu))))
    total_cp = slot_samples - 14 * int(nfft)
    cp0 = int(total_cp - 13 * cp_other) if mu > 0 else int(round((total_cp - 12 * cp_other) / 2.0))
    if cp0 <= 0 or cp_other <= 0:
        raise ValueError(f"Invalid CP computed: cp0={cp0}, cpOther={cp_other}")
    return cp0, cp_other, mu


def build_mseq(tap_a: int, tap_b: int) -> np.ndarray:
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0] = 1
    for n in range(127):
        x[n + 7] = (x[n + tap_a] + x[n + tap_b]) & 1
    return x[:127].copy()


def pss_seq(nid2: int) -> np.ndarray:
    x = build_mseq(4, 0)
    n = np.arange(127, dtype=np.int32)
    seq = 1.0 - 2.0 * x[(n + 43 * int(nid2)) % 127].astype(np.float32)
    return seq.astype(np.complex64)


def sss_bank(nid2: int) -> np.ndarray:
    x0 = build_mseq(4, 0)
    x1 = build_mseq(1, 0)
    n = np.arange(127, dtype=np.int32)
    bank = np.zeros((336, 127), dtype=np.complex64)
    for nid1 in range(336):
        m0 = 15 * (nid1 // 112) + 5 * int(nid2)
        m1 = nid1 % 112
        s0 = 1.0 - 2.0 * x0[(n + m0) % 127].astype(np.float32)
        s1 = 1.0 - 2.0 * x1[(n + m1) % 127].astype(np.float32)
        bank[nid1, :] = (s0 * s1).astype(np.complex64)
    return bank


def split_blocks(length: int, n_blocks: int) -> list[tuple[int, int]]:
    idx = np.array_split(np.arange(length), n_blocks)
    return [(int(v[0]), int(v[-1]) + 1) for v in idx if len(v) > 0]


def eval_metrics(
    r_sss: np.ndarray,
    r_pss: np.ndarray,
    d_pss: np.ndarray,
    bank: np.ndarray,
    bank_diff: np.ndarray,
    blocks: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    # Eq.(18)-(19): brute-force non-coherent full-block (normalized)
    r_sss_norm2 = float(np.vdot(r_sss, r_sss).real)
    bank_norm2 = np.sum(np.abs(bank) ** 2, axis=1).astype(np.float64)
    brute_num = np.abs(bank @ np.conjugate(r_sss)) ** 2
    brute = brute_num / np.maximum(bank_norm2 * r_sss_norm2, 1e-12)

    # Eq.(22): differential non-coherent (normalized)
    rdiff = r_sss[1:] * np.conjugate(r_sss[:-1])
    rdiff_norm2 = float(np.vdot(rdiff, rdiff).real)
    bank_diff_norm2 = np.sum(np.abs(bank_diff) ** 2, axis=1).astype(np.float64)
    diff_num = np.abs(bank_diff @ np.conjugate(rdiff)) ** 2
    diff = diff_num / np.maximum(bank_diff_norm2 * rdiff_norm2, 1e-12)

    # Eq.(23): non-coherent partial-block (block-wise normalized)
    partial = np.zeros(336, dtype=np.float64)
    for b0, b1 in blocks:
        chunk = r_sss[b0:b1]
        chunk_norm2 = float(np.vdot(chunk, chunk).real)
        bank_chunk_norm2 = np.sum(np.abs(bank[:, b0:b1]) ** 2, axis=1).astype(np.float64)
        part_num = np.abs(bank[:, b0:b1] @ np.conjugate(chunk)) ** 2
        partial += part_num / np.maximum(bank_chunk_norm2 * chunk_norm2, 1e-12)

    # Eq.(24): coherent by PSS channel estimate (BPSK PSS => d_pss in {+1,-1}, no zero denom)
    h_pss = r_pss * np.conjugate(d_pss)
    r_coh = r_sss * np.conjugate(h_pss)
    r_coh_norm2 = float(np.vdot(r_coh, r_coh).real)
    coh_num = np.abs(bank @ np.conjugate(r_coh)) ** 2
    coherent = coh_num / np.maximum(bank_norm2 * r_coh_norm2, 1e-12)

    return {
        "brute": brute.astype(np.float64),
        "differential": diff.astype(np.float64),
        "partial": partial.astype(np.float64),
        "coherent": coherent.astype(np.float64),
    }


def symbol_fft(
    signal: np.ndarray,
    start_cp: int,
    cp_len: int,
    nfft: int,
    fs: float,
    freq_hz: float,
) -> np.ndarray | None:
    if start_cp < 0:
        return None
    useful_start = start_cp + cp_len
    useful_end = useful_start + nfft
    if useful_end > len(signal):
        return None
    idx = np.arange(useful_start, useful_end, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * float(freq_hz) * idx / float(fs)).astype(np.complex64)
    return np.fft.fftshift(np.fft.fft(signal[useful_start:useful_end] * phase, nfft))


def summarize_method(
    score_map: np.ndarray,
    freq_idx_map: np.ndarray,
    t_grid: np.ndarray,
    f_grid: np.ndarray,
    expected_symbol2: int,
    nid2: int,
    unique_guard: int,
    top_k: int,
) -> dict:
    flat_best = int(np.argmax(score_map))
    best_nid1, best_col = np.unravel_index(flat_best, score_map.shape)
    best_score = float(score_map[best_nid1, best_col])
    best_sym = int(t_grid[best_col])
    best_freq = float(f_grid[int(freq_idx_map[best_nid1, best_col])])

    expected_col = int(np.argmin(np.abs(t_grid - expected_symbol2)))
    exp_nid1 = int(np.argmax(score_map[:, expected_col]))
    exp_score = float(score_map[exp_nid1, expected_col])

    best_by_t = np.max(score_map, axis=0)
    work = best_by_t.copy()
    left = max(0, best_col - int(unique_guard))
    right = min(len(work), best_col + int(unique_guard) + 1)
    work[left:right] = 0.0
    second_col = int(np.argmax(work))
    second_score = float(work[second_col])
    uniqueness = float(best_score / max(second_score, 1e-12))

    flat = score_map.ravel()
    idxs = np.argpartition(flat, -top_k)[-top_k:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    top = []
    seen = set()
    for idx in idxs:
        nid1, col = np.unravel_index(int(idx), score_map.shape)
        sym = int(t_grid[col])
        key = (nid1, sym)
        if key in seen:
            continue
        seen.add(key)
        top.append(
            {
                "nId1": int(nid1),
                "nIdCell": int(3 * int(nid1) + int(nid2)),
                "symbolStart": sym,
                "deltaFromExpected": int(sym - expected_symbol2),
                "score": float(score_map[nid1, col]),
                "freqHz": float(f_grid[int(freq_idx_map[nid1, col])]),
            }
        )
        if len(top) >= top_k:
            break

    return {
        "best": {
            "nId1": int(best_nid1),
            "nIdCell": int(3 * int(best_nid1) + int(nid2)),
            "symbolStart": int(best_sym),
            "deltaFromExpected": int(best_sym - expected_symbol2),
            "score": float(best_score),
            "freqHz": float(best_freq),
        },
        "expectedBest": {
            "nId1": int(exp_nid1),
            "nIdCell": int(3 * int(exp_nid1) + int(nid2)),
            "symbolStart": int(t_grid[expected_col]),
            "deltaFromExpected": int(t_grid[expected_col] - expected_symbol2),
            "score": float(exp_score),
        },
        "secondByTiming": {
            "symbolStart": int(t_grid[second_col]),
            "deltaFromExpected": int(t_grid[second_col] - expected_symbol2),
            "score": float(second_score),
        },
        "uniquenessRatio": float(uniqueness),
        "bestByTimingStats": {
            "max": float(np.max(best_by_t)),
            "mean": float(np.mean(best_by_t)),
            "std": float(np.std(best_by_t)),
        },
        "topCandidates": top,
    }


def plot_methods(score_maps: dict[str, np.ndarray], t_grid: np.ndarray, expected: int, out_png: Path) -> None:
    methods = ["brute", "differential", "partial", "coherent"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    x = t_grid - expected
    for ax, m in zip(axes.ravel(), methods):
        s = score_maps[m]
        best_by_t = np.max(s, axis=0)
        ax.plot(x, best_by_t, linewidth=1.1, label=m)
        ax.axvline(0, color="tab:orange", linestyle="--", linewidth=1.0)
        ax.set_title(f"{m}: max score over N_ID_1")
        ax.set_xlabel("symbol-2 start delta (samples)")
        ax.set_ylabel("score")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_heatmaps(score_maps: dict[str, np.ndarray], t_grid: np.ndarray, expected: int, out_png: Path) -> None:
    methods = ["brute", "differential", "partial", "coherent"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)
    x = t_grid - expected
    for ax, m in zip(axes.ravel(), methods):
        s = score_maps[m]
        im = ax.imshow(
            s,
            aspect="auto",
            origin="lower",
            extent=[int(x[0]), int(x[-1]), 0, 335],
            cmap="magma",
        )
        ax.axvline(0, color="cyan", linestyle="--", linewidth=1.0)
        ax.set_title(f"{m}: SSS score heatmap")
        ax.set_xlabel("symbol-2 start delta (samples)")
        ax.set_ylabel("N_ID_1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    signal = np.asarray(np.load(args.input_path), dtype=np.complex64).reshape(-1)
    cp0, cp_other, mu = standard_cp(float(args.sample_rate), int(args.scs), int(args.nfft))

    expected_symbol2 = int(args.timing_offset) + (int(args.nfft) + int(cp0)) + (int(args.nfft) + int(cp_other))
    t_grid = np.arange(
        expected_symbol2 - int(args.scan_radius),
        expected_symbol2 + int(args.scan_radius) + 1,
        int(args.scan_step),
        dtype=np.int32,
    )
    # both symbol 0 and symbol 2 must be valid
    sym0_shift = (int(args.nfft) + int(cp0)) + (int(args.nfft) + int(cp_other))
    valid = []
    for t2 in t_grid:
        t0 = int(t2) - sym0_shift
        cond = (
            t0 >= 0
            and t0 + int(cp0) + int(args.nfft) <= len(signal)
            and int(t2) + int(cp_other) + int(args.nfft) <= len(signal)
        )
        if cond:
            valid.append(int(t2))
    t_grid = np.asarray(valid, dtype=np.int32)
    if len(t_grid) == 0:
        raise ValueError("No valid timing candidates in search window")

    if abs(float(args.freq_search_hz)) <= 0.0:
        f_grid = np.asarray([float(args.freq_center_hz)], dtype=np.float64)
    else:
        step = abs(float(args.freq_step_hz))
        f_grid = float(args.freq_center_hz) + np.arange(
            -abs(float(args.freq_search_hz)),
            abs(float(args.freq_search_hz)) + step / 2.0,
            step,
            dtype=np.float64,
        )

    nfft = int(args.nfft)
    ssb_bin_start = nfft // 2 - 120 + int(args.ssb_subcarrier_offset)
    pss_bin_start = ssb_bin_start + 56
    sss_bin_start = ssb_bin_start + 56
    if ssb_bin_start < 0 or ssb_bin_start + 240 > nfft:
        raise ValueError(f"SSB bins out of FFT: ssbBinStart={ssb_bin_start}, nfft={nfft}")

    d_pss = pss_seq(int(args.nid2))
    bank = sss_bank(int(args.nid2))
    bank_diff = bank[:, 1:] * np.conjugate(bank[:, :-1])
    blocks = split_blocks(127, int(args.partial_blocks))

    methods = ["brute", "differential", "partial", "coherent"]
    score_maps = {m: np.zeros((336, len(t_grid)), dtype=np.float64) for m in methods}
    freq_idx_maps = {m: np.zeros((336, len(t_grid)), dtype=np.int16) for m in methods}

    for col, t2 in enumerate(t_grid):
        t0 = int(t2) - sym0_shift
        best = {m: np.full(336, -1.0, dtype=np.float64) for m in methods}
        best_f = {m: np.zeros(336, dtype=np.int16) for m in methods}

        for fi, f_hz in enumerate(f_grid):
            spec0 = symbol_fft(signal, t0, int(cp0), nfft, float(args.sample_rate), float(f_hz))
            spec2 = symbol_fft(signal, int(t2), int(cp_other), nfft, float(args.sample_rate), float(f_hz))
            if spec0 is None or spec2 is None:
                continue
            r_pss = spec0[pss_bin_start:pss_bin_start + 127].astype(np.complex64)
            r_sss = spec2[sss_bin_start:sss_bin_start + 127].astype(np.complex64)
            metrics = eval_metrics(r_sss, r_pss, d_pss, bank, bank_diff, blocks)

            for m in methods:
                better = metrics[m] > best[m]
                best[m][better] = metrics[m][better]
                best_f[m][better] = int(fi)

        for m in methods:
            score_maps[m][:, col] = best[m]
            freq_idx_maps[m][:, col] = best_f[m]

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    prefix = args.output_prefix
    fig_curve = out_dir / f"{prefix}_curves.png"
    fig_heatmap = out_dir / f"{prefix}_heatmaps.png"
    out_json = out_dir / f"{prefix}.json"
    plot_methods(score_maps, t_grid, expected_symbol2, fig_curve)
    plot_heatmaps(score_maps, t_grid, expected_symbol2, fig_heatmap)

    methods_out = {}
    for m in methods:
        methods_out[m] = summarize_method(
            score_map=score_maps[m],
            freq_idx_map=freq_idx_maps[m],
            t_grid=t_grid,
            f_grid=f_grid,
            expected_symbol2=expected_symbol2,
            nid2=int(args.nid2),
            unique_guard=int(args.unique_guard),
            top_k=int(args.top_k),
        )

    out = {
        "inputPath": str(Path(args.input_path).resolve()),
        "sampleRate": float(args.sample_rate),
        "scs": int(args.scs),
        "mu": int(mu),
        "nfft": int(args.nfft),
        "cp0": int(cp0),
        "cpOther": int(cp_other),
        "timingOffset": int(args.timing_offset),
        "ssbSubcarrierOffset": int(args.ssb_subcarrier_offset),
        "nId2": int(args.nid2),
        "expectedSymbol2Start": int(expected_symbol2),
        "scanRadius": int(args.scan_radius),
        "scanStep": int(args.scan_step),
        "timingGridStart": int(t_grid[0]),
        "timingGridEnd": int(t_grid[-1]),
        "timingGridCount": int(len(t_grid)),
        "freqCenterHz": float(args.freq_center_hz),
        "freqSearchHz": float(args.freq_search_hz),
        "freqStepHz": float(args.freq_step_hz),
        "freqGrid": [float(v) for v in f_grid.tolist()],
        "partialBlocks": int(args.partial_blocks),
        "paperRef": {
            "doi": "10.1186/s13638-023-02317-5",
            "equations": {
                "brute": "Eq.(18)-(19)",
                "differential": "Eq.(22)",
                "partial": "Eq.(23), M blocks",
                "coherent": "Eq.(24) + Eq.(18)-(19)",
            },
        },
        "methods": methods_out,
        "figures": {
            "curves": str(fig_curve.resolve()),
            "heatmaps": str(fig_heatmap.resolve()),
        },
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "json": str(out_json.resolve()),
        "figures": out["figures"],
        "best": {m: out["methods"][m]["best"] for m in methods},
        "uniqueness": {m: out["methods"][m]["uniquenessRatio"] for m in methods},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
