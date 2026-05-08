import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone SSS search from scratch (no project module dependency)")
    p.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    p.add_argument("--sample-rate", type=float, required=True)
    p.add_argument("--scs", type=int, required=True)
    p.add_argument("--nfft", type=int, required=True)
    p.add_argument("--timing-offset", type=int, required=True, help="PSS timing offset (sample index, 0-based)")
    p.add_argument("--ssb-subcarrier-offset", type=int, required=True, help="SSB shift from centered 240-subcarrier block")
    p.add_argument("--nid2", type=int, required=True, choices=[0, 1, 2])
    p.add_argument("--freq-center-hz", type=float, default=0.0, help="Base CFO compensation (Hz)")
    p.add_argument("--freq-search-hz", type=float, default=0.0, help="Residual CFO half range (Hz)")
    p.add_argument("--freq-step-hz", type=float, default=1000.0)
    p.add_argument("--scan-radius", type=int, default=6000, help="Timing search radius around expected symbol-2 start")
    p.add_argument("--scan-step", type=int, default=2)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--unique-guard", type=int, default=160)
    p.add_argument("--output-prefix", type=str, default="sss_scratch_scan")
    return p.parse_args()


def standard_cp(fs: float, scs: int, nfft: int) -> tuple[int, int, int]:
    mu_float = np.log2(float(scs) / 15000.0)
    if abs(mu_float - round(mu_float)) > 1e-9:
        raise ValueError(f"Invalid NR numerology SCS={scs}")
    mu = int(round(mu_float))
    cp_other = int(round(144 * int(nfft) / 2048))
    slot_samples = int(round(float(fs) * (1e-3 / (2 ** mu))))
    total_cp = slot_samples - 14 * int(nfft)
    if mu == 0:
        cp0 = int(round((total_cp - 12 * cp_other) / 2.0))
    else:
        cp0 = int(total_cp - 13 * cp_other)
    if cp0 <= 0 or cp_other <= 0:
        raise ValueError(f"Invalid CP computed: cp0={cp0}, cpOther={cp_other}")
    return cp0, cp_other, mu


def build_mseq(tap_a: int, tap_b: int) -> np.ndarray:
    seq = np.zeros(127 + 7, dtype=np.int8)
    seq[0] = 1
    for n in range(127):
        seq[n + 7] = (seq[n + tap_a] + seq[n + tap_b]) & 1
    return seq[:127].copy()


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


def corr_against_bank(rx_sss: np.ndarray, bank: np.ndarray) -> np.ndarray:
    rx_norm = float(np.linalg.norm(rx_sss))
    bank_norm = np.linalg.norm(bank, axis=1)
    corr = np.abs(bank @ np.conjugate(rx_sss))
    return (corr / np.maximum(bank_norm * rx_norm, 1e-12)).astype(np.float32)


def scan_sss(
    signal: np.ndarray,
    fs: float,
    nfft: int,
    cp0: int,
    cp_other: int,
    timing_offset: int,
    ssb_offset: int,
    nid2: int,
    freq_center_hz: float,
    freq_search_hz: float,
    freq_step_hz: float,
    scan_radius: int,
    scan_step: int,
) -> dict:
    ssb_bin_start = int(nfft) // 2 - 120 + int(ssb_offset)
    sss_bin_start = ssb_bin_start + 56
    if ssb_bin_start < 0 or ssb_bin_start + 240 > int(nfft):
        raise ValueError(f"SSB bins out of FFT range: ssbBinStart={ssb_bin_start}, nfft={nfft}")

    expected_symbol2 = int(timing_offset) + (int(nfft) + int(cp0)) + (int(nfft) + int(cp_other))
    t_grid = np.arange(
        expected_symbol2 - int(scan_radius),
        expected_symbol2 + int(scan_radius) + 1,
        int(scan_step),
        dtype=np.int32,
    )
    t_grid = t_grid[(t_grid >= 0) & (t_grid + int(cp_other) + int(nfft) <= len(signal))]
    if len(t_grid) == 0:
        raise ValueError("No valid timing candidates in search range")

    if abs(float(freq_search_hz)) <= 0.0:
        f_grid = np.asarray([float(freq_center_hz)], dtype=np.float64)
    else:
        step = abs(float(freq_step_hz))
        f_grid = float(freq_center_hz) + np.arange(-abs(float(freq_search_hz)), abs(float(freq_search_hz)) + step / 2.0, step, dtype=np.float64)

    bank = sss_bank(int(nid2))
    score_map = np.zeros((336, len(t_grid)), dtype=np.float32)
    best_freq_idx = np.zeros((336, len(t_grid)), dtype=np.int16)

    for col, sym_start in enumerate(t_grid):
        useful_start = int(sym_start) + int(cp_other)
        idx = np.arange(useful_start, useful_start + int(nfft), dtype=np.float64)
        best_corr = np.zeros(336, dtype=np.float32)
        best_fidx = np.zeros(336, dtype=np.int16)
        for fi, freq_hz in enumerate(f_grid):
            phase = np.exp(-1j * 2.0 * np.pi * float(freq_hz) * idx / float(fs)).astype(np.complex64)
            spec = np.fft.fftshift(np.fft.fft(signal[useful_start:useful_start + int(nfft)] * phase, int(nfft)))
            rx_sss = spec[sss_bin_start:sss_bin_start + 127].astype(np.complex64)
            corr = corr_against_bank(rx_sss, bank)
            better = corr > best_corr
            best_corr[better] = corr[better]
            best_fidx[better] = int(fi)
        score_map[:, col] = best_corr
        best_freq_idx[:, col] = best_fidx

    flat_best = int(np.argmax(score_map))
    best_nid1, best_col = np.unravel_index(flat_best, score_map.shape)
    best_score = float(score_map[best_nid1, best_col])
    best_symbol_start = int(t_grid[best_col])
    best_freq_hz = float(f_grid[int(best_freq_idx[best_nid1, best_col])])

    expected_col = int(np.argmin(np.abs(t_grid - expected_symbol2)))
    expected_best_nid1 = int(np.argmax(score_map[:, expected_col]))
    expected_best_score = float(score_map[expected_best_nid1, expected_col])

    best_by_timing = np.max(score_map, axis=0)
    best_nid1_by_timing = np.argmax(score_map, axis=0)
    best_nid1_score_by_timing = score_map[best_nid1_by_timing, np.arange(score_map.shape[1])]
    best_nid1_freq_by_timing = f_grid[best_freq_idx[best_nid1_by_timing, np.arange(score_map.shape[1])]]

    work = best_by_timing.copy()
    left = max(0, best_col - int(unique_guard := 160))
    right = min(len(work), best_col + int(unique_guard) + 1)
    work[left:right] = 0.0
    second_col = int(np.argmax(work))
    second_score = float(work[second_col])
    uniqueness = float(best_score / max(second_score, 1e-12))

    return {
        "scoreMap": score_map,
        "bestFreqIdx": best_freq_idx,
        "timingGrid": t_grid,
        "freqGrid": f_grid,
        "expectedSymbol2Start": int(expected_symbol2),
        "best": {
            "nId1": int(best_nid1),
            "nIdCell": int(3 * int(best_nid1) + int(nid2)),
            "symbolStart": int(best_symbol_start),
            "deltaFromExpected": int(best_symbol_start - expected_symbol2),
            "score": float(best_score),
            "freqHz": float(best_freq_hz),
        },
        "expectedBest": {
            "nId1": int(expected_best_nid1),
            "nIdCell": int(3 * int(expected_best_nid1) + int(nid2)),
            "symbolStart": int(t_grid[expected_col]),
            "deltaFromExpected": int(t_grid[expected_col] - expected_symbol2),
            "score": float(expected_best_score),
        },
        "secondByTiming": {
            "symbolStart": int(t_grid[second_col]),
            "deltaFromExpected": int(t_grid[second_col] - expected_symbol2),
            "score": float(second_score),
        },
        "uniquenessRatio": float(uniqueness),
        "bestByTiming": best_by_timing,
        "bestNid1ByTiming": best_nid1_by_timing,
        "bestNid1ScoreByTiming": best_nid1_score_by_timing,
        "bestNid1FreqByTiming": best_nid1_freq_by_timing,
    }


def plot_result(result: dict, out_png: Path) -> None:
    score_map = result["scoreMap"]
    t_grid = result["timingGrid"]
    expected = int(result["expectedSymbol2Start"])
    best = result["best"]

    x = t_grid - expected
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)

    im = axes[0].imshow(
        score_map,
        aspect="auto",
        origin="lower",
        extent=[int(x[0]), int(x[-1]), 0, 335],
        cmap="magma",
    )
    axes[0].axvline(0, color="cyan", linestyle="--", linewidth=1.0, label="expected symbol2")
    axes[0].axvline(int(best["deltaFromExpected"]), color="lime", linestyle="--", linewidth=1.0, label="best")
    axes[0].axhline(int(best["nId1"]), color="lime", linestyle=":", linewidth=0.9)
    axes[0].set_title("SSS correlation heatmap (from scratch)")
    axes[0].set_xlabel("symbol-2 start delta from expected (samples)")
    axes[0].set_ylabel("N_ID_1")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], label="normalized corr")

    axes[1].plot(x, result["bestByTiming"], color="#183a59", linewidth=1.2)
    axes[1].axvline(0, color="tab:orange", linestyle="--", linewidth=1.0, label="expected")
    axes[1].axvline(int(best["deltaFromExpected"]), color="tab:red", linestyle="--", linewidth=1.0, label="best")
    axes[1].set_title("Best SSS score by timing")
    axes[1].set_xlabel("symbol-2 start delta from expected (samples)")
    axes[1].set_ylabel("max corr over N_ID_1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(x, result["bestNid1ByTiming"], color="#4c78a8", linewidth=0.9, label="best N_ID_1")
    axes[2].set_title("Best N_ID_1 by timing")
    axes[2].set_xlabel("symbol-2 start delta from expected (samples)")
    axes[2].set_ylabel("N_ID_1")
    axes[2].grid(True, alpha=0.3)

    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    signal = np.asarray(np.load(args.input_path), dtype=np.complex64).reshape(-1)
    cp0, cp_other, mu = standard_cp(float(args.sample_rate), int(args.scs), int(args.nfft))

    result = scan_sss(
        signal=signal,
        fs=float(args.sample_rate),
        nfft=int(args.nfft),
        cp0=int(cp0),
        cp_other=int(cp_other),
        timing_offset=int(args.timing_offset),
        ssb_offset=int(args.ssb_subcarrier_offset),
        nid2=int(args.nid2),
        freq_center_hz=float(args.freq_center_hz),
        freq_search_hz=float(args.freq_search_hz),
        freq_step_hz=float(args.freq_step_hz),
        scan_radius=int(args.scan_radius),
        scan_step=int(args.scan_step),
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_png = out_dir / f"{args.output_prefix}.png"
    out_json = out_dir / f"{args.output_prefix}.json"

    plot_result(result, out_png)

    score_map = result.pop("scoreMap")
    best_freq_idx = result.pop("bestFreqIdx")
    best_by_t = result.pop("bestByTiming")
    best_nid1_by_t = result.pop("bestNid1ByTiming")
    best_nid1_score_by_t = result.pop("bestNid1ScoreByTiming")
    best_nid1_freq_by_t = result.pop("bestNid1FreqByTiming")
    timing_grid = result["timingGrid"]
    freq_grid = result["freqGrid"]

    top_k = int(args.top_k)
    flat = score_map.ravel()
    idxs = np.argpartition(flat, -top_k)[-top_k:]
    idxs = idxs[np.argsort(flat[idxs])[::-1]]
    top = []
    seen = set()
    expected = int(result["expectedSymbol2Start"])
    for idx in idxs:
        nid1, col = np.unravel_index(int(idx), score_map.shape)
        sym = int(timing_grid[col])
        key = (nid1, sym)
        if key in seen:
            continue
        seen.add(key)
        fi = int(best_freq_idx[nid1, col])
        top.append(
            {
                "nId1": int(nid1),
                "nIdCell": int(3 * int(nid1) + int(args.nid2)),
                "symbolStart": sym,
                "deltaFromExpected": int(sym - expected),
                "score": float(score_map[nid1, col]),
                "freqHz": float(freq_grid[fi]),
            }
        )
        if len(top) >= top_k:
            break

    out = {
        "inputPath": str(Path(args.input_path).resolve()),
        "sampleRate": float(args.sample_rate),
        "scs": int(args.scs),
        "mu": int(mu),
        "nfft": int(args.nfft),
        "cpFormula": "cpOther=round(144*NFFT/2048); cp0=slotSamples-14*NFFT-13*cpOther for mu>0",
        "cp0": int(cp0),
        "cpOther": int(cp_other),
        "timingOffset": int(args.timing_offset),
        "ssbSubcarrierOffset": int(args.ssb_subcarrier_offset),
        "nId2": int(args.nid2),
        "freqCenterHz": float(args.freq_center_hz),
        "freqSearchHz": float(args.freq_search_hz),
        "freqStepHz": float(args.freq_step_hz),
        "scanRadius": int(args.scan_radius),
        "scanStep": int(args.scan_step),
        "expectedSymbol2Start": int(result["expectedSymbol2Start"]),
        "best": result["best"],
        "expectedBest": result["expectedBest"],
        "secondByTiming": result["secondByTiming"],
        "uniquenessRatio": result["uniquenessRatio"],
        "topCandidates": top,
        "timingGridStart": int(timing_grid[0]),
        "timingGridEnd": int(timing_grid[-1]),
        "freqGrid": [float(f) for f in freq_grid.tolist()],
        "bestByTimingSummary": {
            "max": float(np.max(best_by_t)),
            "mean": float(np.mean(best_by_t)),
            "std": float(np.std(best_by_t)),
        },
        "figure": str(out_png.resolve()),
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "best": out["best"],
        "expectedBest": out["expectedBest"],
        "uniquenessRatio": out["uniquenessRatio"],
        "json": str(out_json.resolve()),
        "png": str(out_png.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
