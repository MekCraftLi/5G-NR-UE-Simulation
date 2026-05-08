import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full PSS->CFO->SSS search pipeline based on DOI:10.1186/s13638-023-02317-5"
    )
    p.add_argument("--input-path", type=str, required=True)
    p.add_argument("--input-fs", type=float, required=True, help="Input sampling rate in Hz")
    p.add_argument("--detector-scs", type=float, default=30000.0, help="Paper uses 30 kHz")
    p.add_argument("--detector-nfft", type=int, default=256, help="Paper uses 256")
    p.add_argument("--ifo-max", type=int, default=1, help="Search G in [-ifo_max, +ifo_max]")
    p.add_argument("--search-start", type=int, default=0)
    p.add_argument("--search-end", type=int, default=-1, help="-1 means full range")
    p.add_argument("--sss-method", type=str, default="all", choices=["all", "brute", "differential", "partial", "coherent"])
    p.add_argument("--partial-blocks", type=int, default=3, help="M in Eq.(23); paper uses M=3")
    p.add_argument("--sss-time-radius", type=int, default=0, help="0 means strict paper-style fixed offset from PSS")
    p.add_argument("--threshold-ratio", type=float, default=1.20, help="Peak-to-2nd ratio for detection flag")
    p.add_argument("--output-prefix", type=str, default="full_sync_paper_s13638")
    return p.parse_args()


def build_mseq(tap_a: int, tap_b: int) -> np.ndarray:
    x = np.zeros(127 + 7, dtype=np.int8)
    x[0] = 1
    for n in range(127):
        x[n + 7] = (x[n + tap_a] + x[n + tap_b]) & 1
    return x[:127].copy()


def pss_sequence(nid2: int) -> np.ndarray:
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


def standard_cp(fs: float, scs: float, nfft: int) -> tuple[int, int]:
    mu_float = np.log2(float(scs) / 15000.0)
    if abs(mu_float - round(mu_float)) > 1e-9:
        raise ValueError(f"Invalid NR numerology for SCS={scs}")
    mu = int(round(mu_float))
    cp_other = int(round(144 * int(nfft) / 2048))
    slot_samples = int(round(float(fs) * (1e-3 / (2 ** mu))))
    total_cp = slot_samples - 14 * int(nfft)
    cp0 = int(total_cp - 13 * cp_other) if mu > 0 else int(round((total_cp - 12 * cp_other) / 2.0))
    if cp0 <= 0 or cp_other <= 0:
        raise ValueError(f"Invalid CP: cp0={cp0}, cpOther={cp_other}")
    return cp0, cp_other


def rational_resample(x: np.ndarray, fs_in: float, fs_out: float) -> tuple[np.ndarray, int, int]:
    if abs(fs_in - fs_out) <= 1e-9:
        return x.astype(np.complex64), 1, 1
    frac = Fraction(fs_out / fs_in).limit_denominator(1024)
    up, down = frac.numerator, frac.denominator
    y = signal.resample_poly(x, up, down, window=("kaiser", 8.6))
    return y.astype(np.complex64), up, down


def make_detector_templates(nfft: int) -> np.ndarray:
    # Paper simplification: SSB centered in analyzed bandwidth.
    # Detector uses 256 bins (first pow2 > 240), so 8-guard on each side.
    ssb_start = (nfft - 240) // 2
    pss_start = ssb_start + 56
    templates = np.zeros((3, nfft), dtype=np.complex64)
    for nid2 in range(3):
        d = np.zeros(nfft, dtype=np.complex64)
        d[pss_start:pss_start + 127] = pss_sequence(nid2)
        # d is in fftshift order; convert to natural order before IFFT
        templates[nid2, :] = np.fft.ifft(np.fft.ifftshift(d), nfft).astype(np.complex64)
    return templates


def pss_search_paper(
    y: np.ndarray,
    fs: float,
    scs: float,
    nfft: int,
    ifo_max: int,
    search_start: int,
    search_end: int,
) -> dict:
    templates = make_detector_templates(nfft)
    nmax = len(y) - nfft
    if nmax <= 0:
        raise ValueError("Signal too short for detector NFFT")
    s0 = max(0, int(search_start))
    s1 = nmax if int(search_end) < 0 else min(nmax, int(search_end))
    if s1 <= s0:
        raise ValueError("Invalid search range")

    mu_grid = np.arange(s0, s1, dtype=np.int32)
    g_grid = np.arange(-int(ifo_max), int(ifo_max) + 1, dtype=np.int32)
    score_cube = np.zeros((len(g_grid), 3, len(mu_grid)), dtype=np.float64)

    n = np.arange(len(y), dtype=np.float64)
    for gi, g in enumerate(g_grid):
        # Eq.(14): IFO compensation in time domain
        y_g = y * np.exp(-1j * 2.0 * np.pi * float(scs) * float(g) * n / float(fs)).astype(np.complex64)
        for l in range(3):
            h = np.conjugate(templates[l][::-1])
            # Eq.(11)/(13) sliding matched filter
            corr = signal.fftconvolve(y_g, h, mode="valid")
            score_cube[gi, l, :] = np.abs(corr[s0:s1]) ** 2

    flat = int(np.argmax(score_cube))
    gi_best, l_best, mi_best = np.unravel_index(flat, score_cube.shape)
    mu_best = int(mu_grid[mi_best])
    g_best = int(g_grid[gi_best])
    peak = float(score_cube[gi_best, l_best, mi_best])

    score_flat = score_cube.ravel()
    if len(score_flat) > 1:
        second = float(np.partition(score_flat, -2)[-2])
    else:
        second = 0.0
    ratio = float(peak / max(second, 1e-12))

    # Eq.(15)-(17): FFO estimation from two half-correlations
    y_best = y * np.exp(-1j * 2.0 * np.pi * float(scs) * float(g_best) * n / float(fs)).astype(np.complex64)
    seg = y_best[mu_best:mu_best + nfft]
    d_hat = templates[l_best]
    hlen = nfft // 2
    c0 = np.vdot(d_hat[:hlen], seg[:hlen])
    c1 = np.vdot(d_hat[hlen:], seg[hlen:])
    dtheta = float(np.angle(c1 * np.conjugate(c0)))
    ffo = float(dtheta * fs / (np.pi * nfft))
    cfo_total = float(ffo + g_best * scs)

    return {
        "muGridStart": int(mu_grid[0]),
        "muGridEnd": int(mu_grid[-1]),
        "gGrid": [int(v) for v in g_grid.tolist()],
        "scoreCubeShape": [int(v) for v in score_cube.shape],
        "best": {
            "timing": mu_best,
            "nId2": int(l_best),
            "ifoG": g_best,
            "peak": peak,
            "secondPeak": second,
            "peakRatio": ratio,
            "ffoHz": ffo,
            "cfoHz": cfo_total,
        },
        "scoreBestByTiming": np.max(score_cube, axis=(0, 1)),
        "scoreBestTimingByNid2": np.max(score_cube, axis=0),
    }


def split_blocks(n: int, m: int) -> list[tuple[int, int]]:
    parts = np.array_split(np.arange(n), m)
    return [(int(v[0]), int(v[-1]) + 1) for v in parts if len(v) > 0]


def sss_metrics(
    r_sss: np.ndarray,
    r_pss: np.ndarray,
    d_pss: np.ndarray,
    bank: np.ndarray,
    bank_diff: np.ndarray,
    blocks: list[tuple[int, int]],
) -> dict[str, np.ndarray]:
    bank_norm2 = np.sum(np.abs(bank) ** 2, axis=1).astype(np.float64)
    r_sss_norm2 = float(np.vdot(r_sss, r_sss).real)
    brute = np.abs(bank @ np.conjugate(r_sss)) ** 2 / np.maximum(bank_norm2 * r_sss_norm2, 1e-12)

    rdiff = r_sss[1:] * np.conjugate(r_sss[:-1])
    rdiff_norm2 = float(np.vdot(rdiff, rdiff).real)
    bank_diff_norm2 = np.sum(np.abs(bank_diff) ** 2, axis=1).astype(np.float64)
    differential = np.abs(bank_diff @ np.conjugate(rdiff)) ** 2 / np.maximum(bank_diff_norm2 * rdiff_norm2, 1e-12)

    partial = np.zeros(336, dtype=np.float64)
    for b0, b1 in blocks:
        chunk = r_sss[b0:b1]
        c_norm2 = float(np.vdot(chunk, chunk).real)
        b_norm2 = np.sum(np.abs(bank[:, b0:b1]) ** 2, axis=1).astype(np.float64)
        partial += np.abs(bank[:, b0:b1] @ np.conjugate(chunk)) ** 2 / np.maximum(b_norm2 * c_norm2, 1e-12)

    h_pss = r_pss * np.conjugate(d_pss)
    r_coh = r_sss * np.conjugate(h_pss)
    r_coh_norm2 = float(np.vdot(r_coh, r_coh).real)
    coherent = np.abs(bank @ np.conjugate(r_coh)) ** 2 / np.maximum(bank_norm2 * r_coh_norm2, 1e-12)

    return {
        "brute": brute.astype(np.float64),
        "differential": differential.astype(np.float64),
        "partial": partial.astype(np.float64),
        "coherent": coherent.astype(np.float64),
    }


def sss_search_paper(
    y_cfo: np.ndarray,
    fs: float,
    scs: float,
    nfft: int,
    pss_timing: int,
    nid2: int,
    partial_blocks: int,
    sss_time_radius: int,
    threshold_ratio: float,
    chosen_method: str,
) -> dict:
    cp0, cp_other = standard_cp(fs, scs, nfft)
    # paper test setup places SSB at symbols #2~#5 => all cpOther inside SSB;
    # PSS(symbol0) -> SSS(symbol2): +2*(NFFT+cpOther) in useful-sample indexing
    delta_pss_to_sss_useful = 2 * (nfft + cp_other)
    sss_expected = int(pss_timing) + int(delta_pss_to_sss_useful)

    if int(sss_time_radius) <= 0:
        t_grid = np.asarray([sss_expected], dtype=np.int32)
    else:
        t_grid = np.arange(
            sss_expected - int(sss_time_radius),
            sss_expected + int(sss_time_radius) + 1,
            1,
            dtype=np.int32,
        )
    t_grid = t_grid[(t_grid >= 0) & (t_grid + nfft <= len(y_cfo))]
    if len(t_grid) == 0:
        raise ValueError("No valid SSS timing samples for evaluation")

    ssb_start = (nfft - 240) // 2
    pss_start = ssb_start + 56
    sss_start = ssb_start + 56
    d_pss = pss_sequence(nid2)
    bank = sss_bank(nid2)
    bank_diff = bank[:, 1:] * np.conjugate(bank[:, :-1])
    blocks = split_blocks(127, int(partial_blocks))

    methods = ["brute", "differential", "partial", "coherent"] if chosen_method == "all" else [chosen_method]
    score_maps = {m: np.zeros((336, len(t_grid)), dtype=np.float64) for m in methods}

    for col, t_sss in enumerate(t_grid):
        t_pss = int(t_sss) - int(delta_pss_to_sss_useful)
        if t_pss < 0 or t_pss + nfft > len(y_cfo):
            continue
        spec_pss = np.fft.fftshift(np.fft.fft(y_cfo[t_pss:t_pss + nfft], nfft))
        spec_sss = np.fft.fftshift(np.fft.fft(y_cfo[int(t_sss):int(t_sss) + nfft], nfft))
        r_pss = spec_pss[pss_start:pss_start + 127].astype(np.complex64)
        r_sss = spec_sss[sss_start:sss_start + 127].astype(np.complex64)
        met = sss_metrics(r_sss, r_pss, d_pss, bank, bank_diff, blocks)
        for m in methods:
            score_maps[m][:, col] = met[m]

    out = {}
    for m in methods:
        s = score_maps[m]
        flat = int(np.argmax(s))
        nid1, col = np.unravel_index(flat, s.shape)
        peak = float(s[nid1, col])
        vals = s.ravel()
        second = float(np.partition(vals, -2)[-2]) if len(vals) > 1 else 0.0
        ratio = float(peak / max(second, 1e-12))
        out[m] = {
            "best": {
                "nId1": int(nid1),
                "nIdCell": int(3 * int(nid1) + int(nid2)),
                "sssTiming": int(t_grid[col]),
                "deltaFromExpected": int(t_grid[col] - sss_expected),
                "score": peak,
                "peakRatio": ratio,
            },
            "secondPeak": second,
            "detected": bool(ratio >= float(threshold_ratio)),
            "scoreStats": {
                "max": float(np.max(s)),
                "mean": float(np.mean(s)),
                "std": float(np.std(s)),
            },
            "bestByTiming": np.max(s, axis=0),
        }

    return {
        "cp0": int(cp0),
        "cpOther": int(cp_other),
        "deltaPssToSssUseful": int(delta_pss_to_sss_useful),
        "sssExpectedTiming": int(sss_expected),
        "timingGridStart": int(t_grid[0]),
        "timingGridEnd": int(t_grid[-1]),
        "timingGridCount": int(len(t_grid)),
        "methods": out,
    }


def plot_results(pss: dict, sss: dict, out_png: Path) -> None:
    pss_curve = np.asarray(pss["scoreBestByTiming"], dtype=np.float64)
    mu0 = int(pss["muGridStart"])
    x_pss = np.arange(mu0, mu0 + len(pss_curve), dtype=np.int32)

    methods = list(sss["methods"].keys())
    n_rows = 1 + len(methods)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3.2 * n_rows), constrained_layout=True)
    if n_rows == 1:
        axes = [axes]

    axes[0].plot(x_pss, pss_curve, linewidth=1.0, color="#1f4e79")
    axes[0].axvline(pss["best"]["timing"], color="tab:red", linestyle="--", linewidth=1.0)
    axes[0].set_title("PSS search score (max over N_ID2 and IFO)")
    axes[0].set_xlabel("timing sample")
    axes[0].set_ylabel("score")
    axes[0].grid(True, alpha=0.3)

    for i, m in enumerate(methods, start=1):
        b = np.asarray(sss["methods"][m]["bestByTiming"], dtype=np.float64)
        t0 = int(sss["timingGridStart"])
        x = np.arange(t0, t0 + len(b), dtype=np.int32)
        axes[i].plot(x, b, linewidth=1.0, label=m)
        axes[i].axvline(sss["sssExpectedTiming"], color="tab:orange", linestyle="--", linewidth=1.0)
        axes[i].axvline(sss["methods"][m]["best"]["sssTiming"], color="tab:red", linestyle="--", linewidth=1.0)
        axes[i].set_title(f"SSS {m} score (max over N_ID1)")
        axes[i].set_xlabel("SSS useful timing sample")
        axes[i].set_ylabel("score")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="upper right")

    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    x_in = np.asarray(np.load(args.input_path), dtype=np.complex64).reshape(-1)

    fs_det = float(args.detector_scs) * int(args.detector_nfft)  # Eq.(10): Bs = SCS * Nsubcarriers
    y, up, down = rational_resample(x_in, float(args.input_fs), fs_det)

    pss = pss_search_paper(
        y=y,
        fs=fs_det,
        scs=float(args.detector_scs),
        nfft=int(args.detector_nfft),
        ifo_max=int(args.ifo_max),
        search_start=int(args.search_start),
        search_end=int(args.search_end),
    )

    n = np.arange(len(y), dtype=np.float64)
    y_cfo = y * np.exp(-1j * 2.0 * np.pi * float(pss["best"]["cfoHz"]) * n / fs_det).astype(np.complex64)

    sss = sss_search_paper(
        y_cfo=y_cfo,
        fs=fs_det,
        scs=float(args.detector_scs),
        nfft=int(args.detector_nfft),
        pss_timing=int(pss["best"]["timing"]),
        nid2=int(pss["best"]["nId2"]),
        partial_blocks=int(args.partial_blocks),
        sss_time_radius=int(args.sss_time_radius),
        threshold_ratio=float(args.threshold_ratio),
        chosen_method=args.sss_method,
    )

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_json = out_dir / f"{args.output_prefix}.json"
    out_png = out_dir / f"{args.output_prefix}.png"
    plot_results(pss, sss, out_png)

    pss_slim = dict(pss)
    pss_slim["scoreBestByTiming"] = {
        "max": float(np.max(pss["scoreBestByTiming"])),
        "mean": float(np.mean(pss["scoreBestByTiming"])),
        "std": float(np.std(pss["scoreBestByTiming"])),
    }
    pss_slim["scoreBestTimingByNid2"] = {
        "nId2_0_max": float(np.max(pss["scoreBestTimingByNid2"][0])),
        "nId2_1_max": float(np.max(pss["scoreBestTimingByNid2"][1])),
        "nId2_2_max": float(np.max(pss["scoreBestTimingByNid2"][2])),
    }
    for m in sss["methods"]:
        if "bestByTiming" in sss["methods"][m]:
            arr = np.asarray(sss["methods"][m].pop("bestByTiming"), dtype=np.float64)
            sss["methods"][m]["bestByTimingStats"] = {
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
            }

    out = {
        "paperRef": {
            "doi": "10.1186/s13638-023-02317-5",
            "pipeline": [
                "Front-end LPF + downsample to Bs = 30kHz * 256 = 7.68MHz (Eq.10, Section 3.3)",
                "PSS exhaustive cross-correlation timing/NID2 (Eq.11) with IFO grid (Eq.13-14)",
                "FFO estimation from two half-correlations (Eq.15-17)",
                "CFO compensation before SSS",
                "SSS detection with brute/differential/partial/coherent (Eq.18/22/23/24)",
            ],
        },
        "input": {
            "path": str(Path(args.input_path).resolve()),
            "inputFs": float(args.input_fs),
            "detectorFs": fs_det,
            "resampleUp": int(up),
            "resampleDown": int(down),
            "detectorScs": float(args.detector_scs),
            "detectorNfft": int(args.detector_nfft),
            "ifoMax": int(args.ifo_max),
            "searchStart": int(args.search_start),
            "searchEnd": int(args.search_end),
            "sssMethod": args.sss_method,
            "partialBlocks": int(args.partial_blocks),
            "sssTimeRadius": int(args.sss_time_radius),
            "thresholdRatio": float(args.threshold_ratio),
        },
        "pss": pss_slim,
        "sss": sss,
        "figure": str(out_png.resolve()),
    }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "json": str(out_json.resolve()),
                "figure": str(out_png.resolve()),
                "pssBest": out["pss"]["best"],
                "sssBest": {k: out["sss"]["methods"][k]["best"] for k in out["sss"]["methods"].keys()},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
