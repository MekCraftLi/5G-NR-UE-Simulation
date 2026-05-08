import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blind joint ranking from PSS candidates + large-range SSS scan")
    p.add_argument("--signal-path", type=str, default="data/rxSignal.npy")
    p.add_argument("--candidate-json", type=str, default="output/matlab_pss_param_scan_result.json")
    p.add_argument("--score-field", type=str, default="bestRxScore", choices=["bestRxScore", "bestTxScore"])
    p.add_argument("--timing-field", type=str, default="bestRxTiming", choices=["bestRxTiming", "bestTxTiming"])
    p.add_argument("--offset-field", type=str, default="bestRxOffset", choices=["bestRxOffset", "bestTxOffset"])
    p.add_argument("--nid2-field", type=str, default="bestRxNid2", choices=["bestRxNid2", "bestTxNid2"])
    p.add_argument("--max-candidates", type=int, default=12)
    p.add_argument("--min-pss-score", type=float, default=0.05)
    p.add_argument("--scan-radius", type=int, default=6000)
    p.add_argument("--scan-step", type=int, default=2)
    p.add_argument("--freq-search-hz", type=float, default=30000.0)
    p.add_argument("--freq-step-hz", type=float, default=2000.0)
    p.add_argument("--unique-guard", type=int, default=160)
    p.add_argument("--time-penalty-scale", type=float, default=4000.0)
    p.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"])
    p.add_argument("--device", type=str, default="cuda", help="torch device, e.g. cuda or cpu")
    p.add_argument("--output-prefix", type=str, default="blind_pss_sss_joint")
    return p.parse_args()


def standard_cp(fs: float, scs: int, nfft: int) -> tuple[int, int, int]:
    mu_float = np.log2(float(scs) / 15000.0)
    if abs(mu_float - round(mu_float)) > 1e-9:
        raise ValueError(f"Invalid numerology: SCS={scs}")
    mu = int(round(mu_float))
    cp_other = int(round(144 * int(nfft) / 2048))
    slot_samples = int(round(float(fs) * (1e-3 / (2**mu))))
    total_cp = slot_samples - 14 * int(nfft)
    if mu == 0:
        cp0 = int(round((total_cp - 12 * cp_other) / 2.0))
    else:
        cp0 = int(total_cp - 13 * cp_other)
    if cp0 <= 0 or cp_other <= 0:
        raise ValueError(f"Invalid CP values cp0={cp0}, cpOther={cp_other}")
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


def corr_against_bank_torch(
    rx_sss: np.ndarray,
    bank_t: "torch.Tensor",
    bank_norm_t: "torch.Tensor",
    device: str,
) -> np.ndarray:
    rx_t = torch.from_numpy(rx_sss).to(device=device, dtype=torch.complex64)
    rx_norm = torch.linalg.norm(rx_t).clamp_min(1e-12)
    corr = torch.abs(torch.matmul(bank_t, torch.conj(rx_t)))
    out = corr / torch.clamp_min(bank_norm_t * rx_norm, 1e-12)
    return out.detach().cpu().numpy().astype(np.float32)


def scan_sss_for_candidate(
    signal: np.ndarray,
    cand: dict,
    score_field: str,
    timing_field: str,
    offset_field: str,
    nid2_field: str,
    scan_radius: int,
    scan_step: int,
    freq_search_hz: float,
    freq_step_hz: float,
    unique_guard: int,
    time_penalty_scale: float,
    backend: str = "numpy",
    device: str = "cuda",
) -> dict:
    fs = float(cand["fs"])
    scs = int(cand["scs"])
    nfft = int(cand["nfft"])
    pss_score = float(cand[score_field])
    pss_timing = int(cand[timing_field])
    ssb_offset = int(cand[offset_field])
    nid2 = int(cand[nid2_field])

    cp0, cp_other, _ = standard_cp(fs, scs, nfft)
    expected_symbol2 = int(pss_timing) + (nfft + cp0) + (nfft + cp_other)
    t_grid = np.arange(
        expected_symbol2 - int(scan_radius),
        expected_symbol2 + int(scan_radius) + 1,
        int(scan_step),
        dtype=np.int32,
    )
    t_grid = t_grid[(t_grid >= 0) & (t_grid + cp_other + nfft <= len(signal))]
    if len(t_grid) == 0:
        return {"valid": False, "reason": "empty_timing_grid"}

    ssb_bin_start = nfft // 2 - 120 + ssb_offset
    sss_bin_start = ssb_bin_start + 56
    if ssb_bin_start < 0 or ssb_bin_start + 240 > nfft:
        return {"valid": False, "reason": "ssb_bins_out_of_range"}

    if abs(float(freq_search_hz)) <= 0.0:
        f_grid = np.asarray([0.0], dtype=np.float64)
    else:
        step = abs(float(freq_step_hz))
        f_grid = np.arange(-abs(float(freq_search_hz)), abs(float(freq_search_hz)) + step / 2.0, step, dtype=np.float64)
    bank = sss_bank(nid2)
    use_torch = (backend == "torch") and (torch is not None)
    bank_t = None
    bank_norm_t = None
    if use_torch:
        if device.startswith("cuda") and (not torch.cuda.is_available()):
            use_torch = False
        else:
            bank_t = torch.from_numpy(bank).to(device=device, dtype=torch.complex64)
            bank_norm_t = torch.linalg.norm(bank_t, dim=1).clamp_min(1e-12)

    score_map = np.zeros((336, len(t_grid)), dtype=np.float32)
    best_freq_idx = np.zeros((336, len(t_grid)), dtype=np.int16)

    for col, sym_start in enumerate(t_grid):
        useful_start = int(sym_start) + int(cp_other)
        idx = np.arange(useful_start, useful_start + int(nfft), dtype=np.float64)
        best_corr = np.zeros(336, dtype=np.float32)
        best_fi = np.zeros(336, dtype=np.int16)
        for fi, f_hz in enumerate(f_grid):
            phase = np.exp(-1j * 2.0 * np.pi * float(f_hz) * idx / float(fs)).astype(np.complex64)
            spec = np.fft.fftshift(np.fft.fft(signal[useful_start:useful_start + int(nfft)] * phase, int(nfft)))
            rx_sss = spec[sss_bin_start:sss_bin_start + 127].astype(np.complex64)
            if use_torch:
                corr = corr_against_bank_torch(rx_sss, bank_t, bank_norm_t, device=device)
            else:
                corr = corr_against_bank(rx_sss, bank)
            better = corr > best_corr
            best_corr[better] = corr[better]
            best_fi[better] = int(fi)
        score_map[:, col] = best_corr
        best_freq_idx[:, col] = best_fi

    flat = int(np.argmax(score_map))
    best_nid1, best_col = np.unravel_index(flat, score_map.shape)
    best_score = float(score_map[best_nid1, best_col])
    best_timing = int(t_grid[best_col])
    best_freq = float(f_grid[int(best_freq_idx[best_nid1, best_col])])

    expected_col = int(np.argmin(np.abs(t_grid - expected_symbol2)))
    expected_best_nid1 = int(np.argmax(score_map[:, expected_col]))
    expected_best_score = float(score_map[expected_best_nid1, expected_col])
    expected_best_freq = float(f_grid[int(best_freq_idx[expected_best_nid1, expected_col])])

    best_by_t = np.max(score_map, axis=0)
    work = best_by_t.copy()
    left = max(0, best_col - int(unique_guard))
    right = min(len(work), best_col + int(unique_guard) + 1)
    work[left:right] = 0.0
    second_col = int(np.argmax(work))
    second_score = float(work[second_col])
    uniqueness = float(best_score / max(second_score, 1e-12))

    delta = int(best_timing - expected_symbol2)
    time_closeness = float(np.exp(-abs(float(delta)) / max(float(time_penalty_scale), 1.0)))
    blended_sss = 0.6 * expected_best_score + 0.4 * best_score
    joint_score = float(pss_score) * float(blended_sss) * float(uniqueness) * float(time_closeness)

    return {
        "valid": True,
        "fs": fs,
        "scs": scs,
        "nfft": nfft,
        "cp0": int(cp0),
        "cpOther": int(cp_other),
        "timingOffset": int(pss_timing),
        "ssbSubcarrierOffset": int(ssb_offset),
        "nId2": int(nid2),
        "pssScore": float(pss_score),
        "expectedSssSymbolStart": int(expected_symbol2),
        "best": {
            "nId1": int(best_nid1),
            "nIdCell": int(3 * int(best_nid1) + int(nid2)),
            "sssSymbolStart": int(best_timing),
            "deltaFromExpected": int(delta),
            "score": float(best_score),
            "freqHz": float(best_freq),
        },
        "expectedTimingBest": {
            "nId1": int(expected_best_nid1),
            "nIdCell": int(3 * int(expected_best_nid1) + int(nid2)),
            "sssSymbolStart": int(t_grid[expected_col]),
            "deltaFromExpected": int(t_grid[expected_col] - expected_symbol2),
            "score": float(expected_best_score),
            "freqHz": float(expected_best_freq),
        },
        "secondPeakByTiming": {
            "sssSymbolStart": int(t_grid[second_col]),
            "deltaFromExpected": int(t_grid[second_col] - expected_symbol2),
            "score": float(second_score),
        },
        "uniquenessRatio": float(uniqueness),
        "timeCloseness": float(time_closeness),
        "jointBlindScore": float(joint_score),
        "backendUsed": "torch" if use_torch else "numpy",
    }


def main() -> None:
    args = parse_args()
    signal = np.asarray(np.load(args.signal_path), dtype=np.complex64).reshape(-1)
    cand_path = Path(args.candidate_json)
    data = json.loads(cand_path.read_text(encoding="utf-8"))
    rows = data.get("top", [])
    selected = [r for r in rows if float(r.get(str(args.score_field), 0.0)) >= float(args.min_pss_score)]
    selected = selected[: int(args.max_candidates)]

    results = []
    for i, cand in enumerate(selected, 1):
        item = scan_sss_for_candidate(
            signal=signal,
            cand=cand,
            score_field=str(args.score_field),
            timing_field=str(args.timing_field),
            offset_field=str(args.offset_field),
            nid2_field=str(args.nid2_field),
            scan_radius=int(args.scan_radius),
            scan_step=int(args.scan_step),
            freq_search_hz=float(args.freq_search_hz),
            freq_step_hz=float(args.freq_step_hz),
            unique_guard=int(args.unique_guard),
            time_penalty_scale=float(args.time_penalty_scale),
            backend=str(args.backend),
            device=str(args.device),
        )
        item["candidateIndex"] = i
        results.append(item)
        if item.get("valid"):
            print(
                f"[{i}/{len(selected)}] fs={cand['fs']/1e6:.2f}MHz scs={cand['scs']/1e3:.0f}k "
                f"nfft={cand['nfft']} nId2={cand[args.nid2_field]} "
                f"-> joint={item['jointBlindScore']:.6f} sss={item['best']['score']:.4f} "
                f"uniq={item['uniquenessRatio']:.3f} dExp={item['best']['deltaFromExpected']}"
            )
        else:
            print(f"[{i}/{len(selected)}] invalid: {item.get('reason', 'unknown')}")

    valid = [r for r in results if r.get("valid")]
    ranked = sorted(valid, key=lambda x: x["jointBlindScore"], reverse=True)

    out = {
        "signalPath": str(Path(args.signal_path).resolve()),
        "candidateSource": str(cand_path.resolve()),
        "scoreField": str(args.score_field),
        "timingField": str(args.timing_field),
        "offsetField": str(args.offset_field),
        "nid2Field": str(args.nid2_field),
        "maxCandidates": int(args.max_candidates),
        "minPssScore": float(args.min_pss_score),
        "scanRadius": int(args.scan_radius),
        "scanStep": int(args.scan_step),
        "freqSearchHz": float(args.freq_search_hz),
        "freqStepHz": float(args.freq_step_hz),
        "timePenaltyScale": float(args.time_penalty_scale),
        "selectedCandidateCount": len(selected),
        "results": results,
        "rankedByJointBlindScore": ranked,
        "bestOverall": ranked[0] if ranked else None,
    }

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
