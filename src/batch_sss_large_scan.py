import argparse
import json
from pathlib import Path

import numpy as np


def _buildMSequence(tapA: int, tapB: int) -> np.ndarray:
    seq = np.zeros(127 + 7, dtype=np.int8)
    seq[0] = 1
    for n in range(127):
        seq[n + 7] = (seq[n + tapA] + seq[n + tapB]) & 1
    return seq[:127].copy()


_SSS_X0 = _buildMSequence(4, 0)
_SSS_X1 = _buildMSequence(1, 0)
_BANK_CACHE: dict[int, np.ndarray] = {}


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch large-range SSS scan for high-score PSS candidates")
    parser.add_argument("--signal-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--score-field", type=str, default="bestRxScore", choices=["bestRxScore", "bestTxScore"])
    parser.add_argument("--timing-field", type=str, default="bestRxTiming", choices=["bestRxTiming", "bestTxTiming"])
    parser.add_argument("--offset-field", type=str, default="bestRxOffset", choices=["bestRxOffset", "bestTxOffset"])
    parser.add_argument("--nid2-field", type=str, default="bestRxNid2", choices=["bestRxNid2", "bestTxNid2"])
    parser.add_argument("--candidate-json", type=str, default="output/matlab_pss_param_scan_result.json")
    parser.add_argument("--min-pss-score", type=float, default=0.17)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--scan-radius", type=int, default=6000)
    parser.add_argument("--scan-step", type=int, default=2)
    parser.add_argument("--freq-comp-mode", type=str, default="zero", choices=["zero", "scs", "custom"])
    parser.add_argument("--custom-freq-hz", type=float, default=0.0)
    parser.add_argument("--unique-guard", type=int, default=160)
    parser.add_argument("--output-prefix", type=str, default="rx_batch_sss_large_scan")
    return parser.parse_args()


def _standardNormalCp(sampleRate: float, scs: int, nfft: int) -> tuple[int, int]:
    muFloat = np.log2(float(scs) / 15000.0)
    if abs(muFloat - round(muFloat)) > 1e-9:
        raise ValueError(f"Invalid SCS for NR numerology: {scs}")
    mu = int(round(muFloat))
    cpOther = int(round(144 * int(nfft) / 2048))
    slotSamples = int(round(float(sampleRate) * (1e-3 / (2**mu))))
    totalCp = slotSamples - 14 * int(nfft)
    if mu == 0:
        cp0 = int(round((totalCp - 12 * cpOther) / 2.0))
    else:
        cp0 = int(totalCp - 13 * cpOther)
    if cp0 <= 0 or cpOther <= 0:
        raise ValueError(f"Invalid CP computed: cp0={cp0}, cpOther={cpOther}")
    return cp0, cpOther


def _sssBank(nId2: int) -> np.ndarray:
    key = int(nId2)
    if key in _BANK_CACHE:
        return _BANK_CACHE[key]
    n = np.arange(127, dtype=np.int32)
    bank = np.zeros((336, 127), dtype=np.complex64)
    for nId1 in range(336):
        m0 = 15 * (nId1 // 112) + 5 * key
        m1 = nId1 % 112
        bank[nId1, :] = (
            (1.0 - 2.0 * _SSS_X0[(n + m0) % 127]) * (1.0 - 2.0 * _SSS_X1[(n + m1) % 127])
        ).astype(np.complex64)
    _BANK_CACHE[key] = bank
    return bank


def _corrAgainstBank(rx: np.ndarray, bank: np.ndarray) -> np.ndarray:
    rxNorm = float(np.linalg.norm(rx))
    bankNorm = np.linalg.norm(bank, axis=1)
    corr = np.abs(bank @ np.conjugate(rx))
    return (corr / np.maximum(bankNorm * rxNorm, 1e-12)).astype(np.float32)


def _scanCandidate(
    signal: np.ndarray,
    cand: dict,
    scoreField: str,
    timingField: str,
    offsetField: str,
    nid2Field: str,
    scanRadius: int,
    scanStep: int,
    freqCompHz: float,
    uniqueGuard: int,
) -> dict:
    fs = float(cand["fs"])
    scs = int(cand["scs"])
    nfft = int(cand["nfft"])
    timingOffset = int(cand[timingField])
    offset = int(cand[offsetField])
    nId2 = int(cand[nid2Field])

    cp0, cpOther = _standardNormalCp(fs, scs, nfft)
    expectedSssSymbolStart = timingOffset + (nfft + cp0) + (nfft + cpOther)
    timingGrid = np.arange(
        expectedSssSymbolStart - int(scanRadius),
        expectedSssSymbolStart + int(scanRadius) + 1,
        int(scanStep),
        dtype=np.int32,
    )
    timingGrid = timingGrid[(timingGrid >= 0) & (timingGrid + cpOther + nfft <= len(signal))]
    if len(timingGrid) == 0:
        return {"valid": False, "reason": "empty_timing_grid"}

    ssbBinStart = nfft // 2 - 120 + offset
    sssBinStart = ssbBinStart + 56
    if ssbBinStart < 0 or ssbBinStart + 240 > nfft:
        return {"valid": False, "reason": "ssb_bin_out_of_range"}

    bank = _sssBank(nId2)
    scoreMap = np.zeros((336, len(timingGrid)), dtype=np.float32)

    for col, symbolStart in enumerate(timingGrid):
        usefulStart = int(symbolStart) + cpOther
        idx = np.arange(usefulStart, usefulStart + nfft, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * idx / fs).astype(np.complex64)
        spec = np.fft.fftshift(np.fft.fft(signal[usefulStart:usefulStart + nfft] * phase, nfft))
        rxSss = spec[sssBinStart:sssBinStart + 127].astype(np.complex64)
        scoreMap[:, col] = _corrAgainstBank(rxSss, bank)

    flatBest = int(np.argmax(scoreMap))
    bestNid1, bestCol = np.unravel_index(flatBest, scoreMap.shape)
    bestScore = float(scoreMap[bestNid1, bestCol])
    bestTiming = int(timingGrid[bestCol])

    expectedCol = int(np.argmin(np.abs(timingGrid - expectedSssSymbolStart)))
    expectedNid1 = int(np.argmax(scoreMap[:, expectedCol]))
    expectedScore = float(scoreMap[expectedNid1, expectedCol])

    bestByTiming = np.max(scoreMap, axis=0)
    work = bestByTiming.copy()
    left = max(0, bestCol - int(uniqueGuard))
    right = min(len(work), bestCol + int(uniqueGuard) + 1)
    work[left:right] = 0.0
    secondCol = int(np.argmax(work))
    secondScore = float(work[secondCol])
    uniqueness = float(bestScore / max(secondScore, 1e-12))

    return {
        "valid": True,
        "fs": fs,
        "scs": scs,
        "nfft": nfft,
        "cp0": int(cp0),
        "cpOther": int(cpOther),
        "timingOffset": int(timingOffset),
        "ssbSubcarrierOffset": int(offset),
        "nId2": int(nId2),
        "pssScore": float(cand[scoreField]),
        "freqCompHz": float(freqCompHz),
        "scanRadius": int(scanRadius),
        "scanStep": int(scanStep),
        "expectedSssSymbolStart": int(expectedSssSymbolStart),
        "best": {
            "nId1": int(bestNid1),
            "nIdCell": int(3 * int(bestNid1) + int(nId2)),
            "sssSymbolStart": int(bestTiming),
            "deltaFromExpected": int(bestTiming - expectedSssSymbolStart),
            "score": float(bestScore),
        },
        "expectedTimingBest": {
            "nId1": int(expectedNid1),
            "nIdCell": int(3 * int(expectedNid1) + int(nId2)),
            "sssSymbolStart": int(timingGrid[expectedCol]),
            "deltaFromExpected": int(timingGrid[expectedCol] - expectedSssSymbolStart),
            "score": float(expectedScore),
        },
        "secondPeakByTiming": {
            "sssSymbolStart": int(timingGrid[secondCol]),
            "deltaFromExpected": int(timingGrid[secondCol] - expectedSssSymbolStart),
            "score": float(secondScore),
        },
        "uniquenessRatio": uniqueness,
    }


def _pickFreqComp(mode: str, scs: int, customHz: float) -> float:
    if mode == "zero":
        return 0.0
    if mode == "scs":
        return float(scs)
    return float(customHz)


def main() -> None:
    args = _parseArgs()
    signal = np.asarray(np.load(args.signal_path), dtype=np.complex64).reshape(-1)
    p = Path(args.candidate_json)
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data.get("top", [])
    selected = [r for r in rows if float(r.get(str(args.score_field), 0.0)) >= float(args.min_pss_score)]
    selected = selected[: int(args.max_candidates)]

    results = []
    for i, cand in enumerate(selected, 1):
        freqCompHz = _pickFreqComp(str(args.freq_comp_mode), int(cand["scs"]), float(args.custom_freq_hz))
        item = _scanCandidate(
            signal=signal,
            cand=cand,
            scoreField=str(args.score_field),
            timingField=str(args.timing_field),
            offsetField=str(args.offset_field),
            nid2Field=str(args.nid2_field),
            scanRadius=int(args.scan_radius),
            scanStep=int(args.scan_step),
            freqCompHz=float(freqCompHz),
            uniqueGuard=int(args.unique_guard),
        )
        item["candidateIndex"] = i
        results.append(item)
        print(
            f"[{i}/{len(selected)}] fs={cand['fs']/1e6:.2f}MHz scs={cand['scs']/1e3:.0f}k nfft={cand['nfft']} "
            f"-> valid={item.get('valid')} best={item.get('best',{}).get('score',0):.4f} uniq={item.get('uniquenessRatio',0):.3f}"
        )

    valid = [r for r in results if r.get("valid")]
    valid = sorted(valid, key=lambda x: (x["best"]["score"], x["uniquenessRatio"]), reverse=True)

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    outPath = outDir / f"{args.output_prefix}_result.json"
    out = {
        "signalPath": str(Path(args.signal_path).resolve()),
        "scoreField": str(args.score_field),
        "timingField": str(args.timing_field),
        "offsetField": str(args.offset_field),
        "nid2Field": str(args.nid2_field),
        "candidateSource": str(p.resolve()),
        "minPssScore": float(args.min_pss_score),
        "scanRadius": int(args.scan_radius),
        "scanStep": int(args.scan_step),
        "freqCompMode": str(args.freq_comp_mode),
        "customFreqHz": float(args.custom_freq_hz),
        "selectedCandidateCount": len(selected),
        "results": results,
        "validRanked": valid,
        "bestOverall": valid[0] if valid else None,
    }
    outPath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {outPath.resolve()}")


if __name__ == "__main__":
    main()
