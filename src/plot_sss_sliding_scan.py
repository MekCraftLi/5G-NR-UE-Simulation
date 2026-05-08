import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SSS sliding timing/N_ID_1 scan with NR standard CP")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--scs", type=int, default=30000)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--timing-offset", type=int, default=0)
    parser.add_argument("--ssb-subcarrier-offset", type=int, default=36)
    parser.add_argument("--nid2", type=int, default=1)
    parser.add_argument("--freq-comp-hz", type=float, default=30000.0)
    parser.add_argument("--scan-radius", type=int, default=512)
    parser.add_argument("--scan-step", type=int, default=1)
    parser.add_argument("--output-prefix", type=str, default="rxSignal_sss_sliding_scan")
    return parser.parse_args()


def _standardNormalCp(sampleRate: float, scs: int, nfft: int) -> tuple[int, int, int]:
    muFloat = np.log2(float(scs) / 15000.0)
    if abs(muFloat - round(muFloat)) > 1e-9:
        raise ValueError(f"SCS={scs} is not a valid NR 15k*2^mu numerology")
    mu = int(round(muFloat))
    cpOther = int(round(144 * int(nfft) / 2048))
    slotSamples = int(round(float(sampleRate) * (1e-3 / (2 ** mu))))
    totalCp = slotSamples - 14 * int(nfft)
    if totalCp <= 0:
        raise ValueError("Invalid fs/SCS/NFFT combination: slot CP budget is non-positive")
    if mu == 0:
        cp0 = int(round((totalCp - 12 * cpOther) / 2.0))
    else:
        cp0 = int(totalCp - 13 * cpOther)
    if cp0 <= 0 or cpOther <= 0:
        raise ValueError(f"Invalid CP result: cp0={cp0}, cpOther={cpOther}")
    return cp0, cpOther, mu


def _buildMSequence(tapA: int, tapB: int) -> np.ndarray:
    seq = np.zeros(127 + 7, dtype=np.int8)
    seq[0] = 1
    for n in range(127):
        seq[n + 7] = (seq[n + tapA] + seq[n + tapB]) & 1
    return seq[:127].copy()


_SSS_X0 = _buildMSequence(4, 0)
_SSS_X1 = _buildMSequence(1, 0)


def _sssBank(nId2: int) -> np.ndarray:
    n = np.arange(127, dtype=np.int32)
    bank = np.zeros((336, 127), dtype=np.complex64)
    for nId1 in range(336):
        m0 = 15 * (nId1 // 112) + 5 * int(nId2)
        m1 = nId1 % 112
        bank[nId1, :] = (
            (1.0 - 2.0 * _SSS_X0[(n + m0) % 127])
            * (1.0 - 2.0 * _SSS_X1[(n + m1) % 127])
        ).astype(np.complex64)
    return bank


def _corrAgainstBank(rx: np.ndarray, bank: np.ndarray) -> np.ndarray:
    rxNorm = float(np.linalg.norm(rx))
    bankNorm = np.linalg.norm(bank, axis=1)
    corr = np.abs(bank @ np.conjugate(rx))
    return (corr / np.maximum(bankNorm * rxNorm, 1e-12)).astype(np.float32)


def main() -> None:
    args = _parseArgs()
    inputPath = Path(args.input_path)
    x = np.asarray(np.load(inputPath), dtype=np.complex64).reshape(-1)
    cp0, cpOther, mu = _standardNormalCp(float(args.sample_rate), int(args.scs), int(args.nfft))

    # SS/PBCH block symbols: PSS at symbol 0, SSS at symbol 2.
    expectedSssSymbolStart = int(args.timing_offset) + (int(args.nfft) + cp0) + (int(args.nfft) + cpOther)
    timingGrid = np.arange(
        expectedSssSymbolStart - int(args.scan_radius),
        expectedSssSymbolStart + int(args.scan_radius) + 1,
        int(args.scan_step),
        dtype=np.int32,
    )
    timingGrid = timingGrid[(timingGrid >= 0) & (timingGrid + cpOther + int(args.nfft) <= len(x))]
    if len(timingGrid) == 0:
        raise ValueError("No valid SSS timing candidates")

    ssbBinStart = int(args.nfft) // 2 - 120 + int(args.ssb_subcarrier_offset)
    sssBinStart = ssbBinStart + 56
    if ssbBinStart < 0 or ssbBinStart + 240 > int(args.nfft):
        raise ValueError("SSB bins are outside FFT range")

    bank = _sssBank(int(args.nid2))
    scoreMap = np.zeros((336, len(timingGrid)), dtype=np.float32)

    for col, symbolStart in enumerate(timingGrid):
        usefulStart = int(symbolStart) + cpOther
        idx = np.arange(usefulStart, usefulStart + int(args.nfft), dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(args.freq_comp_hz) * idx / float(args.sample_rate)).astype(np.complex64)
        spec = np.fft.fftshift(np.fft.fft(x[usefulStart:usefulStart + int(args.nfft)] * phase, int(args.nfft)))
        rxSss = spec[sssBinStart:sssBinStart + 127].astype(np.complex64)
        scoreMap[:, col] = _corrAgainstBank(rxSss, bank)

    flatBest = int(np.argmax(scoreMap))
    bestNid1, bestCol = np.unravel_index(flatBest, scoreMap.shape)
    bestTiming = int(timingGrid[bestCol])
    bestScore = float(scoreMap[bestNid1, bestCol])
    bestNidCell = int(3 * int(bestNid1) + int(args.nid2))

    bestByTiming = np.max(scoreMap, axis=0)
    bestNid1ByTiming = np.argmax(scoreMap, axis=0)
    expectedCol = int(np.argmin(np.abs(timingGrid - expectedSssSymbolStart)))

    topFlat = np.argpartition(scoreMap.ravel(), -30)[-30:]
    topFlat = topFlat[np.argsort(scoreMap.ravel()[topFlat])[::-1]]
    topCandidates = []
    seen = set()
    for flat in topFlat:
        nid1, col = np.unravel_index(int(flat), scoreMap.shape)
        key = (int(nid1), int(timingGrid[col]))
        if key in seen:
            continue
        seen.add(key)
        topCandidates.append({
            "nId1": int(nid1),
            "nIdCell": int(3 * int(nid1) + int(args.nid2)),
            "sssSymbolStart": int(timingGrid[col]),
            "deltaFromExpected": int(timingGrid[col] - expectedSssSymbolStart),
            "score": float(scoreMap[nid1, col]),
        })
        if len(topCandidates) >= 15:
            break

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}.png"
    jsonPath = outDir / f"{args.output_prefix}.json"

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), constrained_layout=True)
    extent = [
        int(timingGrid[0] - expectedSssSymbolStart),
        int(timingGrid[-1] - expectedSssSymbolStart),
        0,
        335,
    ]
    im = axes[0].imshow(scoreMap, aspect="auto", origin="lower", extent=extent, cmap="magma")
    axes[0].axvline(0, color="cyan", linestyle="--", linewidth=1.0, label="expected symbol2 start")
    axes[0].axvline(bestTiming - expectedSssSymbolStart, color="lime", linestyle="--", linewidth=1.0, label="best")
    axes[0].axhline(bestNid1, color="lime", linestyle=":", linewidth=0.9)
    axes[0].set_title("SSS correlation heatmap over timing and N_ID_1")
    axes[0].set_xlabel("SSS symbol start delta from expected (samples)")
    axes[0].set_ylabel("N_ID_1")
    axes[0].legend(loc="upper right")
    fig.colorbar(im, ax=axes[0], label="FD normalized correlation")

    axes[1].plot(timingGrid - expectedSssSymbolStart, bestByTiming, color="#183a59", linewidth=1.2)
    axes[1].axvline(0, color="tab:orange", linestyle="--", linewidth=1.0, label="expected")
    axes[1].axvline(bestTiming - expectedSssSymbolStart, color="tab:red", linestyle="--", linewidth=1.0, label="best")
    axes[1].scatter([bestTiming - expectedSssSymbolStart], [bestScore], color="tab:red", zorder=3)
    axes[1].set_title("Best SSS score at each timing")
    axes[1].set_xlabel("SSS symbol start delta from expected (samples)")
    axes[1].set_ylabel("max corr over N_ID_1")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(timingGrid - expectedSssSymbolStart, bestNid1ByTiming, color="#4c78a8", linewidth=0.9)
    axes[2].axvline(0, color="tab:orange", linestyle="--", linewidth=1.0)
    axes[2].axvline(bestTiming - expectedSssSymbolStart, color="tab:red", linestyle="--", linewidth=1.0)
    axes[2].set_title("N_ID_1 selected by max score at each timing")
    axes[2].set_xlabel("SSS symbol start delta from expected (samples)")
    axes[2].set_ylabel("best N_ID_1")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"{inputPath.name} SSS sliding scan | fs={float(args.sample_rate)/1e6:g}MHz "
        f"SCS={int(args.scs)/1000:g}kHz NFFT={int(args.nfft)} cp0={cp0} cpOther={cpOther} "
        f"offset={int(args.ssb_subcarrier_offset)} N_ID_2={int(args.nid2)} freqComp={float(args.freq_comp_hz):.1f}Hz",
        fontsize=13,
    )
    fig.savefig(figPath, dpi=170, bbox_inches="tight")
    plt.close(fig)

    result = {
        "inputPath": str(inputPath.resolve()),
        "sampleRate": float(args.sample_rate),
        "scs": int(args.scs),
        "mu": int(mu),
        "nfft": int(args.nfft),
        "cpFormula": "cpOther=round(144*NFFT/2048); cp0=slotSamples-14*NFFT-13*cpOther for mu>0",
        "cp0": int(cp0),
        "cpOther": int(cpOther),
        "timingOffset": int(args.timing_offset),
        "expectedSssSymbolStart": int(expectedSssSymbolStart),
        "expectedSssUsefulStart": int(expectedSssSymbolStart + cpOther),
        "ssbSubcarrierOffset": int(args.ssb_subcarrier_offset),
        "ssbBinStart": int(ssbBinStart),
        "sssBinStart": int(sssBinStart),
        "nId2": int(args.nid2),
        "freqCompHz": float(args.freq_comp_hz),
        "scanRadius": int(args.scan_radius),
        "scanStep": int(args.scan_step),
        "best": {
            "nId1": int(bestNid1),
            "nIdCell": int(bestNidCell),
            "sssSymbolStart": int(bestTiming),
            "sssUsefulStart": int(bestTiming + cpOther),
            "deltaFromExpected": int(bestTiming - expectedSssSymbolStart),
            "score": float(bestScore),
        },
        "expectedTimingBest": {
            "nId1": int(bestNid1ByTiming[expectedCol]),
            "nIdCell": int(3 * int(bestNid1ByTiming[expectedCol]) + int(args.nid2)),
            "sssSymbolStart": int(timingGrid[expectedCol]),
            "sssUsefulStart": int(timingGrid[expectedCol] + cpOther),
            "deltaFromExpected": int(timingGrid[expectedCol] - expectedSssSymbolStart),
            "score": float(bestByTiming[expectedCol]),
        },
        "topCandidates": topCandidates,
        "figure": str(figPath.resolve()),
    }
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps({
        "cp0": cp0,
        "cpOther": cpOther,
        "expectedSssSymbolStart": expectedSssSymbolStart,
        "best": result["best"],
        "expectedTimingBest": result["expectedTimingBest"],
        "figure": str(figPath.resolve()),
        "json": str(jsonPath.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
