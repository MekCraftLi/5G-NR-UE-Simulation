import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pss.pssTemplateFactory import generatePssSequence


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot normalized PSS search scores over a timing window")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--cp", type=int, default=352)
    parser.add_argument("--offset-min", type=int, default=-72)
    parser.add_argument("--offset-max", type=int, default=72)
    parser.add_argument("--offset-step", type=int, default=2)
    parser.add_argument("--focus-offset", type=int, default=36)
    parser.add_argument("--focus-nid2", type=int, default=1)
    parser.add_argument("--time-start", type=int, default=1, help="1-based plot start")
    parser.add_argument("--time-end", type=int, default=15000, help="1-based plot end")
    parser.add_argument("--freq-comp-hz", type=float, default=0.0)
    parser.add_argument("--output-prefix", type=str, default="rxSignal_pss_search_1_15000")
    return parser.parse_args()


def _nextPow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _buildPssTemplate(nfft: int, cp: int, nId2: int, offset: int) -> np.ndarray:
    pssStart = int(nfft) // 2 - 63 + int(offset)
    if pssStart < 0 or pssStart + 127 > int(nfft):
        raise ValueError("PSS bins out of FFT range")
    grid = np.zeros(int(nfft), dtype=np.complex64)
    grid[pssStart:pssStart + 127] = generatePssSequence(int(nId2))
    useful = np.fft.ifft(np.fft.ifftshift(grid)).astype(np.complex64)
    template = np.concatenate([useful[-int(cp):], useful]).astype(np.complex64)
    norm = float(np.linalg.norm(template))
    return (template / max(norm, 1e-12)).astype(np.complex64)


def _windowEnergySqrt(signal: np.ndarray, windowLen: int) -> np.ndarray:
    power = np.abs(signal).astype(np.float64) ** 2
    prefix = np.concatenate([np.asarray([0.0]), np.cumsum(power, dtype=np.float64)])
    return np.sqrt(np.maximum(prefix[windowLen:] - prefix[:-windowLen], 0.0)).astype(np.float32)


def _scoreTemplate(signalFft: np.ndarray, corrNfft: int, template: np.ndarray, energy: np.ndarray) -> np.ndarray:
    kernel = np.conjugate(template[::-1]).astype(np.complex64)
    corr = np.fft.ifft(signalFft * np.fft.fft(kernel, n=corrNfft), n=corrNfft)
    validStart = len(template) - 1
    validEnd = validStart + len(energy)
    mag = np.abs(corr[validStart:validEnd]).astype(np.float32)
    score = np.zeros_like(mag, dtype=np.float32)
    mask = energy > 1e-12
    score[mask] = mag[mask] / energy[mask]
    return score


def _topPeaks(score: np.ndarray, x1Based: np.ndarray, count: int, minDistance: int) -> list[dict]:
    work = score.copy()
    peaks: list[dict] = []
    for _ in range(count):
        idx = int(np.argmax(work))
        val = float(work[idx])
        if val <= 0.0:
            break
        peaks.append({"time1Based": int(x1Based[idx]), "timingOffset": int(x1Based[idx] - 1), "score": val})
        left = max(0, idx - int(minDistance))
        right = min(len(work), idx + int(minDistance) + 1)
        work[left:right] = 0.0
    return peaks


def main() -> None:
    args = _parseArgs()
    inputPath = Path(args.input_path)
    x = np.asarray(np.load(inputPath), dtype=np.complex64).reshape(-1)
    nfft = int(args.nfft)
    cp = int(args.cp)
    templateLen = nfft + cp
    timeStart = max(1, int(args.time_start))
    timeEnd = max(timeStart, int(args.time_end))
    maxTiming = timeEnd - 1
    needed = maxTiming + templateLen + 1
    if needed > len(x):
        raise ValueError(f"Signal too short: need {needed}, have {len(x)}")

    signal = x[:needed].astype(np.complex64)
    if abs(float(args.freq_comp_hz)) > 0.0:
        idx = np.arange(len(signal), dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(args.freq_comp_hz) * idx / float(args.sample_rate)).astype(np.complex64)
        signal = (signal * phase).astype(np.complex64)

    corrNfft = _nextPow2(len(signal) + templateLen - 1)
    signalFft = np.fft.fft(signal, n=corrNfft).astype(np.complex64)
    energy = _windowEnergySqrt(signal, templateLen)

    plotSlice = slice(timeStart - 1, timeEnd)
    xAxis = np.arange(timeStart, timeEnd + 1, dtype=np.int32)
    offsets = list(range(int(args.offset_min), int(args.offset_max) + 1, int(args.offset_step)))

    focusScores: dict[int, np.ndarray] = {}
    for nId2 in (0, 1, 2):
        tmpl = _buildPssTemplate(nfft, cp, nId2, int(args.focus_offset))
        focusScores[nId2] = _scoreTemplate(signalFft, corrNfft, tmpl, energy)[plotSlice]

    blindMax = np.zeros(len(xAxis), dtype=np.float32)
    blindBestOffset = np.zeros(len(xAxis), dtype=np.int32)
    blindBestNid2 = np.zeros(len(xAxis), dtype=np.int32)
    heatmap = np.zeros((len(offsets), len(xAxis)), dtype=np.float32)
    allTopRecords: list[dict] = []

    for row, offset in enumerate(offsets):
        offsetMax = np.zeros(len(xAxis), dtype=np.float32)
        for nId2 in (0, 1, 2):
            tmpl = _buildPssTemplate(nfft, cp, nId2, int(offset))
            score = _scoreTemplate(signalFft, corrNfft, tmpl, energy)[plotSlice]
            better = score > blindMax
            blindMax[better] = score[better]
            blindBestOffset[better] = int(offset)
            blindBestNid2[better] = int(nId2)
            offsetMax = np.maximum(offsetMax, score)
            localIdx = int(np.argmax(score))
            allTopRecords.append({
                "offset": int(offset),
                "nId2": int(nId2),
                "time1Based": int(xAxis[localIdx]),
                "timingOffset": int(xAxis[localIdx] - 1),
                "score": float(score[localIdx]),
            })
        heatmap[row, :] = offsetMax

    focusPeaks = {
        str(nId2): _topPeaks(score, xAxis, count=8, minDistance=max(templateLen // 2, 1))
        for nId2, score in focusScores.items()
    }
    blindPeaks = _topPeaks(blindMax, xAxis, count=12, minDistance=max(templateLen // 2, 1))
    for peak in blindPeaks:
        idx = int(peak["time1Based"] - timeStart)
        peak["offset"] = int(blindBestOffset[idx])
        peak["nId2"] = int(blindBestNid2[idx])

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}.png"
    jsonPath = outDir / f"{args.output_prefix}.json"

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), constrained_layout=True)

    colors = {0: "#4c78a8", 1: "#f58518", 2: "#54a24b"}
    for nId2 in (0, 1, 2):
        axes[0].plot(xAxis, focusScores[nId2], linewidth=1.0, color=colors[nId2], label=f"N_ID_2={nId2}")
    axes[0].set_title(f"Fixed SSB offset={args.focus_offset}: normalized PSS correlation")
    axes[0].set_ylabel("NCC score")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(xAxis, blindMax, linewidth=1.0, color="#183a59", label="max over offsets and N_ID_2")
    for peak in blindPeaks[:8]:
        axes[1].axvline(peak["time1Based"], color="#d1495b", linestyle="--", linewidth=0.8, alpha=0.6)
        axes[1].text(
            peak["time1Based"],
            peak["score"],
            f"{peak['score']:.3f}\noff={peak['offset']},n2={peak['nId2']}",
            fontsize=7,
            rotation=90,
            va="bottom",
            ha="center",
        )
    axes[1].set_title("Blind PSS envelope: max over SSB offsets and N_ID_2")
    axes[1].set_ylabel("NCC score")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    im = axes[2].imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[timeStart, timeEnd, offsets[0], offsets[-1]],
        cmap="magma",
    )
    axes[2].set_title("PSS heatmap: max over N_ID_2 for each SSB offset")
    axes[2].set_xlabel("Time index, 1-based sample")
    axes[2].set_ylabel("SSB subcarrier offset")
    fig.colorbar(im, ax=axes[2], label="NCC score")

    fig.suptitle(
        f"{inputPath.name} PSS search | time={timeStart}..{timeEnd} | NFFT={nfft} CP={cp} "
        f"| freqComp={float(args.freq_comp_hz):.1f} Hz",
        fontsize=14,
    )
    fig.savefig(figPath, dpi=170, bbox_inches="tight")
    plt.close(fig)

    bestIdx = int(np.argmax(blindMax))
    result = {
        "inputPath": str(inputPath.resolve()),
        "sampleRate": float(args.sample_rate),
        "nfft": nfft,
        "cp": cp,
        "templateLength": templateLen,
        "timeStart1Based": timeStart,
        "timeEnd1Based": timeEnd,
        "freqCompHz": float(args.freq_comp_hz),
        "focusOffset": int(args.focus_offset),
        "focusNid2": int(args.focus_nid2),
        "bestBlind": {
            "time1Based": int(xAxis[bestIdx]),
            "timingOffset": int(xAxis[bestIdx] - 1),
            "score": float(blindMax[bestIdx]),
            "offset": int(blindBestOffset[bestIdx]),
            "nId2": int(blindBestNid2[bestIdx]),
        },
        "focusPeaks": focusPeaks,
        "blindPeaks": blindPeaks,
        "topTemplateRecords": sorted(allTopRecords, key=lambda r: r["score"], reverse=True)[:30],
        "figure": str(figPath.resolve()),
    }
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    print(json.dumps(result["bestBlind"], ensure_ascii=False, indent=2))
    print(str(figPath.resolve()))
    print(str(jsonPath.resolve()))


if __name__ == "__main__":
    main()
