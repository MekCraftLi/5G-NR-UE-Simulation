import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pbch.pbchDecoder import PbchDecoder
from pss.pssTemplateFactory import generatePssSequence


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Numerology:
    sampleRate: float
    scs: int
    nfft: int
    cp0: int
    cpOther: int
    cpKind: str


@dataclass(frozen=True)
class PssHit:
    windowStart: int
    windowScoreDb: float
    numerology: Numerology
    ssbSubcarrierOffset: int
    nId2: int
    timingOffset: int
    pssTdScore: float
    pssFdScore: float
    sssFdScore: float
    nId1: int
    nIdCell: int
    freqHz: float
    dmrsScore: float
    iSsbBar: int

    @property
    def jointScore(self) -> float:
        return float(
            self.pssTdScore
            * max(self.pssFdScore, 1e-6)
            * max(self.sssFdScore, 1e-6)
            * max(self.dmrsScore, 1e-6)
        )


def _parseCsvFloats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parseCsvInts(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan RX sync parameters on high-SNR time windows")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rates-mhz", type=str, default="30.72,61.44,122.88,245.76")
    parser.add_argument("--scs-list", type=str, default="15000,30000,60000,120000")
    parser.add_argument("--window-samples", type=int, default=65536)
    parser.add_argument("--hop-samples", type=int, default=32768)
    parser.add_argument("--spectrum-nfft", type=int, default=65536)
    parser.add_argument("--signal-band-mhz", type=float, default=7.2)
    parser.add_argument("--noise-start-mhz", type=float, default=10.0)
    parser.add_argument("--top-windows", type=int, default=6)
    parser.add_argument("--scan-samples", type=int, default=120000)
    parser.add_argument("--offset-min", type=int, default=-72)
    parser.add_argument("--offset-max", type=int, default=72)
    parser.add_argument("--offset-step", type=int, default=4)
    parser.add_argument("--peaks-per-template", type=int, default=3)
    parser.add_argument("--top-coarse-per-window", type=int, default=24)
    parser.add_argument("--validate-top", type=int, default=80)
    parser.add_argument("--freq-search-hz", type=float, default=200000.0)
    parser.add_argument("--freq-step-hz", type=float, default=10000.0)
    parser.add_argument("--output-prefix", type=str, default="rx_high_snr_sync_scan")
    return parser.parse_args()


def _db10(x: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def _db20(x: np.ndarray | float) -> np.ndarray | float:
    return 20.0 * np.log10(np.maximum(x, 1e-30))


def _nextPow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _normalCp(nfft: int) -> int:
    return int(round(144 * int(nfft) / 2048))


def _muFromScs(scs: int) -> int | None:
    mu = np.log2(float(scs) / 15000.0)
    if abs(mu - round(mu)) > 1e-9:
        return None
    return int(round(mu))


def _standardNormalCpPair(sampleRate: float, scs: int, nfft: int) -> tuple[int, int] | None:
    """Return NR normal CP for the first SSB symbol and ordinary symbols.

    TS 38.211 defines normal CP from the numerology. In FFT-sample units the
    ordinary CP is 144*Nfft/2048. The first OFDM symbol of a slot is longer;
    for mu=0 another long symbol occurs at symbol 7, but SSB symbols 0..3 only
    need the symbol-0 CP plus ordinary CPs.
    """
    mu = _muFromScs(scs)
    if mu is None:
        return None
    cpOther = _normalCp(nfft)
    slotDuration = 1e-3 / (2 ** mu)
    slotSamples = int(round(float(sampleRate) * slotDuration))
    totalCp = slotSamples - 14 * int(nfft)
    if totalCp <= 0:
        return None
    if mu == 0:
        # One 15 kHz slot spans the whole subframe; symbols 0 and 7 share the
        # extra CP. Only symbol 0 is inside the four-symbol SS/PBCH block.
        cp0 = int(round((totalCp - 12 * cpOther) / 2.0))
    else:
        cp0 = int(totalCp - 13 * cpOther)
    if cp0 <= 0 or cpOther <= 0:
        return None
    return int(cp0), int(cpOther)


def _buildNumerologies(sampleRatesMHz: list[float], scsList: list[int]) -> list[Numerology]:
    out: list[Numerology] = []
    seen: set[tuple[float, int, int, int, int]] = set()
    for fsMHz in sampleRatesMHz:
        fs = float(fsMHz) * 1e6
        for scs in scsList:
            if _muFromScs(int(scs)) is None:
                continue
            nfft = int(round(fs / float(scs)))
            if nfft < 256 or nfft > 8192:
                continue
            if abs(fs / nfft - float(scs)) > 1e-6:
                continue
            cpPair = _standardNormalCpPair(fs, int(scs), nfft)
            if cpPair is None:
                continue
            cp0, cpOther = cpPair
            key = (fs, int(scs), int(nfft), int(cp0), int(cpOther))
            if key not in seen:
                out.append(Numerology(fs, int(scs), int(nfft), int(cp0), int(cpOther), "normal"))
                seen.add(key)
    return sorted(out, key=lambda x: (x.nfft, x.sampleRate, x.scs, x.cp0, x.cpOther))


def _buildPssTemplate(nfft: int, cp: int, nId2: int, offset: int) -> np.ndarray:
    pssStart = int(nfft) // 2 - 63 + int(offset)
    if pssStart < 0 or pssStart + 127 > int(nfft):
        raise ValueError("PSS bins out of FFT range")
    freq = np.zeros(int(nfft), dtype=np.complex64)
    freq[pssStart:pssStart + 127] = generatePssSequence(int(nId2))
    useful = np.fft.ifft(np.fft.ifftshift(freq)).astype(np.complex64)
    template = np.concatenate([useful[-int(cp):], useful]).astype(np.complex64)
    norm = float(np.linalg.norm(template))
    return (template / max(norm, 1e-12)).astype(np.complex64)


def _windowEnergySqrt(signal: np.ndarray, windowLen: int) -> np.ndarray:
    power = np.abs(signal).astype(np.float64) ** 2
    prefix = np.concatenate([np.asarray([0.0]), np.cumsum(power, dtype=np.float64)])
    return np.sqrt(np.maximum(prefix[windowLen:] - prefix[:-windowLen], 0.0)).astype(np.float32)


def _topPssScores(
    signal: np.ndarray,
    signalFft: np.ndarray,
    corrNfft: int,
    template: np.ndarray,
    energy: np.ndarray,
    topK: int,
    minDistance: int,
) -> list[tuple[int, float]]:
    kernel = np.conjugate(template[::-1]).astype(np.complex64)
    corr = np.fft.ifft(signalFft * np.fft.fft(kernel, n=corrNfft), n=corrNfft)
    validStart = len(template) - 1
    validEnd = validStart + len(signal) - len(template) + 1
    mag = np.abs(corr[validStart:validEnd]).astype(np.float32)
    score = np.zeros_like(mag)
    mask = energy > 1e-12
    score[mask] = mag[mask] / energy[mask]
    work = score.copy()
    hits: list[tuple[int, float]] = []
    radius = max(1, int(minDistance))
    for _ in range(max(1, int(topK))):
        idx = int(np.argmax(work))
        val = float(work[idx])
        if val <= 0.0:
            break
        hits.append((idx, val))
        left = max(0, idx - radius)
        right = min(len(work), idx + radius + 1)
        work[left:right] = 0.0
    return hits


def _buildMSequence(tapA: int, tapB: int) -> np.ndarray:
    seq = np.zeros(127 + 7, dtype=np.int8)
    seq[0] = 1
    for n in range(127):
        seq[n + 7] = (seq[n + tapA] + seq[n + tapB]) & 1
    return seq[:127].copy()


_SSS_X0 = _buildMSequence(4, 0)
_SSS_X1 = _buildMSequence(1, 0)
_SSS_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _sssSequence(nId1: int, nId2: int) -> np.ndarray:
    key = (int(nId1), int(nId2))
    if key in _SSS_CACHE:
        return _SSS_CACHE[key]
    n = np.arange(127, dtype=np.int32)
    m0 = 15 * (int(nId1) // 112) + 5 * int(nId2)
    m1 = int(nId1) % 112
    seq = ((1.0 - 2.0 * _SSS_X0[(n + m0) % 127]) * (1.0 - 2.0 * _SSS_X1[(n + m1) % 127])).astype(np.complex64)
    _SSS_CACHE[key] = seq
    return seq


def _fdCorrScore(rx: np.ndarray, ref: np.ndarray) -> float:
    return float(abs(np.vdot(ref, rx)) / (np.linalg.norm(ref) * np.linalg.norm(rx) + 1e-12))


def _symbolStartAndCp(num: Numerology, symbol: int) -> tuple[int, int]:
    start = 0
    for l in range(symbol):
        start += num.nfft + (num.cp0 if l == 0 else num.cpOther)
    return start, num.cp0 if symbol == 0 else num.cpOther


def _validatePssSss(
    signal: np.ndarray,
    num: Numerology,
    timing: int,
    offset: int,
    nId2: int,
    freqSearchHz: float,
    freqStepHz: float,
) -> tuple[float, float, int, int, float]:
    ssbStart = num.nfft // 2 - 120 + int(offset)
    pssStart = ssbStart + 56
    sssStart = ssbStart + 56
    if ssbStart < 0 or ssbStart + 240 > num.nfft:
        return 0.0, 0.0, -1, -1, 0.0

    freqGrid = np.arange(-abs(freqSearchHz), abs(freqSearchHz) + abs(freqStepHz) / 2.0, abs(freqStepHz))
    pssRel, pssCp = _symbolStartAndCp(num, 0)
    sssRel, sssCp = _symbolStartAndCp(num, 2)
    pssUseful = int(timing) + pssRel + pssCp
    sssUseful = int(timing) + sssRel + sssCp
    if pssUseful < 0 or pssUseful + num.nfft > len(signal) or sssUseful < 0 or sssUseful + num.nfft > len(signal):
        return 0.0, 0.0, -1, -1, 0.0

    pssIdx = np.arange(pssUseful, pssUseful + num.nfft, dtype=np.float64)
    sssIdx = np.arange(sssUseful, sssUseful + num.nfft, dtype=np.float64)
    pssRef = generatePssSequence(int(nId2))
    best = (0.0, 0.0, -1, -1, 0.0)
    for freqHz in freqGrid:
        pssPhase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * pssIdx / num.sampleRate).astype(np.complex64)
        sssPhase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * sssIdx / num.sampleRate).astype(np.complex64)
        pssSpec = np.fft.fftshift(np.fft.fft(signal[pssUseful:pssUseful + num.nfft] * pssPhase, num.nfft))
        sssSpec = np.fft.fftshift(np.fft.fft(signal[sssUseful:sssUseful + num.nfft] * sssPhase, num.nfft))
        pssScore = _fdCorrScore(pssSpec[pssStart:pssStart + 127], pssRef)
        sssRx = sssSpec[sssStart:sssStart + 127]
        bestSss = 0.0
        bestNid1 = -1
        for nId1 in range(336):
            score = _fdCorrScore(sssRx, _sssSequence(nId1, nId2))
            if score > bestSss:
                bestSss = score
                bestNid1 = int(nId1)
        if pssScore * bestSss > best[0] * best[1]:
            best = (float(pssScore), float(bestSss), bestNid1, int(3 * bestNid1 + nId2), float(freqHz))
    return best


def _extractSsbGrid(signal: np.ndarray, num: Numerology, timing: int, offset: int, freqHz: float) -> np.ndarray | None:
    ssbStart = num.nfft // 2 - 120 + int(offset)
    if ssbStart < 0 or ssbStart + 240 > num.nfft:
        return None
    grid = np.zeros((240, 4), dtype=np.complex64)
    for symbol in range(4):
        rel, cp = _symbolStartAndCp(num, symbol)
        useful = int(timing) + rel + cp
        if useful < 0 or useful + num.nfft > len(signal):
            return None
        idx = np.arange(useful, useful + num.nfft, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * idx / num.sampleRate).astype(np.complex64)
        spec = np.fft.fftshift(np.fft.fft(signal[useful:useful + num.nfft] * phase, num.nfft))
        grid[:, symbol] = spec[ssbStart:ssbStart + 240]
    return grid


def _validateDmrs(signal: np.ndarray, num: Numerology, timing: int, offset: int, nIdCell: int, freqHz: float) -> tuple[float, int]:
    if nIdCell < 0:
        return 0.0, -1
    grid = _extractSsbGrid(signal, num, timing, offset, freqHz)
    if grid is None:
        return 0.0, -1
    items = PbchDecoder._dmrsReList(int(nIdCell))
    rx = np.asarray([grid[item.k, item.l] for item in items], dtype=np.complex64)
    bestScore = 0.0
    bestIssb = -1
    for iSsbBar in range(8):
        ref = PbchDecoder.generatePbchDmrs(int(nIdCell), int(iSsbBar))
        score = _fdCorrScore(rx, ref)
        if score > bestScore:
            bestScore = float(score)
            bestIssb = int(iSsbBar)
    return bestScore, bestIssb


def _scoreWindows(
    x: np.ndarray,
    sampleRatesMHz: list[float],
    windowSamples: int,
    hopSamples: int,
    spectrumNfft: int,
    signalBandMHz: float,
    noiseStartMHz: float,
) -> list[dict]:
    if spectrumNfft < windowSamples:
        spectrumNfft = _nextPow2(windowSamples)
    starts = list(range(0, max(1, len(x) - windowSamples + 1), hopSamples))
    rows: list[dict] = []
    for start in starts:
        segment = x[start:start + windowSamples]
        specPower = np.abs(np.fft.fftshift(np.fft.fft(segment, n=spectrumNfft))) ** 2
        bestScore = -300.0
        bestFs = 0.0
        for fsMHz in sampleRatesMHz:
            fs = float(fsMHz) * 1e6
            freq = np.fft.fftshift(np.fft.fftfreq(spectrumNfft, d=1.0 / fs))
            signalMask = np.abs(freq) <= float(signalBandMHz) * 1e6 / 2.0
            noiseMask = np.abs(freq) >= float(noiseStartMHz) * 1e6
            if not np.any(signalMask) or not np.any(noiseMask):
                continue
            inband = float(np.mean(specPower[signalMask]))
            floor = float(np.median(specPower[noiseMask]))
            scoreDb = float(_db10(max(inband - floor, 1e-30) / max(floor, 1e-30)))
            if scoreDb > bestScore:
                bestScore = scoreDb
                bestFs = float(fs)
        rows.append({"start": int(start), "scoreDb": float(bestScore), "bestSampleRate": bestFs})
    return rows


def _selectTopWindows(rows: list[dict], topK: int, guardSamples: int) -> list[dict]:
    selected: list[dict] = []
    for row in sorted(rows, key=lambda r: r["scoreDb"], reverse=True):
        if all(abs(int(row["start"]) - int(prev["start"])) >= guardSamples for prev in selected):
            selected.append(row)
            if len(selected) >= topK:
                break
    return selected


def _plotWindowScores(rows: list[dict], selected: list[dict], outputPath: Path) -> None:
    starts = np.asarray([r["start"] for r in rows], dtype=np.float64)
    scores = np.asarray([r["scoreDb"] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
    ax.plot(starts, scores, marker="o", linewidth=1.0, markersize=3, label="sliding-window spectral score")
    for row in selected:
        ax.axvline(int(row["start"]), color="tab:red", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(int(row["start"]), row["scoreDb"], f"{row['scoreDb']:.1f} dB", rotation=90, va="bottom", fontsize=8)
    ax.set_xlabel("Window start sample")
    ax.set_ylabel("Estimated in-band SNR score (dB)")
    ax.set_title("High-SNR window selection for blind sync scan")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(outputPath, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _scanWindow(
    fullSignal: np.ndarray,
    windowRow: dict,
    numerologies: list[Numerology],
    offsets: list[int],
    scanSamples: int,
    peaksPerTemplate: int,
    topCoarsePerWindow: int,
    validateTop: int,
    freqSearchHz: float,
    freqStepHz: float,
) -> list[PssHit]:
    windowStart = int(windowRow["start"])
    scanStart = max(0, min(windowStart, len(fullSignal) - scanSamples))
    signal = np.asarray(fullSignal[scanStart:scanStart + scanSamples], dtype=np.complex64)
    coarse: list[tuple[float, Numerology, int, int, int]] = []
    for num in numerologies:
        templateLen = num.nfft + num.cp0
        ssbLen = 4 * num.nfft + num.cp0 + 3 * num.cpOther
        if len(signal) < ssbLen:
            continue
        corrNfft = _nextPow2(len(signal) + templateLen - 1)
        signalFft = np.fft.fft(signal, n=corrNfft).astype(np.complex64)
        energy = _windowEnergySqrt(signal, templateLen)
        for offset in offsets:
            for nId2 in (0, 1, 2):
                try:
                    template = _buildPssTemplate(num.nfft, num.cp0, nId2, offset)
                except ValueError:
                    continue
                hits = _topPssScores(
                    signal=signal,
                    signalFft=signalFft,
                    corrNfft=corrNfft,
                    template=template,
                    energy=energy,
                    topK=peaksPerTemplate,
                    minDistance=max(templateLen // 2, 1),
                )
                for timing, score in hits:
                    coarse.append((float(score), num, int(offset), int(nId2), int(timing)))
        coarse = sorted(coarse, key=lambda x: x[0], reverse=True)[: max(topCoarsePerWindow * 3, topCoarsePerWindow)]

    results: list[PssHit] = []
    for pssTdScore, num, offset, nId2, timing in sorted(coarse, key=lambda x: x[0], reverse=True)[:validateTop]:
        pssFd, sssFd, nId1, nIdCell, freqHz = _validatePssSss(
            signal=signal,
            num=num,
            timing=timing,
            offset=offset,
            nId2=nId2,
            freqSearchHz=freqSearchHz,
            freqStepHz=freqStepHz,
        )
        dmrsScore, iSsbBar = _validateDmrs(signal, num, timing, offset, nIdCell, freqHz)
        results.append(PssHit(
            windowStart=scanStart,
            windowScoreDb=float(windowRow["scoreDb"]),
            numerology=num,
            ssbSubcarrierOffset=int(offset),
            nId2=int(nId2),
            timingOffset=int(scanStart + timing),
            pssTdScore=float(pssTdScore),
            pssFdScore=float(pssFd),
            sssFdScore=float(sssFd),
            nId1=int(nId1),
            nIdCell=int(nIdCell),
            freqHz=float(freqHz),
            dmrsScore=float(dmrsScore),
            iSsbBar=int(iSsbBar),
        ))
    return sorted(results, key=lambda x: x.jointScore, reverse=True)


def _hitToDict(hit: PssHit) -> dict:
    num = hit.numerology
    return {
        "jointScore": hit.jointScore,
        "windowStart": int(hit.windowStart),
        "windowScoreDb": float(hit.windowScoreDb),
        "sampleRateHz": float(num.sampleRate),
        "sampleRateMHz": float(num.sampleRate / 1e6),
        "scs": int(num.scs),
        "nfft": int(num.nfft),
        "cp0": int(num.cp0),
        "cpOther": int(num.cpOther),
        "cpKind": num.cpKind,
        "ssbSubcarrierOffset": int(hit.ssbSubcarrierOffset),
        "ssbBinStart": int(num.nfft // 2 - 120 + hit.ssbSubcarrierOffset),
        "ssbBinsMatlab": [
            int(num.nfft // 2 - 120 + hit.ssbSubcarrierOffset + 1),
            int(num.nfft // 2 + 119 + hit.ssbSubcarrierOffset + 1),
        ],
        "nId2": int(hit.nId2),
        "nId1": int(hit.nId1),
        "nIdCell": int(hit.nIdCell),
        "timingOffset": int(hit.timingOffset),
        "freqHz": float(hit.freqHz),
        "pssTdScore": float(hit.pssTdScore),
        "pssFdScore": float(hit.pssFdScore),
        "sssFdScore": float(hit.sssFdScore),
        "dmrsScore": float(hit.dmrsScore),
        "iSsbBar": int(hit.iSsbBar),
    }


def main() -> None:
    args = _parseArgs()
    inputPath = Path(args.input_path)
    x = np.asarray(np.load(inputPath), dtype=np.complex64).reshape(-1)
    sampleRatesMHz = _parseCsvFloats(args.sample_rates_mhz)
    scsList = _parseCsvInts(args.scs_list)
    numerologies = _buildNumerologies(sampleRatesMHz, scsList)
    if not numerologies:
        raise ValueError("No valid numerologies generated from sample-rate/SCS lists")

    logger.info("Loaded %s samples from %s", len(x), inputPath)
    logger.info("Numerologies: %s", [(n.sampleRate / 1e6, n.scs, n.nfft, n.cpKind, n.cp0, n.cpOther) for n in numerologies])

    rows = _scoreWindows(
        x=x,
        sampleRatesMHz=sampleRatesMHz,
        windowSamples=int(args.window_samples),
        hopSamples=int(args.hop_samples),
        spectrumNfft=int(args.spectrum_nfft),
        signalBandMHz=float(args.signal_band_mhz),
        noiseStartMHz=float(args.noise_start_mhz),
    )
    selected = _selectTopWindows(rows, int(args.top_windows), int(args.window_samples))
    logger.info("Selected windows: %s", selected)

    offsets = list(range(int(args.offset_min), int(args.offset_max) + 1, int(args.offset_step)))
    allHits: list[PssHit] = []
    for idx, row in enumerate(selected, 1):
        logger.info("Scanning high-SNR window %s/%s: start=%s score=%.2f dB", idx, len(selected), row["start"], row["scoreDb"])
        hits = _scanWindow(
            fullSignal=x,
            windowRow=row,
            numerologies=numerologies,
            offsets=offsets,
            scanSamples=int(args.scan_samples),
            peaksPerTemplate=int(args.peaks_per_template),
            topCoarsePerWindow=int(args.top_coarse_per_window),
            validateTop=int(args.validate_top),
            freqSearchHz=float(args.freq_search_hz),
            freqStepHz=float(args.freq_step_hz),
        )
        logger.info("Best for window %s: %s", row["start"], _hitToDict(hits[0]) if hits else None)
        allHits.extend(hits)

    allHits = sorted(allHits, key=lambda x: x.jointScore, reverse=True)
    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{args.output_prefix}_windows.png"
    _plotWindowScores(rows, selected, figPath)
    resultPath = outDir / f"{args.output_prefix}_result.json"
    result = {
        "inputPath": str(inputPath.resolve()),
        "samples": int(len(x)),
        "windowSamples": int(args.window_samples),
        "hopSamples": int(args.hop_samples),
        "spectrumNfft": int(args.spectrum_nfft),
        "sampleRatesMHz": sampleRatesMHz,
        "scsList": scsList,
        "offsetRange": [int(args.offset_min), int(args.offset_max), int(args.offset_step)],
        "numerologies": [
            {
                "sampleRateMHz": n.sampleRate / 1e6,
                "scs": n.scs,
                "nfft": n.nfft,
                "cp0": n.cp0,
                "cpOther": n.cpOther,
                "cpKind": n.cpKind,
            }
            for n in numerologies
        ],
        "windowScores": rows,
        "selectedWindows": selected,
        "topCandidates": [_hitToDict(hit) for hit in allHits[:50]],
        "figure": str(figPath.resolve()),
        "note": "Sample-rate hypotheses are evaluated through fs/SCS-derived NFFT and frequency-compensation phase. A reliable lock should show high PSS/SSS/DMRS scores and a clear joint-score lead.",
    }
    resultPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info("Saved window plot: %s", figPath.resolve())
    logger.info("Saved result: %s", resultPath.resolve())
    if allHits:
        logger.info("Best overall: %s", _hitToDict(allHits[0]))


if __name__ == "__main__":
    main()
