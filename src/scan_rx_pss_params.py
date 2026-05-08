import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pss.pssTemplateFactory import generatePssSequence
from pbch.pbchDecoder import PbchDecoder
from common.gscnRasterTable import getRasterEntries


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScanCandidate:
    scs: int
    nfft: int
    cp: int
    cpKind: str
    ssbSubcarrierOffset: int
    nId2: int
    timingOffset: int
    score: float
    mainPeakScore: float = 0.0
    secondPeakScore: float = 0.0
    peakRatio: float = 0.0
    segmentPeaks: tuple[tuple[int, float], ...] = ()
    validation: dict | None = None


def _parseCsvInts(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind scan RX PSS numerology/FFT/CP/SSB-bin candidates")
    parser.add_argument("--input-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--sample-rate", type=float, default=122.88e6)
    parser.add_argument("--scs-list", type=str, default="15000,30000,60000")
    parser.add_argument("--prb-count", type=int, default=51, help="NRB used for GSCN raster constraint")
    parser.add_argument("--gscn-constrained", action="store_true", help="Filter SCS list by available GSCN raster entries")
    parser.add_argument("--max-samples", type=int, default=200000, help="Scan prefix length; use 0 for full signal")
    parser.add_argument("--segment-starts", type=str, default="0", help="Comma-separated segment starts for repeated-frame consistency")
    parser.add_argument("--consistency-tol", type=int, default=256, help="Allowed timing spread across segments")
    parser.add_argument("--offset-min", type=int, default=-48)
    parser.add_argument("--offset-max", type=int, default=48)
    parser.add_argument("--offset-step", type=int, default=1)
    parser.add_argument("--offset-two-stage", action="store_true", help="Enable two-stage offset scan (coarse then refine)")
    parser.add_argument("--offset-coarse-step", type=int, default=4, help="Coarse offset step when two-stage is enabled")
    parser.add_argument("--offset-refine-radius", type=int, default=8, help="Refine around best coarse offsets +/- this value")
    parser.add_argument("--offset-refine-seed-count", type=int, default=4, help="Number of top coarse offsets used as refine seeds")
    parser.add_argument("--no-offset-two-stage", action="store_true", help="Disable two-stage offset scan")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--refine-top-k", type=int, default=10)
    parser.add_argument("--peaks-per-template", type=int, default=5)
    parser.add_argument("--single-peak-only", action="store_true", help="For each (scs,nfft,cp,offset,nId2), keep only the strongest PSS peak")
    parser.add_argument("--validate-top-k", type=int, default=80)
    parser.add_argument("--validate-freq-search-hz", type=float, default=20000.0)
    parser.add_argument("--validate-freq-step-hz", type=float, default=2000.0)
    parser.add_argument("--freq-search-hz", type=float, default=20000.0)
    parser.add_argument("--freq-step-hz", type=float, default=1000.0)
    parser.add_argument("--timing-refine", type=int, default=3)
    parser.add_argument("--output-prefix", type=str, default="rx_pss_param_scan")
    return parser.parse_args(argv)


def _normalCp(nfft: int) -> int:
    return int(round(144 * int(nfft) / 2048))


def _longCp(sampleRate: float, scs: int, nfft: int, normalCp: int) -> int | None:
    mu = np.log2(float(scs) / 15000.0)
    if abs(mu - round(mu)) > 1e-9:
        return None
    slotDuration = 1e-3 / (2 ** int(round(mu)))
    totalSamples = int(round(float(sampleRate) * slotDuration))
    totalCp = totalSamples - 14 * int(nfft)
    longCp = totalCp - 13 * int(normalCp)
    return int(longCp) if longCp > 0 else None


def _cpCandidates(sampleRate: float, scs: int, nfft: int) -> list[tuple[int, str]]:
    normal = _normalCp(nfft)
    items: list[tuple[int, str]] = [(normal, "normal")]
    longCp = _longCp(sampleRate, scs, nfft, normal)
    if longCp is not None:
        items.append((longCp, "long"))
    # Provided lab waveform uses 2200-sample symbols in DigitalReceiver.
    if int(nfft) == 2048 and 152 not in [x[0] for x in items]:
        items.append((152, "lab152"))
    unique: list[tuple[int, str]] = []
    seen = set()
    for cp, label in items:
        if cp not in seen:
            unique.append((int(cp), label))
            seen.add(cp)
    return unique


def _buildTemplate(nfft: int, cp: int, nId2: int, ssbSubcarrierOffset: int) -> np.ndarray:
    pssStart = int(nfft) // 2 - 63 + int(ssbSubcarrierOffset)
    if pssStart < 0 or pssStart + 127 > int(nfft):
        raise ValueError("PSS start out of FFT range")
    freq = np.zeros(int(nfft), dtype=np.complex64)
    freq[pssStart:pssStart + 127] = generatePssSequence(int(nId2))
    useful = np.fft.ifft(np.fft.ifftshift(freq)).astype(np.complex64)
    template = np.concatenate([useful[-int(cp):], useful]).astype(np.complex64)
    norm = float(np.linalg.norm(template))
    if norm <= 1e-12:
        raise RuntimeError("zero PSS template")
    return (template / norm).astype(np.complex64)


def _nextPow2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _windowEnergySqrt(signal: np.ndarray, windowLen: int) -> np.ndarray:
    power = np.abs(signal).astype(np.float64) ** 2
    prefix = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(power, dtype=np.float64)])
    return np.sqrt(np.maximum(prefix[windowLen:] - prefix[:-windowLen], 0.0)).astype(np.float32)


def _scoreTemplate(signal: np.ndarray, signalFft: np.ndarray, corrNfft: int, template: np.ndarray, energy: np.ndarray) -> tuple[float, int]:
    kernel = np.conjugate(template[::-1]).astype(np.complex64)
    corr = np.fft.ifft(signalFft * np.fft.fft(kernel, n=corrNfft), n=corrNfft)
    validStart = len(template) - 1
    validEnd = validStart + len(signal) - len(template) + 1
    mag = np.abs(corr[validStart:validEnd]).astype(np.float32)
    score = np.zeros_like(mag)
    mask = energy > 1e-12
    score[mask] = mag[mask] / energy[mask]
    idx = int(np.argmax(score))
    return float(score[idx]), idx


def _topScoresTemplate(
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
    peaks: list[tuple[int, float]] = []
    radius = max(1, int(minDistance))
    for _ in range(max(1, int(topK))):
        idx = int(np.argmax(work))
        value = float(work[idx])
        if value <= 0.0:
            break
        peaks.append((idx, value))
        left = max(0, idx - radius)
        right = min(len(work), idx + radius + 1)
        work[left:right] = 0.0
    return peaks


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


def _fdCorrScore(re: np.ndarray, ref: np.ndarray) -> float:
    return float(abs(np.vdot(ref, re)) / (np.linalg.norm(ref) * np.linalg.norm(re) + 1e-12))


def _validateCandidate(
    signal: np.ndarray,
    sampleRate: float,
    cand: ScanCandidate,
    freqSearchHz: float,
    freqStepHz: float,
) -> dict:
    nfft = int(cand.nfft)
    cp = int(cand.cp)
    offset = int(cand.ssbSubcarrierOffset)
    nId2 = int(cand.nId2)
    ssbBinStart = nfft // 2 - 120 + offset
    pssStart = ssbBinStart + 56
    sssStart = ssbBinStart + 56
    freqGrid = np.arange(-abs(freqSearchHz), abs(freqSearchHz) + abs(freqStepHz) / 2.0, abs(freqStepHz), dtype=np.float64)
    best = {
        "valid": False,
        "score": 0.0,
        "pssFdScore": 0.0,
        "sssFdScore": 0.0,
        "nId1": -1,
        "nIdCell": -1,
        "freqHz": 0.0,
    }
    for freqHz in freqGrid:
        pssRel, pssCp = _symbolStartAndCp(cand, 0)
        sssRel, sssCp = _symbolStartAndCp(cand, 2)
        pssUseful = int(cand.timingOffset) + pssRel + pssCp
        sssUseful = int(cand.timingOffset) + sssRel + sssCp
        if pssUseful < 0 or pssUseful + nfft > len(signal) or sssUseful < 0 or sssUseful + nfft > len(signal):
            continue
        pssIdx = np.arange(pssUseful, pssUseful + nfft, dtype=np.float64)
        sssIdx = np.arange(sssUseful, sssUseful + nfft, dtype=np.float64)
        pssPhase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * pssIdx / float(sampleRate)).astype(np.complex64)
        sssPhase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * sssIdx / float(sampleRate)).astype(np.complex64)
        pssSpec = np.fft.fftshift(np.fft.fft(signal[pssUseful:pssUseful + nfft] * pssPhase, nfft)).astype(np.complex64)
        sssSpec = np.fft.fftshift(np.fft.fft(signal[sssUseful:sssUseful + nfft] * sssPhase, nfft)).astype(np.complex64)
        pssRe = pssSpec[pssStart:pssStart + 127]
        sssRe = sssSpec[sssStart:sssStart + 127]
        pssScore = _fdCorrScore(pssRe, generatePssSequence(nId2))
        bestSssScore = 0.0
        bestNid1 = -1
        for nId1 in range(336):
            score = _fdCorrScore(sssRe, _sssSequence(nId1, nId2))
            if score > bestSssScore:
                bestSssScore = score
                bestNid1 = nId1
        combined = float(cand.score) * max(pssScore, 1e-6) * max(bestSssScore, 1e-6)
        if combined > best["score"]:
            best = {
                "valid": True,
                "score": combined,
                "pssFdScore": float(pssScore),
                "sssFdScore": float(bestSssScore),
                "nId1": int(bestNid1),
                "nIdCell": int(3 * bestNid1 + nId2),
                "freqHz": float(freqHz),
            }
    return best


def _symbolStartAndCp(cand: ScanCandidate, symbol: int) -> tuple[int, int]:
    nfft = int(cand.nfft)
    cp0 = int(cand.cp)
    normal = _normalCp(nfft)
    if cand.cpKind == "long":
        start = 0
        for l in range(symbol):
            start += nfft + (cp0 if l == 0 else normal)
        return start, cp0 if symbol == 0 else normal
    start = symbol * (nfft + cp0)
    return start, cp0


def _extractGrid(signal: np.ndarray, cand: ScanCandidate, freqHz: float) -> np.ndarray | None:
    nfft = int(cand.nfft)
    ssbBinStart = nfft // 2 - 120 + int(cand.ssbSubcarrierOffset)
    if ssbBinStart < 0 or ssbBinStart + 240 > nfft:
        return None
    grid = np.zeros((240, 4), dtype=np.complex64)
    for symbol in range(4):
        relStart, cp = _symbolStartAndCp(cand, symbol)
        usefulStart = int(cand.timingOffset) + relStart + cp
        usefulEnd = usefulStart + nfft
        if usefulStart < 0 or usefulEnd > len(signal):
            return None
        idx = np.arange(usefulStart, usefulEnd, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * idx / float(cand.scs * cand.nfft)).astype(np.complex64)
        spectrum = np.fft.fftshift(np.fft.fft(signal[usefulStart:usefulEnd] * phase, nfft)).astype(np.complex64)
        grid[:, symbol] = spectrum[ssbBinStart:ssbBinStart + 240]
    return grid


def _validatePbchDmrs(signal: np.ndarray, cand: ScanCandidate, validation: dict) -> dict:
    if not validation.get("valid"):
        return {"valid": False, "score": 0.0}
    nIdCell = int(validation["nIdCell"])
    grid = _extractGrid(signal, cand, float(validation["freqHz"]))
    if grid is None:
        return {"valid": False, "score": 0.0}
    items = PbchDecoder._dmrsReList(nIdCell)
    rx = np.asarray([grid[item.k, item.l] for item in items], dtype=np.complex64)
    best = {"valid": False, "score": 0.0, "iSsbBar": -1}
    for iSsbBar in range(8):
        ref = PbchDecoder.generatePbchDmrs(nIdCell, iSsbBar)
        score = _fdCorrScore(rx, ref)
        if score > best["score"]:
            best = {
                "valid": True,
                "score": float(score),
                "iSsbBar": int(iSsbBar),
            }
    return best


def _refineCandidate(
    signal: np.ndarray,
    sampleRate: float,
    cand: ScanCandidate,
    freqSearchHz: float,
    freqStepHz: float,
    timingRefine: int,
) -> dict:
    template = _buildTemplate(cand.nfft, cand.cp, cand.nId2, cand.ssbSubcarrierOffset)
    n = np.arange(len(template), dtype=np.float64)
    freqs = np.arange(-abs(freqSearchHz), abs(freqSearchHz) + abs(freqStepHz) / 2.0, abs(freqStepHz), dtype=np.float64)
    best = {
        "freqHz": 0.0,
        "timingOffset": int(cand.timingOffset),
        "score": float(cand.score),
    }
    for timing in range(max(0, cand.timingOffset - timingRefine), min(len(signal) - len(template), cand.timingOffset + timingRefine) + 1):
        window = signal[timing:timing + len(template)]
        winNorm = float(np.linalg.norm(window))
        if winNorm <= 1e-12:
            continue
        for freq in freqs:
            phase = np.exp(-1j * 2.0 * np.pi * float(freq) * (n + timing) / float(sampleRate)).astype(np.complex64)
            score = float(abs(np.vdot(template, window * phase)) / winNorm)
            if score > best["score"]:
                best = {
                    "freqHz": float(freq),
                    "timingOffset": int(timing),
                    "score": score,
                }
    return best


def _asDict(candidate: ScanCandidate) -> dict:
    return {
        "scs": int(candidate.scs),
        "nfft": int(candidate.nfft),
        "cp": int(candidate.cp),
        "cpKind": candidate.cpKind,
        "ssbSubcarrierOffset": int(candidate.ssbSubcarrierOffset),
        "ssbBinStart": int(candidate.nfft // 2 - 120 + candidate.ssbSubcarrierOffset),
        "ssbBinsMatlab": [
            int(candidate.nfft // 2 - 120 + candidate.ssbSubcarrierOffset + 1),
            int(candidate.nfft // 2 + 119 + candidate.ssbSubcarrierOffset + 1),
        ],
        "nId2": int(candidate.nId2),
        "timingOffset": int(candidate.timingOffset),
        "score": float(candidate.score),
        "mainPeakScore": float(candidate.mainPeakScore),
        "secondPeakScore": float(candidate.secondPeakScore),
        "peakRatio": float(candidate.peakRatio),
        "segmentPeaks": [
            {"timingOffset": int(t), "score": float(s)}
            for t, s in candidate.segmentPeaks
        ],
        "validation": candidate.validation,
    }


def _scanOffsetSet(
    segmentPlans: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    scs: int,
    nfft: int,
    cp: int,
    cpKind: str,
    offsets: list[int],
    peaksPerTemplate: int,
    consistencyTol: int,
    templateLen: int,
    singlePeakOnly: bool,
) -> tuple[list[ScanCandidate], int]:
    out: list[ScanCandidate] = []
    count = 0
    for offset in offsets:
        for nId2 in (0, 1, 2):
            template = _buildTemplate(nfft, cp, nId2, offset)
            top2 = _topScoresTemplate(
                signal=segmentPlans[0][0],
                signalFft=segmentPlans[0][1],
                corrNfft=int(len(segmentPlans[0][1])),
                template=template,
                energy=segmentPlans[0][2],
                topK=2,
                minDistance=max(templateLen // 2, 1),
            )
            mainPeakScore = float(top2[0][1]) if len(top2) >= 1 else 0.0
            secondPeakScore = float(top2[1][1]) if len(top2) >= 2 else 0.0
            peakRatio = float(mainPeakScore / max(secondPeakScore, 1e-12)) if mainPeakScore > 0.0 else 0.0
            perSegmentPeaks: list[list[tuple[int, float]]] = []
            if bool(singlePeakOnly):
                for segment, signalFft, energy in segmentPlans:
                    score0, timing0 = _scoreTemplate(
                        signal=segment,
                        signalFft=signalFft,
                        corrNfft=int(len(signalFft)),
                        template=template,
                        energy=energy,
                    )
                    perSegmentPeaks.append([(int(timing0), float(score0))])
            else:
                for segment, signalFft, energy in segmentPlans:
                    perSegmentPeaks.append(
                        _topScoresTemplate(
                            signal=segment,
                            signalFft=signalFft,
                            corrNfft=int(len(signalFft)),
                            template=template,
                            energy=energy,
                            topK=int(peaksPerTemplate),
                            minDistance=max(templateLen // 2, 1),
                        )
                    )
            for timing, score0 in perSegmentPeaks[0]:
                peaks = [(int(timing), float(score0))]
                for otherPeaks in perSegmentPeaks[1:]:
                    nearest = min(otherPeaks, key=lambda x: abs(int(x[0]) - int(timing)))
                    peaks.append((int(nearest[0]), float(nearest[1])))
                timings = [x[0] for x in peaks]
                scores = [x[1] for x in peaks]
                timingSpread = max(timings) - min(timings) if timings else 0
                consistency = 1.0 if timingSpread <= int(consistencyTol) else 0.25
                score = float(min(scores) * consistency)
                out.append(
                    ScanCandidate(
                        scs=int(scs),
                        nfft=int(nfft),
                        cp=int(cp),
                        cpKind=str(cpKind),
                        ssbSubcarrierOffset=int(offset),
                        nId2=int(nId2),
                        timingOffset=int(timing),
                        score=float(score),
                        mainPeakScore=float(mainPeakScore),
                        secondPeakScore=float(secondPeakScore),
                        peakRatio=float(peakRatio),
                        segmentPeaks=tuple(peaks),
                    )
                )
                count += 1
    return out, count


def main(argv: list[str] | None = None) -> None:
    args = _parseArgs(argv)
    useTwoStageOffsets = bool(args.offset_two_stage) and not bool(args.no_offset_two_stage)
    fullSignal = np.asarray(np.load(args.input_path), dtype=np.complex64).reshape(-1)
    segmentStarts = _parseCsvInts(args.segment_starts)
    segmentLen = int(args.max_samples) if int(args.max_samples) > 0 else len(fullSignal)
    segments = []
    validSegmentStarts = []
    for start in segmentStarts:
        end = int(start) + segmentLen
        if start < 0 or end > len(fullSignal):
            logger.warning("Skip segment [%s,%s): out of range for len=%s", start, end, len(fullSignal))
            continue
        segments.append(fullSignal[int(start):end])
        validSegmentStarts.append(int(start))
    if not segments:
        raise ValueError("No valid scan segments")
    logger.info(
        "Loaded %s samples from %s; scanning %s segment(s), length=%s",
        len(fullSignal),
        args.input_path,
        len(segments),
        segmentLen,
    )

    scsList = _parseCsvInts(args.scs_list)
    gscnScsMeta = []
    if bool(args.gscn_constrained):
        filtered = []
        for scs in scsList:
            try:
                entries, meta = getRasterEntries(prbCount=int(args.prb_count), scsHz=int(scs))
                filtered.append(int(scs))
                gscnScsMeta.append(
                    {
                        "scs": int(scs),
                        "valid": True,
                        "entryCount": int(len(entries)),
                        "caseSet": list(meta.get("caseSet", [])),
                        "bandwidthClass": str(meta.get("bandwidthClass", "")),
                    }
                )
            except Exception as ex:
                gscnScsMeta.append(
                    {
                        "scs": int(scs),
                        "valid": False,
                        "reason": str(ex),
                    }
                )
        scsList = filtered
        if not scsList:
            raise ValueError("No valid SCS remained after GSCN raster filtering")
        logger.info("GSCN constrained SCS list: %s", scsList)
    offsets = list(range(int(args.offset_min), int(args.offset_max) + 1, int(args.offset_step)))
    top: list[ScanCandidate] = []
    total = 0

    for scs in scsList:
        nfft = int(round(float(args.sample_rate) / int(scs)))
        if nfft <= 0:
            continue
        for cp, cpKind in _cpCandidates(float(args.sample_rate), int(scs), nfft):
            templateLen = nfft + cp
            if any(len(segment) < templateLen for segment in segments):
                continue
            corrNfft = _nextPow2(segmentLen + templateLen - 1)
            segmentPlans = [
                (
                    segment,
                    np.fft.fft(segment, n=corrNfft).astype(np.complex64),
                    _windowEnergySqrt(segment, templateLen),
                )
                for segment in segments
            ]
            if useTwoStageOffsets:
                coarseStep = max(int(args.offset_step), int(args.offset_coarse_step))
                coarseOffsets = list(range(int(args.offset_min), int(args.offset_max) + 1, coarseStep))
                logger.info(
                    "Scanning SCS=%s, FFT=%s, CP=%s(%s), offsets two-stage: coarse=%s(step=%s)",
                    scs,
                    nfft,
                    cp,
                    cpKind,
                    len(coarseOffsets),
                    coarseStep,
                )
                coarseCandidates, coarseCount = _scanOffsetSet(
                    segmentPlans=segmentPlans,
                    scs=int(scs),
                    nfft=int(nfft),
                    cp=int(cp),
                    cpKind=str(cpKind),
                    offsets=coarseOffsets,
                    peaksPerTemplate=int(args.peaks_per_template),
                    consistencyTol=int(args.consistency_tol),
                    templateLen=int(templateLen),
                    singlePeakOnly=bool(args.single_peak_only),
                )
                total += int(coarseCount)

                coarseRanked = sorted(coarseCandidates, key=lambda x: x.score, reverse=True)
                seedOffsets: list[int] = []
                for c in coarseRanked:
                    off = int(c.ssbSubcarrierOffset)
                    if off not in seedOffsets:
                        seedOffsets.append(off)
                    if len(seedOffsets) >= int(args.offset_refine_seed_count):
                        break

                refineSet = set()
                for seed in seedOffsets:
                    left = max(int(args.offset_min), seed - int(args.offset_refine_radius))
                    right = min(int(args.offset_max), seed + int(args.offset_refine_radius))
                    refineSet.update(range(left, right + 1, int(args.offset_step)))
                refineOffsets = sorted(refineSet.difference(set(coarseOffsets)))
                logger.info(
                    "Refine offsets around seeds=%s: refine=%s",
                    seedOffsets,
                    len(refineOffsets),
                )
                refineCandidates: list[ScanCandidate] = []
                if len(refineOffsets) > 0:
                    refineCandidates, refineCount = _scanOffsetSet(
                        segmentPlans=segmentPlans,
                        scs=int(scs),
                        nfft=int(nfft),
                        cp=int(cp),
                        cpKind=str(cpKind),
                        offsets=refineOffsets,
                        peaksPerTemplate=int(args.peaks_per_template),
                        consistencyTol=int(args.consistency_tol),
                        templateLen=int(templateLen),
                        singlePeakOnly=bool(args.single_peak_only),
                    )
                    total += int(refineCount)
                top.extend(coarseCandidates)
                top.extend(refineCandidates)
            else:
                logger.info("Scanning SCS=%s, FFT=%s, CP=%s(%s), offsets=%s", scs, nfft, cp, cpKind, len(offsets))
                stageCandidates, stageCount = _scanOffsetSet(
                    segmentPlans=segmentPlans,
                    scs=int(scs),
                    nfft=int(nfft),
                    cp=int(cp),
                    cpKind=str(cpKind),
                    offsets=offsets,
                    peaksPerTemplate=int(args.peaks_per_template),
                    consistencyTol=int(args.consistency_tol),
                    templateLen=int(templateLen),
                    singlePeakOnly=bool(args.single_peak_only),
                )
                top.extend(stageCandidates)
                total += int(stageCount)
            top = sorted(top, key=lambda x: x.score, reverse=True)[: max(int(args.top_k) * 5, int(args.top_k))]

    top = sorted(top, key=lambda x: x.score, reverse=True)[: max(int(args.validate_top_k), int(args.top_k))]
    logger.info("Coarse scan complete: %s candidates evaluated", total)

    validated: list[ScanCandidate] = []
    for cand in top[: int(args.validate_top_k)]:
        validation = _validateCandidate(
            signal=segments[0],
            sampleRate=float(args.sample_rate),
            cand=cand,
            freqSearchHz=float(args.validate_freq_search_hz),
            freqStepHz=float(args.validate_freq_step_hz),
        )
        pbchDmrs = _validatePbchDmrs(segments[0], cand, validation)
        validation["pbchDmrs"] = pbchDmrs
        jointScore = (
            float(validation["score"]) * max(float(pbchDmrs.get("score", 0.0)), 1e-6)
            if validation.get("valid")
            else cand.score * 1e-9
        )
        validated.append(
            ScanCandidate(
                scs=cand.scs,
                nfft=cand.nfft,
                cp=cand.cp,
                cpKind=cand.cpKind,
                ssbSubcarrierOffset=cand.ssbSubcarrierOffset,
                nId2=cand.nId2,
                timingOffset=cand.timingOffset,
                score=float(jointScore),
                segmentPeaks=cand.segmentPeaks,
                validation=validation,
            )
        )
    if validated:
        top = sorted(validated, key=lambda x: x.score, reverse=True)[: int(args.top_k)]
    else:
        top = top[: int(args.top_k)]

    refined = []
    for cand in top[: int(args.refine_top_k)]:
        ref = _refineCandidate(
            signal=segments[0],
            sampleRate=float(args.sample_rate),
            cand=cand,
            freqSearchHz=float(args.freq_search_hz),
            freqStepHz=float(args.freq_step_hz),
            timingRefine=int(args.timing_refine),
        )
        item = _asDict(cand)
        item["refined"] = ref
        refined.append(item)

    result = {
        "inputPath": str(Path(args.input_path).resolve()),
        "sampleRate": float(args.sample_rate),
        "scanSamples": int(segmentLen),
        "segmentStarts": validSegmentStarts,
        "scsList": scsList,
        "gscnConstraint": {
            "enabled": bool(args.gscn_constrained),
            "prbCount": int(args.prb_count),
            "scsValidation": gscnScsMeta,
        },
        "offsetRange": [int(args.offset_min), int(args.offset_max), int(args.offset_step)],
        "offsetStrategy": {
            "twoStage": bool(useTwoStageOffsets),
            "coarseStep": int(args.offset_coarse_step),
            "refineRadius": int(args.offset_refine_radius),
            "refineSeedCount": int(args.offset_refine_seed_count),
        },
        "coarseCandidateCount": int(total),
        "topCandidates": [_asDict(x) for x in top],
        "refinedTopCandidates": refined,
    }
    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    outPath = outDir / f"{args.output_prefix}_result.json"
    outPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")

    best = refined[0] if refined else _asDict(top[0])
    logger.info("Best coarse: %s", _asDict(top[0]))
    if refined:
        logger.info("Best refined: %s", best)
    logger.info("Saved scan result: %s", outPath.resolve())


if __name__ == "__main__":
    main()
