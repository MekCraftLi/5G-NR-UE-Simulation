import logging

import numpy as np

from common.config import SsbConfig

logger = logging.getLogger(__name__)


class CpFreqOffsetEstimator:
    def __init__(self, config: SsbConfig):
        self.fftSize = int(config.FftSize)
        self.cpLength = int(config.NormalCpLength)
        self.sampleRate = float(config.SampleRate)
        self.ssbSymbolCount = int(config.SsbSymbolCount)
        self.symbolLength = self.fftSize + self.cpLength

    def estimate(
        self,
        rxSignal: np.ndarray,
        timingOffset: int,
        baseFreqHz: float,
        symbolCount: int | None = None,
    ) -> dict | None:
        useSymbolCount = int(symbolCount) if symbolCount is not None else self.ssbSymbolCount
        if useSymbolCount <= 0:
            return None

        windowIndex = []
        phaseRad = []
        coherence = []
        residualHz = []
        pointIndex = []
        pointWindowIndex = []
        pointSampleIndex = []
        pointPhaseRad = []
        pointWeight = []
        pointResidualHz = []

        for symIdx in range(useSymbolCount):
            symbolStart = int(timingOffset) + symIdx * self.symbolLength
            cpStart = symbolStart
            cpEnd = cpStart + self.cpLength
            tailStart = symbolStart + self.fftSize
            tailEnd = tailStart + self.cpLength

            if cpStart < 0 or tailEnd > len(rxSignal):
                break

            cpSegment = np.asarray(rxSignal[cpStart:cpEnd], dtype=np.complex64)
            tailSegment = np.asarray(rxSignal[tailStart:tailEnd], dtype=np.complex64)
            if len(cpSegment) != self.cpLength or len(tailSegment) != self.cpLength:
                continue

            if abs(float(baseFreqHz)) > 0.0:
                cpIndex = np.arange(cpStart, cpEnd, dtype=np.float64)
                tailIndex = np.arange(tailStart, tailEnd, dtype=np.float64)
                cpPhase = np.exp(-1j * 2.0 * np.pi * float(baseFreqHz) * cpIndex / self.sampleRate).astype(np.complex64)
                tailPhase = np.exp(-1j * 2.0 * np.pi * float(baseFreqHz) * tailIndex / self.sampleRate).astype(np.complex64)
                cpSegment = (cpSegment * cpPhase).astype(np.complex64)
                tailSegment = (tailSegment * tailPhase).astype(np.complex64)

            metric = np.vdot(tailSegment, cpSegment)
            metricAbs = float(np.abs(metric))
            energy = float(np.sqrt(np.sum(np.abs(cpSegment) ** 2) * np.sum(np.abs(tailSegment) ** 2)))
            coh = float(metricAbs / (energy + 1e-12))
            phi = float(np.angle(metric))
            deltaFreqHz = float(-phi * self.sampleRate / (2.0 * np.pi * self.fftSize))

            # Point-wise CP phase difference diagnostics.
            pointMetric = cpSegment * np.conjugate(tailSegment)
            pointPhi = np.unwrap(np.angle(pointMetric)).astype(np.float64)
            pointMag = np.abs(pointMetric).astype(np.float64)
            pointDeltaHz = (-pointPhi * self.sampleRate / (2.0 * np.pi * self.fftSize)).astype(np.float64)
            for localIdx in range(self.cpLength):
                pointIndex.append(symIdx * self.cpLength + localIdx)
                pointWindowIndex.append(symIdx)
                pointSampleIndex.append(localIdx)
            pointPhaseRad.extend(pointPhi.tolist())
            pointWeight.extend(pointMag.tolist())
            pointResidualHz.extend(pointDeltaHz.tolist())

            windowIndex.append(symIdx)
            phaseRad.append(phi)
            coherence.append(coh)
            residualHz.append(deltaFreqHz)

        if not residualHz:
            logger.warning("CP-based CFO estimation failed: no valid CP windows")
            return None

        residualArray = np.asarray(residualHz, dtype=np.float64)
        weightArray = np.asarray(coherence, dtype=np.float64)
        weightArray = np.maximum(weightArray, 1e-12)
        residualFreqHz = float(np.sum(weightArray * residualArray) / np.sum(weightArray))
        refinedFreqHz = float(baseFreqHz + residualFreqHz)
        residualStdHz = float(np.sqrt(np.average((residualArray - residualFreqHz) ** 2, weights=weightArray)))

        return {
            "method": "cp_phase",
            "baseFreqHz": float(baseFreqHz),
            "residualFreqHz": residualFreqHz,
            "refinedFreqHz": refinedFreqHz,
            "fftSize": int(self.fftSize),
            "cpLength": int(self.cpLength),
            "symbolCountUsed": int(len(windowIndex)),
            "residualStdHz": residualStdHz,
            "pointCount": int(len(pointIndex)),
            "windowIndex": np.asarray(windowIndex, dtype=np.int32),
            "phaseRad": np.asarray(phaseRad, dtype=np.float32),
            "coherence": np.asarray(coherence, dtype=np.float32),
            "residualFreqByWindowHz": residualArray.astype(np.float32),
            "pointIndex": np.asarray(pointIndex, dtype=np.int32),
            "pointWindowIndex": np.asarray(pointWindowIndex, dtype=np.int16),
            "pointSampleIndex": np.asarray(pointSampleIndex, dtype=np.int16),
            "pointPhaseRad": np.asarray(pointPhaseRad, dtype=np.float32),
            "pointWeight": np.asarray(pointWeight, dtype=np.float32),
            "pointResidualHz": np.asarray(pointResidualHz, dtype=np.float32),
        }
