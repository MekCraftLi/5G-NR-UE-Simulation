import logging

import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager

logger = logging.getLogger(__name__)


class SssConstants:
    SequenceLength = 127
    Nid1Count = 336


class SssDetector:
    def __init__(self, config: SsbConfig):
        self.config = config
        self.fftSize = int(config.FftSize)
        cpManager = CpManager(config)
        self.cpLength = int(cpManager.normalCpLength)
        self.symbolLength = int(self.fftSize + self.cpLength)
        self.sampleRate = float(config.SampleRate)
        self.expectedSssOffsetSamples = int(cpManager.getSymbolLength(0) + cpManager.getSymbolLength(1))

        self._x0 = self._buildMSequence(tapA=4, tapB=0)
        self._x1 = self._buildMSequence(tapA=1, tapB=0)
        self._templateBankByNid2: dict[int, np.ndarray] = {}

    @staticmethod
    def _buildMSequence(tapA: int, tapB: int) -> np.ndarray:
        seq = np.zeros(SssConstants.SequenceLength + 7, dtype=np.int8)
        seq[0] = 1
        for n in range(SssConstants.SequenceLength):
            seq[n + 7] = (seq[n + tapA] + seq[n + tapB]) & 1
        return seq[:SssConstants.SequenceLength].copy()

    def _buildSssSequence(self, nId1: int, nId2: int) -> np.ndarray:
        if nId1 < 0 or nId1 >= SssConstants.Nid1Count:
            raise ValueError(f"Invalid N_ID_1: {nId1}")
        if nId2 not in (0, 1, 2):
            raise ValueError(f"Invalid N_ID_2: {nId2}")

        # TS 38.211, Clause 7.4.2.3.1
        m0 = 15 * (nId1 // 112) + 5 * nId2
        m1 = nId1 % 112
        n = np.arange(SssConstants.SequenceLength, dtype=np.int32)
        s0 = 1.0 - 2.0 * self._x0[(n + m0) % SssConstants.SequenceLength].astype(np.float32)
        s1 = 1.0 - 2.0 * self._x1[(n + m1) % SssConstants.SequenceLength].astype(np.float32)
        return (s0 * s1).astype(np.complex64)

    def _getTemplateBank(self, nId2: int) -> np.ndarray:
        if nId2 in self._templateBankByNid2:
            return self._templateBankByNid2[nId2]

        center = self.fftSize // 2
        sssStart = center - 63
        templateLen = self.symbolLength
        bank = np.zeros((SssConstants.Nid1Count, templateLen), dtype=np.complex64)

        for nId1 in range(SssConstants.Nid1Count):
            seq = self._buildSssSequence(nId1=nId1, nId2=nId2)
            freqDomain = np.zeros(self.fftSize, dtype=np.complex64)
            freqDomain[sssStart:sssStart + SssConstants.SequenceLength] = seq
            timeNoCp = np.fft.ifft(np.fft.ifftshift(freqDomain)).astype(np.complex64)
            withCp = np.concatenate([timeNoCp[-self.cpLength:], timeNoCp]).astype(np.complex64)
            norm = float(np.linalg.norm(withCp))
            if norm > 1e-12:
                bank[nId1, :] = (withCp / norm).astype(np.complex64)

        self._templateBankByNid2[nId2] = bank
        return bank

    def detectSss(
        self,
        rxSignal: np.ndarray,
        pssResult: dict,
        freqCompHz: float | None = None,
        startSymbol: float = 0.5,
        endSymbol: float = 2.5,
        stepSamples: int = 1,
    ) -> dict:
        nId2 = int(pssResult["nId2"])
        timingOffset = int(pssResult["timingOffset"])
        if freqCompHz is None:
            est = pssResult.get("freqOffsetEstimation")
            if isinstance(est, dict) and "refinedFreqHz" in est:
                freqCompHz = float(est["refinedFreqHz"])
            else:
                freqCompHz = float(pssResult.get("freqOffsetParabolic", pssResult["freqOffset"]))
        if stepSamples <= 0:
            raise ValueError("stepSamples must be > 0")
        if endSymbol <= startSymbol:
            raise ValueError("endSymbol must be > startSymbol")

        startOffset = int(np.round(float(startSymbol) * self.symbolLength))
        endOffset = int(np.round(float(endSymbol) * self.symbolLength))
        offsetGrid = np.arange(startOffset, endOffset + 1, int(stepSamples), dtype=np.int32)
        absStartGrid = timingOffset + offsetGrid

        templateLen = self.symbolLength
        validMask = (absStartGrid >= 0) & ((absStartGrid + templateLen) <= len(rxSignal))
        offsetGrid = offsetGrid[validMask]
        absStartGrid = absStartGrid[validMask]
        if len(offsetGrid) == 0:
            raise ValueError("No valid SSS sliding windows in requested range")

        regionStart = int(absStartGrid[0])
        regionEnd = int(absStartGrid[-1] + templateLen)
        region = np.asarray(rxSignal[regionStart:regionEnd], dtype=np.complex64)
        if len(region) < templateLen:
            raise ValueError("SSS region too short for sliding search")

        if abs(float(freqCompHz)) > 0.0:
            sampleIndex = np.arange(regionStart, regionEnd, dtype=np.float64)
            phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * sampleIndex / self.sampleRate).astype(np.complex64)
            region = (region * phase).astype(np.complex64)

        relStartGrid = (absStartGrid - regionStart).astype(np.int32)
        allWindows = np.lib.stride_tricks.sliding_window_view(region, window_shape=templateLen)
        windows = np.asarray(allWindows[relStartGrid], dtype=np.complex64)
        windowNorm = np.linalg.norm(windows, axis=1).astype(np.float32)

        templateBank = self._getTemplateBank(nId2=nId2)
        corrComplex = windows @ np.conjugate(templateBank.T)
        corrMatrix = np.abs(corrComplex).astype(np.float32)
        corrMatrix /= np.maximum(windowNorm[:, np.newaxis], 1e-12)

        bestNid1ByOffset = np.argmax(corrMatrix, axis=1).astype(np.int32)
        rowIndex = np.arange(len(offsetGrid), dtype=np.int32)
        corrMaxByOffset = corrMatrix[rowIndex, bestNid1ByOffset].astype(np.float32)

        globalFlat = int(np.argmax(corrMatrix))
        globalRow, globalCol = np.unravel_index(globalFlat, corrMatrix.shape)
        bestOffsetSamples = int(offsetGrid[globalRow])
        bestSymbolStart = int(absStartGrid[globalRow])
        bestNid1 = int(globalCol)
        bestScore = float(corrMatrix[globalRow, globalCol])
        nIdCell = int(3 * bestNid1 + nId2)

        topK = 10
        flat = corrMatrix.ravel()
        topIdx = np.argpartition(flat, -topK)[-topK:]
        topIdx = topIdx[np.argsort(flat[topIdx])[::-1]]
        topCandidates = []
        for idx in topIdx:
            r, c = np.unravel_index(int(idx), corrMatrix.shape)
            topCandidates.append({
                "rank": int(len(topCandidates) + 1),
                "offsetSamples": int(offsetGrid[r]),
                "symbolStart": int(absStartGrid[r]),
                "nId1": int(c),
                "nIdCell": int(3 * int(c) + nId2),
                "score": float(corrMatrix[r, c]),
            })

        logger.info(
            "SSS sliding search done: "
            f"windows={len(offsetGrid)}, nid1={SssConstants.Nid1Count}, "
            f"bestNId1={bestNid1}, nIdCell={nIdCell}, bestOffset={bestOffsetSamples}, score={bestScore:.6f}"
        )

        return {
            "method": "sss_sliding_symbol_correlation",
            "nId2": int(nId2),
            "nId1": int(bestNid1),
            "nIdCell": int(nIdCell),
            "timingBase": int(timingOffset),
            "expectedSssOffsetSamples": int(self.expectedSssOffsetSamples),
            "symbolLengthSamples": int(self.symbolLength),
            "searchStartSymbol": float(startSymbol),
            "searchEndSymbol": float(endSymbol),
            "searchStepSamples": int(stepSamples),
            "offsetStartSamples": int(startOffset),
            "offsetEndSamples": int(endOffset),
            "freqCompHz": float(freqCompHz),
            "bestOffsetSamples": int(bestOffsetSamples),
            "bestSymbolStart": int(bestSymbolStart),
            "bestScore": float(bestScore),
            "offsetGridSamples": offsetGrid.astype(np.int32),
            "bestNid1ByOffset": bestNid1ByOffset.astype(np.int32),
            "corrMaxByOffset": corrMaxByOffset.astype(np.float32),
            "corrMatrix": corrMatrix.astype(np.float32),
            "topCandidates": topCandidates,
        }
