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

        # ── 置信度计算 ─────────────────────────────────────────────────────────
        # 置信度指标综合评估:
        #   1. 平均 coherence: CP 与尾部相关性强度 (理想值接近 1.0)
        #   2. coherence 稳定性: 各符号 coherence 的一致性 (标准差小为佳)
        #   3. 残余频偏一致性: 各符号估计值的离散程度 (极差小为佳)
        #   4. 有效符号数: 参与估计的符号数量 (越多越可靠)
        #
        # 置信度公式:
        #   confidence = meanCoh × (1 - stdCoh/meanCoh) × (1 - rangeHz/|residualHz|) × symbolRatio
        #
        # 各分量阈值:
        #   - meanCoh < 0.3: 低相关性，置信度显著下降
        #   - stdCoh/meanCoh > 0.5: coherence 不稳定
        #   - rangeHz > 1000 Hz: 各符号估计不一致
        #   - symbolCount < 2: 样本不足
        meanCoh = float(np.mean(weightArray))
        stdCoh = float(np.std(weightArray))
        rangeHz = float(np.max(residualArray) - np.min(residualArray))
        symbolRatio = float(len(windowIndex)) / float(max(useSymbolCount, 1))

        # coherence 分量: 平均值 × 稳定性因子
        cohStabilityFactor = 1.0 - min(stdCoh / max(meanCoh, 1e-6), 1.0)
        cohScore = meanCoh * cohStabilityFactor

        # 残余频偏一致性分量
        residualAbsHz = max(abs(residualFreqHz), 1.0)
        rangeRatio = min(rangeHz / residualAbsHz, 1.0)
        consistencyScore = 1.0 - rangeRatio

        # 综合置信度
        confidence = float(cohScore * consistencyScore * symbolRatio)

        # 置信度等级判定
        if confidence >= 0.5:
            confidenceLevel = "high"
        elif confidence >= 0.2:
            confidenceLevel = "medium"
        else:
            confidenceLevel = "low"

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
            # ── 置信度指标 ─────────────────────────────────────────────────────
            "confidence": confidence,
            "confidenceLevel": confidenceLevel,
            "meanCoherence": meanCoh,
            "stdCoherence": stdCoh,
            "coherenceStabilityFactor": cohStabilityFactor,
            "residualRangeHz": rangeHz,
            "consistencyScore": consistencyScore,
            "symbolRatio": symbolRatio,
            # ── 原始数据 ───────────────────────────────────────────────────────
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
