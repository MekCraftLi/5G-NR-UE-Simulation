import logging

import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager

logger = logging.getLogger(__name__)


class SssConstants:
    """TS 38.211 Clause 7.4.2.3 中定义的 SSS 常量。"""

    SequenceLength = 127
    Nid1Count = 336


class SssDetector:
    """检测 SSS 并恢复完整物理层小区 ID。

    规范锚点：
    - TS 38.211 Clause 7.4.2.1:
      `N_ID_cell = 3 * N_ID_1 + N_ID_2`；PSS 给出 `N_ID_2`，
      SSS 给出范围为 0..335 的 `N_ID_1`。
    - TS 38.211 Clause 7.4.2.3.1:
      SSS 是长度为 127 的序列，由两条 m 序列以及 `N_ID_1/N_ID_2`
      决定的循环移位生成。
    - TS 38.211 Table 7.4.3.1-1:
      SSS 映射在 SS/PBCH block 符号 l=2 的子载波 k=56..182。

    数据交接：
    `detectSss(rxSignal, pssResult, freqCompHz)` 消费 PSS 输出
    (`pssResult["nId2"]`、`pssResult["timingOffset"]`) 和单位为 Hz 的
    频偏补偿值。它返回 `nId1`、`nIdCell`、`bestSymbolStart`、
    `verifiedSymbolStart`、`verifiedFreqCompHz` 和相关矩阵。选中的
    SSS 结果会原样传入 PBCH。
    """

    def __init__(self, config: SsbConfig):
        self.config = config
        self.fftSize = int(config.FftSize)
        self.cpManager = CpManager(config)
        self.cpLength = int(self.cpManager.normalCpLength)
        self.symbolLength = int(self.fftSize + self.cpLength)
        self.sampleRate = float(config.SampleRate)
        self.ssbSubcarrierOffset = int(getattr(config, "SsbSubcarrierOffset", 0))
        self.expectedSssOffsetSamples = int(self.cpManager.getSymbolLength(0) + self.cpManager.getSymbolLength(1))
        self._x0 = self._buildMSequence(tapA=4, tapB=0)
        self._x1 = self._buildMSequence(tapA=1, tapB=0)
        self._templateBankByNid2: dict[int, np.ndarray] = {}
        self._sssSeqCache: dict[tuple[int, int], np.ndarray] = {}

    @staticmethod
    def _buildMSequence(tapA: int, tapB: int) -> np.ndarray:
        """构造 SSS 生成中使用的一条长度 127 的 m 序列。"""
        seq = np.zeros(SssConstants.SequenceLength + 7, dtype=np.int8)
        seq[0] = 1
        for n in range(SssConstants.SequenceLength):
            seq[n + 7] = (seq[n + tapA] + seq[n + tapB]) & 1
        return seq[:SssConstants.SequenceLength].copy()

    def _buildSssSequence(self, nId1: int, nId2: int) -> np.ndarray:
        """构造一条 SSS 频域参考序列。

        输入：
        - `nId2`：由 PSS 固定。
        - `nId1`：当前测试的候选值，范围 0..335。

        输出：
        - 127 个复数 BPSK 值，后续映射到 SS/PBCH k=56..182。
        """
        key = (int(nId1), int(nId2))
        if key in self._sssSeqCache:
            return self._sssSeqCache[key]
        if nId1 < 0 or nId1 >= SssConstants.Nid1Count:
            raise ValueError(f"N_ID_1 非法：{nId1}")
        if nId2 not in (0, 1, 2):
            raise ValueError(f"N_ID_2 非法：{nId2}")

        # TS 38.211 7.4.2.3.1
        m0 = 15 * (nId1 // 112) + 5 * nId2
        m1 = nId1 % 112
        n = np.arange(SssConstants.SequenceLength, dtype=np.int32)
        s0 = 1.0 - 2.0 * self._x0[(n + m0) % SssConstants.SequenceLength].astype(np.float32)
        s1 = 1.0 - 2.0 * self._x1[(n + m1) % SssConstants.SequenceLength].astype(np.float32)
        seq = (s0 * s1).astype(np.complex64)
        self._sssSeqCache[key] = seq
        return seq

    def _getTemplateBank(self, nId2: int) -> np.ndarray:
        """返回某个 `nId2` 下 336 条归一化 SSS 时域模板。

        数据流：
        候选 `nId1` -> SSS 序列 -> 按配置的 `ssbSubcarrierOffset` 映射到
        FFT 网格 -> IFFT -> `[CP | useful symbol]` 模板。
        得到的模板库形状为 `(336, symbolLength)`，会和所有 SSS 滑窗一次
        矩阵相乘。
        """
        if nId2 in self._templateBankByNid2:
            return self._templateBankByNid2[nId2]

        center = self.fftSize // 2
        sssStart = center - 63 + self.ssbSubcarrierOffset
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

    def _frequencyDomainScore(self, rxSignal: np.ndarray, symbolStart: int, nId1: int, nId2: int, freqCompHz: float) -> float:
        """直接在频域 RE 上复核一个 SSS 候选。

        时域滑窗相关可能出现接近并列的候选。这里提取 `symbolStart` 对应的
        useful OFDM 符号，做频偏补偿和 FFT，再按 `ssbSubcarrierOffset`
        切出 k=56..182，与规范 SSS 序列计算归一化相关。
        """
        usefulStart = int(symbolStart) + self.cpLength
        usefulEnd = usefulStart + self.fftSize
        if usefulStart < 0 or usefulEnd > len(rxSignal):
            return 0.0
        idx = np.arange(usefulStart, usefulEnd, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * idx / self.sampleRate).astype(np.complex64)
        useful = np.asarray(rxSignal[usefulStart:usefulEnd], dtype=np.complex64) * phase
        spectrum = np.fft.fftshift(np.fft.fft(useful)).astype(np.complex64)
        sssStart = self.fftSize // 2 - 63 + self.ssbSubcarrierOffset
        re = spectrum[sssStart:sssStart + SssConstants.SequenceLength]
        ref = self._buildSssSequence(nId1=nId1, nId2=nId2)
        return float(abs(np.vdot(ref, re)) / (np.linalg.norm(ref) * np.linalg.norm(re) + 1e-12))

    def _verifyTopCandidates(
        self,
        rxSignal: np.ndarray,
        nId2: int,
        topCandidates: list[dict],
        freqCompHz: float,
        freqSearchHz: float,
        freqStepHz: float,
    ) -> tuple[dict | None, list[dict]]:
        """对 SSS 滑窗 Top 候选执行残余频偏复核。

        输入候选来自时域 `corrMatrix`。输出的 `verifiedFreqCompHz` 是 PBCH
        应优先使用的频偏交接值，因为它在真实 SSS 资源元素上复核过，比
        更早的 PSS-only 频偏更贴近当前候选。
        """
        if not topCandidates:
            return None, []
        if freqStepHz <= 0 or freqSearchHz <= 0:
            freqGrid = np.asarray([float(freqCompHz)], dtype=np.float64)
        else:
            residual = np.arange(-abs(float(freqSearchHz)), abs(float(freqSearchHz)) + abs(float(freqStepHz)) / 2.0, abs(float(freqStepHz)))
            freqGrid = float(freqCompHz) + residual

        verified: list[dict] = []
        seen = set()
        for candidate in topCandidates[:20]:
            key = (int(candidate["symbolStart"]), int(candidate["nId1"]))
            if key in seen:
                continue
            seen.add(key)
            bestScore = -1.0
            bestFreq = float(freqCompHz)
            for freqHz in freqGrid:
                score = self._frequencyDomainScore(
                    rxSignal=rxSignal,
                    symbolStart=int(candidate["symbolStart"]),
                    nId1=int(candidate["nId1"]),
                    nId2=int(nId2),
                    freqCompHz=float(freqHz),
                )
                if score > bestScore:
                    bestScore = score
                    bestFreq = float(freqHz)
            item = dict(candidate)
            item["fdScore"] = float(bestScore)
            item["verifiedFreqCompHz"] = float(bestFreq)
            item["residualFreqHz"] = float(bestFreq - float(freqCompHz))
            verified.append(item)

        if not verified:
            return None, []
        best = max(verified, key=lambda x: x["fdScore"])
        return best, sorted(verified, key=lambda x: x["fdScore"], reverse=True)

    def detectSss(
        self,
        rxSignal: np.ndarray,
        pssResult: dict,
        freqCompHz: float | None = None,
        startSymbol: float = 0.5,
        endSymbol: float = 2.5,
        stepSamples: int = 1,
        freqSearchHz: float = 1000.0,
        freqStepHz: float = 50.0,
    ) -> dict:
        """围绕 PSS 定时锚点滑窗检测 SSS。

        数据流：
        1. 从 `pssResult` 读取 `nId2` 和 `timingOffset`。
        2. 根据 `startSymbol..endSymbol` 构造采样级 `offsetGrid`。
           偏移量相对 PSS CP 起点；理论 SSS 起点约为
           `PSS timing + symbol0 长度 + symbol1 长度`。
        3. 从 `rxSignal` 切出连续 `region`，用 `freqCompHz` 补偿频偏，
           再在每个绝对起点上展开候选窗口。
        4. 计算 `corrMatrix = windows x conj(templateBank.T)`；列是
           `N_ID_1` 假设，行是定时假设。
        5. 用最佳 `N_ID_1` 和 PSS 的 `N_ID_2` 计算 `nIdCell`。
        6. 对 Top 候选做频域复核，同时返回原始字段和复核字段。
        """
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
            raise ValueError("SSS 滑窗搜索区域长度不足")

        if abs(float(freqCompHz)) > 0.0:
            sampleIndex = np.arange(regionStart, regionEnd, dtype=np.float64)
            phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * sampleIndex / self.sampleRate).astype(np.complex64)
            region = (region * phase).astype(np.complex64)

        relStartGrid = (absStartGrid - regionStart).astype(np.int32)
        # `windows` 每一行是一个包含 CP 的候选 SSS OFDM 符号。
        # `templateBank` 每一列对应一个 N_ID_1 假设，N_ID_2 由 PSS 提供。
        # 因此这个矩阵乘法同时搜索定时和 N_ID_1。
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

        topK = min(30, corrMatrix.size)
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

        verifiedBest, verifiedCandidates = self._verifyTopCandidates(
            rxSignal=rxSignal,
            nId2=nId2,
            topCandidates=topCandidates,
            freqCompHz=float(freqCompHz),
            freqSearchHz=float(freqSearchHz),
            freqStepHz=float(freqStepHz),
        )
        if verifiedBest is not None:
            bestNid1 = int(verifiedBest["nId1"])
            bestOffsetSamples = int(verifiedBest["offsetSamples"])
            bestSymbolStart = int(verifiedBest["symbolStart"])
        nIdCell = int(3 * bestNid1 + nId2)

        logger.info(
            "SSS search done: windows=%s, bestNId1=%s, nIdCell=%s, offset=%s, tdScore=%.6f, fdScore=%.6f",
            len(offsetGrid),
            bestNid1,
            nIdCell,
            bestOffsetSamples,
            bestScore,
            float(verifiedBest.get("fdScore", 0.0)) if verifiedBest else 0.0,
        )

        return {
            "method": "sss_sliding_symbol_correlation_with_fd_verification",
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
            "freqVerificationSearchHz": float(freqSearchHz),
            "freqVerificationStepHz": float(freqStepHz),
            "bestOffsetSamples": int(bestOffsetSamples),
            "bestSymbolStart": int(bestSymbolStart),
            "bestScore": float(bestScore),
            "verifiedSymbolStart": int(bestSymbolStart),
            "verifiedFreqCompHz": float(verifiedBest.get("verifiedFreqCompHz", freqCompHz)) if verifiedBest else float(freqCompHz),
            "verifiedResidualFreqHz": float(verifiedBest.get("residualFreqHz", 0.0)) if verifiedBest else 0.0,
            "verifiedFdScore": float(verifiedBest.get("fdScore", 0.0)) if verifiedBest else 0.0,
            "offsetGridSamples": offsetGrid.astype(np.int32),
            "bestNid1ByOffset": bestNid1ByOffset.astype(np.int32),
            "corrMaxByOffset": corrMaxByOffset.astype(np.float32),
            "corrMatrix": corrMatrix.astype(np.float32),
            "topCandidates": topCandidates[:10],
            "verifiedCandidates": verifiedCandidates[:10],
        }
