import logging

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal as scipySignal

from common.config import SsbConfig

logger = logging.getLogger(__name__)


class PssConstants:
    """
    PSS 物理层常量 —— 依据 3GPP TS 38.211 Clause 7.4.2.2

    字段说明:
        SequenceLength         : m 序列长度，PSS 占用 127 个子载波 (Clause 7.4.2.2.1)
        ShiftRegisterLength    : 生成多项式阶数 7，对应 x^7 + x^4 + 1 (Clause 7.4.2.2.1)
        SectorOffsetMultiplier : 扇区偏移乘数 43，用于循环移位区分 N_ID_2 (Clause 7.4.2.2.1)
        SectorCount            : 每基站扇区数 3，对应 N_ID_2 ∈ {0, 1, 2} (Clause 7.4.2.1)
    """
    SequenceLength = 127
    ShiftRegisterLength = 7
    SectorOffsetMultiplier = 43
    SectorCount = 3


class PssDetector:
    """
    PSS 盲搜检测器 —— 分段互相关 + overlap-save 加速

    处理流程:
        1. 构造本地 PSS 时域模板 (3 个扇区)
        2. 将每个模板预分段并计算频域共轭 (分段相关抗频偏)
        3. 遍历候选 GSCN 频点:
           a. 频偏补偿
           b. 分段 overlap-save 互相关 (NumPy 批量 FFT 加速)
           c. 非相干累加各段相关幅度
        4. 多峰分析 + CP 回波识别
    """

    # 分段相关参数
    NumSegments = 4  # 分段数 (越大抗频偏能力越强，但 SNR 损失越大)

    def __init__(self, config: SsbConfig):
        self.config = config
        self.localPssTime = self._generateLocalPssTemplates()
        self._precomputeCorrelationParams()

    def _generateLocalPssTemplates(self) -> list:
        """
        生成 3 组 PSS 时域模板 —— TS 38.211 Clause 7.4.2.2.1

        步骤:
            1) 生成长度 127 的 m 序列 x(i)
            2) 按 N_ID_2 循环移位，生成频域 PSS 序列 d_PSS(n)
            3) 映射到 OFDM 子载波网格并 IFFT 得到时域模板

        返回:
            list[np.ndarray] — 长度为 3，每个元素为 FftSize 点时域序列
        """
        # =================================================================
        # 步骤 1: 生成 m 序列
        # 递推: x(i+7) = [x(i+4) + x(i)] mod 2
        # 初始: [x(0)..x(6)] = [0, 1, 1, 0, 1, 1, 1]
        # =================================================================
        mSequence = np.zeros(PssConstants.SequenceLength, dtype=int)
        mSequence[0: PssConstants.ShiftRegisterLength] = [0, 1, 1, 0, 1, 1, 1]

        loopCount = PssConstants.SequenceLength - PssConstants.ShiftRegisterLength
        for i in range(loopCount):
            mSequence[i + PssConstants.ShiftRegisterLength] = (mSequence[i + 4] + mSequence[i]) % 2

        templates = []

        # =================================================================
        # 步骤 2 & 3: 对每个 N_ID_2 生成频域 PSS 并 IFFT
        # d_PSS(n) = 1 - 2 × x[(n + 43 × N_ID_2) mod 127]
        # =================================================================
        for nId2 in range(PssConstants.SectorCount):
            freqSeq = np.zeros(PssConstants.SequenceLength, dtype=complex)

            for n in range(PssConstants.SequenceLength):
                m = (n + PssConstants.SectorOffsetMultiplier * nId2) % PssConstants.SequenceLength
                freqSeq[n] = 1 - 2 * mSequence[m]

            # 映射到 OFDM 子载波网格中央
            grid = np.zeros(self.config.FftSize, dtype=complex)
            centerStart = self.config.FftSize // 2 - (PssConstants.SequenceLength // 2)
            grid[centerStart: centerStart + PssConstants.SequenceLength] = freqSeq

            timeSeq = np.fft.ifft(np.fft.ifftshift(grid))
            templates.append(timeSeq)

        return templates

    def _precomputeCorrelationParams(self):
        """
        预计算分段相关的频域模板参数

        将每个 N_ID_2 的 PSS 时域模板分为 K 段，每段零填充至 2×段长，
        预计算 FFT 共轭，供后续 overlap-save 相关使用。

        数据结构:
            segmentFftConj[nId2][segIdx] — 复数数组, 长度 fftBlockSize
        """
        K = self.NumSegments
        self.segmentLength = self.config.FftSize // K
        self.fftBlockSize = 2 * self.segmentLength

        self.segmentFftConj = []
        for template in self.localPssTime:
            segFfts = []
            for k in range(K):
                # 提取第 k 段
                seg = template[k * self.segmentLength: (k + 1) * self.segmentLength]
                # 零填充至 fftBlockSize 用于 overlap-save
                padded = np.zeros(self.fftBlockSize, dtype=complex)
                padded[:self.segmentLength] = seg
                # 预计算 FFT 共轭 (相关 = IFFT(FFT(x) × conj(FFT(h))))
                segFfts.append(np.conj(np.fft.fft(padded)))
            self.segmentFftConj.append(segFfts)

    # =====================================================================
    # PSS 检测主入口
    # =====================================================================

    def detectPss(self, rxSignal: np.ndarray, validGscnList: list) -> dict:
        """
        PSS 频域盲搜 —— 分段 overlap-save 互相关检测

        算法:
            对每个候选 GSCN 频点:
              1. 频偏补偿: r'(t) = r(t) × exp(-j·2π·Δf·t)
              2. 将补偿后信号按分段偏移切分，预计算各段的输入块 FFT
              3. 对每个 N_ID_2:
                 a. 各段分别执行 overlap-save FFT 相关
                 b. 非相干累加: C(τ) = Σ_k |C_k(τ)|  (抗频偏)
              4. 记录全局最大相关峰

        分段相关的抗频偏原理:
            完整 PSS 模板长度 L = 1024 采样点, 在 30.72 MHz 采样率下
            相干时间 T_coh = L / fs ≈ 33 μs。若载波频偏 Δf 较大,
            在相干时间内累积的相位旋转 2πΔf·T_coh 会显著降低相关峰。
            分段后每段长度 L/K = 256, 相干时间缩短为 T_coh/K,
            相位旋转降为原来的 1/K, 有效提升抗频偏能力。

        参数:
            rxSignal      — 接收基带 I/Q 信号 (复数数组)
            validGscnList — 候选频点列表 [(gscn, freqOffset), ...]

        返回:
            dict — 检测参数及多峰分析结果
        """
        outputLength = len(rxSignal) - self.config.FftSize + 1

        bestCorr = 0.0
        bestCorrResult = None
        bestCorrResultsPerNid2 = [None, None, None]  # 最优 GSCN 下 3 个 N_ID_2 的相关曲线
        bestResult = {
            "timingOffset": 0,
            "nId2":         0,
            "freqOffset":   0,
            "gscn":         0,
            "peakValue":    0.0,
            "corrArray":    None,
            "searchMatrix": None,
            "peakAnalysis": None
        }

        searchMatrix = np.zeros((len(validGscnList), outputLength))
        timeVector = np.arange(len(rxSignal)) / self.config.SampleRate

        # =================================================================
        # 主搜索循环
        # =================================================================
        for freqIdx, (gscn, testFreqOffset) in enumerate(validGscnList):
            # 频偏补偿
            rxShifted = rxSignal * np.exp(-1j * 2 * np.pi * testFreqOffset * timeVector)

            # 预计算所有分段的输入块 FFT (3 个 N_ID_2 共享, 节省 2/3 FFT 计算)
            segBlockFfts = self._precomputeInputBlockFfts(rxShifted, outputLength)

            # 对每个 N_ID_2 执行分段相关，收集全部 3 条曲线
            currentCorrResultsPerNid2 = []
            for nId2 in range(PssConstants.SectorCount):
                corrResult = self._segmentedCorrelate(segBlockFfts, nId2, outputLength)
                currentCorrResultsPerNid2.append(corrResult)
                searchMatrix[freqIdx, :] = np.maximum(searchMatrix[freqIdx, :], corrResult)

            # 在全部 3 条曲线中找全局最优
            for nId2 in range(PssConstants.SectorCount):
                corrResult = currentCorrResultsPerNid2[nId2]
                peakValue = np.max(corrResult)
                if peakValue > bestCorr:
                    bestCorr = peakValue
                    bestCorrResult = corrResult
                    bestCorrResultsPerNid2 = list(currentCorrResultsPerNid2)
                    bestResult["timingOffset"] = int(np.argmax(corrResult))
                    bestResult["nId2"] = nId2
                    bestResult["freqOffset"] = testFreqOffset
                    bestResult["gscn"] = gscn
                    bestResult["peakValue"] = float(peakValue)

        # 多峰分析（传入全部 3 条 N_ID_2 曲线，用于标注每个峰的扇区 ID）
        bestResult["corrArray"] = bestCorrResult
        bestResult["searchMatrix"] = searchMatrix
        bestResult["peakAnalysis"] = self._analyzePeaks(bestCorrResult, bestCorrResultsPerNid2)

        return bestResult

    # =====================================================================
    # 分段 overlap-save 相关核心实现
    # =====================================================================

    def _precomputeInputBlockFfts(self, rxSignal: np.ndarray, outputLength: int) -> list:
        """
        预计算所有分段的输入块 FFT

        对每个分段偏移 offset = segIdx × M:
            1. 从 rxSignal[offset:] 截取
            2. 零填充保证所有块完整
            3. 使用 sliding_window_view 创建步长为 M 的重叠块视图
            4. 批量 FFT (np.fft.fft 支持 axis 参数, 单次调用处理所有块)

        参数:
            rxSignal     — 频偏补偿后的接收信号
            outputLength — 相关输出长度

        返回:
            list[np.ndarray] — segmentFfts[segIdx], shape (numBlocks, fftBlockSize)
        """
        M = self.segmentLength
        B = self.fftBlockSize
        numBlocks = (outputLength + M - 1) // M

        segBlockFfts = []
        for segIdx in range(self.NumSegments):
            offset = segIdx * M
            rxSeg = rxSignal[offset:]

            # 零填充确保最后一个块完整
            paddedLen = (numBlocks - 1) * M + B
            rxPadded = np.zeros(paddedLen, dtype=complex)
            copyLen = min(len(rxSeg), paddedLen)
            if copyLen > 0:
                rxPadded[:copyLen] = rxSeg[:copyLen]

            # sliding_window_view 创建步长为 M 的重叠块 (零拷贝视图)
            windows = sliding_window_view(rxPadded, window_shape=B)[::M][:numBlocks]

            # 批量 FFT: 一次调用计算所有块的 FFT
            blockFfts = np.fft.fft(windows, axis=1)
            segBlockFfts.append(blockFfts)

        return segBlockFfts

    def _segmentedCorrelate(self, segBlockFfts: list, nId2: int, outputLength: int) -> np.ndarray:
        """
        分段非相干累加互相关

        对每个分段:
            1. 输入块 FFT × 模板 FFT 共轭 → 频域相关
            2. 批量 IFFT → 时域相关
            3. 提取有效样本 (overlap-save: 前 M 个样本)
            4. 取幅度 |C_k(τ)|

        非相干累加: C(τ) = Σ_{k=0}^{K-1} |C_k(τ)|
            - 各段独立取幅度后累加, 不要求段间相位一致
            - 抗频偏: 每段相干时间缩短为原来的 1/K
            - 代价: 相干增益损失约 10×log10(K) dB

        参数:
            segBlockFfts — 各段输入块 FFT, 由 _precomputeInputBlockFfts 生成
            nId2         — 扇区 ID
            outputLength — 输出长度

        返回:
            np.ndarray — 累加相关幅度曲线
        """
        M = self.segmentLength
        corrAccum = np.zeros(outputLength)

        for segIdx in range(self.NumSegments):
            templateConj = self.segmentFftConj[nId2][segIdx]
            blockFfts = segBlockFfts[segIdx]

            # 频域相关: 批量乘法 + 批量 IFFT
            # templateConj shape: (B,), blockFfts shape: (numBlocks, B)
            # 广播乘法: (numBlocks, B) × (B,) → (numBlocks, B)
            corrBlocks = np.fft.ifft(blockFfts * templateConj[np.newaxis, :], axis=1)

            # overlap-save: 有效输出为每个块的前 M 个样本
            # reshape 为 1D 后截取 outputLength
            validCorr = np.abs(corrBlocks[:, :M]).ravel()
            actualLen = min(len(validCorr), outputLength)
            corrAccum[:actualLen] += validCorr[:actualLen]

        return corrAccum

    # =====================================================================
    # 多峰分析
    # =====================================================================

    def _analyzePeaks(self, corrArray: np.ndarray, corrResultsPerNid2: list = None) -> dict:
        """
        多峰分析: 在相关曲线上检测显著峰并识别 CP 回波

        检测逻辑:
            1. 使用 find_peaks 检测所有高于阈值的局部极大值
            2. 按幅度降序排列
            3. 检查相邻峰间距是否匹配 CP 长度 (CP 回波特征)
            4. 若检测到 CP 回波对，则:
               - 主峰 (较大) = OFDM 符号起点，为真正的定时点
               - 次峰 (较小) = CP 回波峰
               - 峰间距 ≈ CP 长度，可用于验证检测结果

        参数:
            corrArray:          互相关幅度曲线（最优 N_ID_2）
            corrResultsPerNid2: 3 个 N_ID_2 的相关曲线列表，用于标注每个峰的扇区 ID

        返回:
            dict — 多峰分析结果，每个峰包含 nId2 字段
        """
        if corrArray is None:
            return None

        # 峰值检测阈值: 最大峰值的 80% (仅保留显著峰)
        peakThreshold = np.max(corrArray) * 0.8

        # 最小峰间距: 一个 OFDM 符号长度 (避免检测到主瓣旁瓣)
        minPeakDistance = max(self.config.FftSize + self.config.NormalCpLength, 1)

        # 检测局部极大值
        peakIndices, _ = scipySignal.find_peaks(
            corrArray,
            height=peakThreshold,
            distance=minPeakDistance
        )

        peakValues = corrArray[peakIndices]

        # 按幅度降序排列，取前 N 个显著峰
        sortedOrder = np.argsort(peakValues)[::-1]
        maxPeaksToReport = 5
        topIndices = peakIndices[sortedOrder[:maxPeaksToReport]]
        topValues = peakValues[sortedOrder[:maxPeaksToReport]]

        peaks = []
        for i in range(len(topIndices)):
            idx = int(topIndices[i])
            # 查找该峰位置上 3 条 N_ID_2 曲线中值最大的扇区 ID
            nId2 = 0
            if corrResultsPerNid2 is not None and all(c is not None for c in corrResultsPerNid2):
                maxVal = corrResultsPerNid2[0][idx]
                for nid in range(1, PssConstants.SectorCount):
                    if corrResultsPerNid2[nid][idx] > maxVal:
                        maxVal = corrResultsPerNid2[nid][idx]
                        nId2 = nid
            peaks.append({
                "index": idx,
                "value": float(topValues[i]),
                "nId2":  nId2
            })

        # =================================================================
        # CP 回波识别 & 近距双峰分析
        # =================================================================
        cpLength = self.config.NormalCpLength
        symbolLength = self.config.FftSize + cpLength
        cpTolerance = cpLength * 0.2
        cpEchoPair = None
        closePeakPairs = []

        for i in range(len(topIndices)):
            for j in range(i + 1, len(topIndices)):
                peakDistance = int(topIndices[i]) - int(topIndices[j])
                if peakDistance < 0:
                    peakDistance = -peakDistance

                # CP 回波检测: 间距 ≈ CP 长度
                if cpEchoPair is None and abs(peakDistance - cpLength) <= cpTolerance:
                    mainPeakIdx = int(topIndices[i])
                    cpEchoIdx = int(topIndices[j])
                    # 查找主峰和回波峰各自对应的 N_ID_2
                    mainNId2 = peaks[i]["nId2"] if i < len(peaks) else 0
                    cpNId2 = peaks[j]["nId2"] if j < len(peaks) else 0
                    cpEchoPair = {
                        "mainPeakIndex":   mainPeakIdx,
                        "mainPeakValue":   float(corrArray[mainPeakIdx]),
                        "mainPeakNId2":    mainNId2,
                        "cpEchoPeakIndex": cpEchoIdx,
                        "cpEchoPeakValue": float(corrArray[cpEchoIdx]),
                        "cpEchoPeakNId2":  cpNId2,
                        "measuredCpLength": peakDistance,
                        "expectedCpLength": cpLength,
                        "cpLengthError":    abs(peakDistance - cpLength)
                    }

                # 近距双峰记录: 间距 < 2 个符号长度
                if peakDistance < symbolLength * 2:
                    closePeakPairs.append({
                        "peak1Index":       int(topIndices[i]),
                        "peak1Value":       float(topValues[i]),
                        "peak2Index":       int(topIndices[j]),
                        "peak2Value":       float(topValues[j]),
                        "distance":         peakDistance,
                        "distanceInSymbols": round(peakDistance / symbolLength, 2)
                    })

        result = {
            "detectedPeaks":  peaks,
            "peakCount":      len(peakIndices),
            "closePeakPairs": closePeakPairs if closePeakPairs else None,
            "cpEchoPair":     cpEchoPair
        }

        if closePeakPairs:
            for pair in closePeakPairs:
                logger.info(f"  近距双峰: 峰1@{pair['peak1Index']}({pair['peak1Value']:.1f}), "
                            f"峰2@{pair['peak2Index']}({pair['peak2Value']:.1f}), "
                            f"间距={pair['distance']}点 ({pair['distanceInSymbols']}符号)")
        if cpEchoPair is not None:
            logger.info(f"  CP 回波检测: 主峰@{cpEchoPair['mainPeakIndex']}, "
                        f"回波峰@{cpEchoPair['cpEchoPeakIndex']}, "
                        f"测量CP长度={cpEchoPair['measuredCpLength']} "
                        f"(预期={cpLength}, 误差={cpEchoPair['cpLengthError']})")

        return result
