import logging

import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager

logger = logging.getLogger(__name__)


class OfdmDemodulator:
    """
    OFDM 符号提取与 FFT 解调工具 —— 供 SSS/PBCH 阶段使用

    功能:
        - 从时域接收信号中提取指定 OFDM 符号
        - 去 CP + FFT 变换得到频域数据
        - 批量提取 SS/PBCH block 的 4 个符号

    依据:
        - TS 38.211 Clause 5.3.1 (OFDM 基带信号生成)
        - TS 38.211 Table 7.4.3.1-1 (SS/PBCH block 资源映射)
    """

    def __init__(self, config: SsbConfig):
        """
        初始化 OFDM 解调器

        参数:
            config: SSB 配置 (含 FftSize, SampleRate 等)
        """
        self.config = config
        self.cpManager = CpManager(config)

    def extractSymbol(self, rxSignal: np.ndarray, symbolStart: int) -> np.ndarray:
        """
        从时域信号中提取单个 OFDM 符号并 FFT 到频域

        步骤:
            1. 从 symbolStart 处截取 CP + FFT 长度的数据
            2. 跳过 CP，取 FFT 点数据进行 FFT
            3. fftshift 将直流分量移到中央

        参数:
            rxSignal    — 接收基带 I/Q 信号 (复数数组)
            symbolStart — 符号 CP 起始位置 (采样点索引)

        返回:
            np.ndarray — 频域数据 (长度为 FftSize)
        """
        fftSize = self.config.FftSize
        end = symbolStart + fftSize + self.cpManager.normalCpLength
        if end > len(rxSignal):
            logger.warning(f"符号提取越界: start={symbolStart}, end={end}, signalLen={len(rxSignal)}")
            return np.zeros(fftSize, dtype=complex)

        # 计算该符号的 CP 长度 (需要知道时隙内符号索引，此处使用常规 CP)
        cpLength = self.cpManager.normalCpLength

        # 跳过 CP，取 FFT 数据
        fftData = rxSignal[symbolStart + cpLength: symbolStart + cpLength + fftSize]

        # FFT + fftshift
        return np.fft.fftshift(np.fft.fft(fftData))

    def extractSsbSymbols(self, rxSignal: np.ndarray, ssbStart: int) -> list:
        """
        提取 SS/PBCH block 的 4 个 OFDM 符号的频域数据

        SS/PBCH block 结构 (TS 38.211 Table 7.4.3.1-1):
            符号 0: PSS  (子载波 56~182)
            符号 1: PBCH + DM-RS (子载波 0~239)
            符号 2: SSS + PBCH + DM-RS
            符号 3: PBCH + DM-RS (子载波 0~239)

        参数:
            rxSignal — 接收基带 I/Q 信号 (复数数组)
            ssbStart — SS/PBCH block 首符号 CP 起始位置

        返回:
            list[np.ndarray] — 长度为 4，每个元素为对应符号的频域数据
        """
        symbols = []
        currentPos = ssbStart

        for symIdx in range(self.config.SsbSymbolCount):
            symbolLength = self.cpManager.getSymbolLength(symIdx)
            freqData = self.extractSymbol(rxSignal, currentPos)
            symbols.append(freqData)
            currentPos += symbolLength

        return symbols

    def getSssSymbolOffset(self, ssbStart: int) -> int:
        """
        计算 SSS 符号 (SS/PBCH block 符号 2) 相对于 ssbStart 的采样偏移

        SSS 位于 SS/PBCH block 的第 2 个符号 (TS 38.211 Table 7.4.3.1-1)。
        偏移 = 符号 0 长度 + 符号 1 长度

        参数:
            ssbStart — SS/PBCH block 首符号 CP 起始位置

        返回:
            int — SSS 符号的 CP 起始位置 (绝对采样点索引)
        """
        offset = 0
        for symIdx in range(2):  # 跳过符号 0 和符号 1
            offset += self.cpManager.getSymbolLength(symIdx)
        return ssbStart + offset
