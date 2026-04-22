import numpy as np

from common.config import SsbConfig, OfdmConstants


class CpManager:
    """
    循环前缀管理器 —— 依据 TS 38.211 Clause 5.3.1 计算 CP 长度

    CP 类型:
        - 常规 CP (Normal CP): 除每时隙第一个符号外的所有符号
        - 长 CP (Long CP): 每时隙第一个符号，用于对齐半毫秒边界

    计算逻辑:
        1. 常规 CP 从基准值缩放: CP = 144 × (N_FFT / 2048)
        2. 长 CP 由时隙总采样数反推，保证时隙边界对齐
    """

    def __init__(self, config: SsbConfig):
        self.config = config

        # =================================================================
        # 常规 CP 长度: 从基准 FFT=2048 的 CP=144 缩放
        # 参考: TS 38.211 Clause 5.3.1
        # =================================================================
        scalingFactor = self.config.FftSize / OfdmConstants.BaseFftSize
        self.normalCpLength = int(OfdmConstants.BaseNormalCpLength * scalingFactor)

        # =================================================================
        # 长 CP 长度: 由时隙总采样数反推
        #
        # 时隙时长 = 0.5ms (对于 μ=0) 或 0.5ms / 2^μ (对于 μ>0)
        # 其中 μ = log2(SCS / 15kHz)
        #
        # 时隙总采样数 = fs × 时隙时长
        # 总 CP 采样数 = 时隙总采样数 - 14 × N_FFT
        # 长 CP = 总 CP - 13 × 常规 CP
        # =================================================================
        # 计算子载波间隔参数 μ (mu)
        mu = int(np.log2(self.config.SubcarrierSpacing / 15000))

        # 时隙时长 (秒)
        slotDuration = 0.5e-3 / (2 ** mu)

        # 时隙总采样数
        totalSamplesPerSlot = int(self.config.SampleRate * slotDuration)

        # 14 个符号的 FFT 总采样数
        totalFftSamples = self.config.FftSize * OfdmConstants.SymbolsPerSlot

        # 时隙内所有 CP 的总采样数
        totalCpSamples = totalSamplesPerSlot - totalFftSamples

        # 长 CP = 总 CP - 13 × 常规 CP (第一个符号用长 CP，其余 13 个用常规 CP)
        self.longCpLength = totalCpSamples - ((OfdmConstants.SymbolsPerSlot - 1) * self.normalCpLength)

    def getCpLength(self, symbolIndexInSlot: int) -> int:
        """
        获取指定符号的 CP 长度

        参数:
            symbolIndexInSlot: 时隙内符号索引 (0~13)

        返回:
            CP 长度 (采样点数)
        """
        if symbolIndexInSlot == 0:
            return self.longCpLength
        return self.normalCpLength

    def getSymbolLength(self, symbolIndexInSlot: int) -> int:
        """
        获取指定符号的总长度 (FFT + CP)

        参数:
            symbolIndexInSlot: 时隙内符号索引 (0~13)

        返回:
            符号长度 (采样点数)
        """
        return self.config.FftSize + self.getCpLength(symbolIndexInSlot)
