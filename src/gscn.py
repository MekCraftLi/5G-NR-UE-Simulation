import numpy as np

class GscnConstants:
    """
        3GPP TS 38.101-1 GSCN 计算相关常量 (大驼峰命名)
        """
    # GSCN 范围边界
    MaxGscnBelow3GHz = 7499
    MaxGscnBelow24GHz = 22255
    OffsetGscnMidRange = 7499  # 3GHz - 24.25GHz 的 GSCN 偏移量
    OffsetGscnHighRange = 22256  # > 24.25GHz 的 GSCN 偏移量

    # 绝对频率基准点 (Hz)
    BaseFreqMidRange = 3000e6  # 3000 MHz
    BaseFreqHighRange = 24250.08e6  # 24250.08 MHz

    # 频率步进系数 (Hz)
    StepNLowRange = 1.2e6  # < 3GHz 的 N 步进 (1.2 MHz)
    StepMLowRange = 50e3  # < 3GHz 的 M 步进 (50 kHz)
    StepNMidRange = 1.44e6  # 3-24GHz 的 N 步进 (1.44 MHz)
    StepNHighRange = 17.28e6  # >24GHz 的 N 步进 (17.28 MHz)

class GscnRaster:
    '''
    根据 3GPP TS 38.101-1表 5.4.3.1-1 实现GSCN到绝对物理频率的转换
    '''


    @staticmethod
    def getAbsoluteFrequency(gscn: int) -> float:
        if gscn <= GscnConstants.MaxGscnBelow3GHz:
            # 频段 0 - 3000MHz
            # F_SSB = N * 1.2MHz + M * 50kHz
            # GSCN = 3N + (M - 3) / 2, 其中M in {1, 3, 5}

            n = (gscn + 1) // 3

            mRemainder = (gscn + 1) % 3

            if mRemainder == 0:
                m = 1
            elif mRemainder == 1:
                m = 3
            else:
                m = 5

            return n * GscnConstants.StepNLowRange + m * GscnConstants.StepMLowRange

        elif gscn <= GscnConstants.MaxGscnBelow24GHz:
            # 频段 3000MHz - 24250.08MHz
            n = gscn - GscnConstants.OffsetGscnMidRange
            return GscnConstants.BaseFreqMidRange + n * GscnConstants.StepNMidRange

        else:
            n = gscn - GscnConstants.OffsetGscnHighRange
            return GscnConstants.BaseFreqHighRange + n * GscnConstants.StepNHighRange

