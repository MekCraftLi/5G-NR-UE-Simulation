class GscnConstants:
    """
    3GPP TS 38.101-1 GSCN 计算相关常量

    字段说明:
        MaxGscnBelow3GHz   : <3GHz 频段最大 GSCN 值
        MaxGscnBelow24GHz  : <24.25GHz 频段最大 GSCN 值
        OffsetGscnMidRange : 3GHz~24.25GHz 的 GSCN 偏移量
        OffsetGscnHighRange: >24.25GHz 的 GSCN 偏移量
        BaseFreqMidRange   : 中频段基准频率 3000 MHz
        BaseFreqHighRange  : 高频段基准频率 24250.08 MHz
        StepNLowRange      : <3GHz 的 N 步进 1.2 MHz
        StepMLowRange      : <3GHz 的 M 步进 50 kHz
        StepNMidRange      : 3~24GHz 的 N 步进 1.44 MHz
        StepNHighRange     : >24GHz 的 N 步进 17.28 MHz
    """
    # GSCN 范围边界 (>3MHz)
    MaxGscnBelow3GHz = 7499
    MaxGscnBelow24GHz = 22255
    MaxGscnBelow100GHz = 26639
    OffsetGscnMidRange = 7499       # 3GHz ~ 24.25GHz 的 GSCN 偏移量
    OffsetGscnHighRange = 22256     # > 24.25GHz 的 GSCN 偏移量

    # 绝对频率基准点 (Hz)
    BaseFreqMidRange = 3000e6       # 3000 MHz
    BaseFreqHighRange = 24250.08e6  # 24250.08 MHz

    # 频率步进系数 (Hz)
    StepNLowRange = 1.2e6           # < 3GHz 的 N 步进 (1.2 MHz)
    StepMLowRange = 50e3            # < 3GHz 的 M 步进 (50 kHz)
    StepNMidRange = 1.44e6          # 3 ~ 24GHz 的 N 步进 (1.44 MHz)
    StepNHighRange = 17.28e6        # > 24GHz 的 N 步进 (17.28 MHz)

    # 3MHz 专用 GSCN (TS 38.101-1 Table 5.4.3.1-2 / 5.4.3.1-3)
    OffsetGscn3MHz = 26638
    MinGscn3MHz = 26640
    MaxGscn3MHz = 31634
    MinGscn3MHzExtended = 33802
    MaxGscn3MHzExtended = 33804
    StepN3MHz = 600e3
    Offset3MHz = 300e3
    AdditionalGscnN100 = {
        41637: 920.73e6,
        41638: 921.45e6,
    }


class GscnRaster:
    """
    GSCN 到绝对频率转换 —— 依据 TS 38.101-1 Table 5.4.3.1-1

    三段公式:
        - <3GHz:       F_SSB = N × 1.2MHz + M × 50kHz, M ∈ {1, 3, 5}
        - 3~24.25GHz:  F_SSB = 3000MHz + N × 1.44MHz
        - >24.25GHz:   F_SSB = 24250.08MHz + N × 17.28MHz
    """

    @staticmethod
    def getAbsoluteFrequency(gscn: int) -> float:
        """
        将 GSCN 编号转换为绝对频率 (Hz)

        参数:
            gscn: 全局同步栅格信道编号

        返回:
            SSREF 绝对频率 (Hz)
        """
        # 3MHz channel raster specific entries
        if gscn in GscnConstants.AdditionalGscnN100:
            return GscnConstants.AdditionalGscnN100[gscn]

        if GscnConstants.MinGscn3MHz <= gscn <= GscnConstants.MaxGscn3MHz or \
                GscnConstants.MinGscn3MHzExtended <= gscn <= GscnConstants.MaxGscn3MHzExtended:
            return GscnRaster._getAbsoluteFrequencyFor3MHz(gscn)

        # >3MHz global raster entries
        if gscn <= GscnConstants.MaxGscnBelow3GHz:
            # 频段 0 ~ 3000MHz
            # F_SSB = N × 1.2MHz + M × 50kHz
            # GSCN = 3N + (M - 3) / 2, 其中 M ∈ {1, 3, 5}
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
            # 频段 3000MHz ~ 24250.08MHz
            n = gscn - GscnConstants.OffsetGscnMidRange
            return GscnConstants.BaseFreqMidRange + n * GscnConstants.StepNMidRange

        elif gscn <= GscnConstants.MaxGscnBelow100GHz:
            # 频段 > 24250.08MHz
            n = gscn - GscnConstants.OffsetGscnHighRange
            return GscnConstants.BaseFreqHighRange + n * GscnConstants.StepNHighRange

        raise ValueError(f"Unsupported GSCN value: {gscn}")

    @staticmethod
    def _getAbsoluteFrequencyFor3MHz(gscn: int) -> float:
        """
        3MHz channel bandwidth 专用 GSCN 频率映射 (TS 38.101-1 Table 5.4.3.1-2)

        SSREF = N*600k + M*50k + 300k
        GSCN  = 26638 + 3N + (M-3)/2, M∈{1,3,5}
        """
        candidates = (
            (1, 26637),  # M=1 => gscn = 26637 + 3N
            (3, 26638),  # M=3 => gscn = 26638 + 3N
            (5, 26639),  # M=5 => gscn = 26639 + 3N
        )
        for m, offset in candidates:
            delta = gscn - offset
            if delta % 3 == 0:
                n = delta // 3
                if n >= 1:
                    return n * GscnConstants.StepN3MHz + m * GscnConstants.StepMLowRange + GscnConstants.Offset3MHz
        raise ValueError(f"Invalid 3MHz GSCN value: {gscn}")
