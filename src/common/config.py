from dataclasses import dataclass


@dataclass
class BandRasterConfig:
    """
    频段同步栅格配置 —— 依据 TS 38.101-1 Table 5.4.3.3-1

    字段说明:
        FirstGscn      : 该频段起始 GSCN
        LastGscn       : 该频段结束 GSCN
        StepSize       : GSCN 步进
        Scs            : SS Block 子载波间隔 (Hz)
        SsBlockPattern : SS Block Pattern (Case A/B/C/D/E)
    """
    FirstGscn: int
    LastGscn: int
    StepSize: int
    Scs: int = 30000                # 子载波间隔 (Hz)
    SsBlockPattern: str = "Case C"  # SS Block Pattern


class BandConstants:
    """
    3GPP TS 38.101-1 Table 5.4.3.3-1 各频段同步栅格常量

    SS Block Pattern 定义 (TS 38.213 Clause 4.1):
        - Case A: SCS = 15 kHz
        - Case B: SCS = 30 kHz
        - Case C: SCS = 30 kHz (FR1)
        - Case D: SCS = 120 kHz (FR2)
        - Case E: SCS = 240 kHz (FR2)

    字段说明:
        BandN1   : n1  频段, SCS=15kHz, Case A
        BandN41A : n41 频段, SCS=15kHz, Case A
        BandN41C : n41 频段, SCS=30kHz, Case C
        BandN78  : n78 频段, SCS=30kHz, Case C
        BandN104 : n104频段, SCS=30kHz, Case C
    """
    # n1 频段: 2100 MHz 范围
    BandN1 = BandRasterConfig(FirstGscn=5279, LastGscn=5419, StepSize=1, Scs=15000, SsBlockPattern="Case A")

    # n41 频段: 2500 MHz 范围 (两种 SCS 配置)
    BandN41A = BandRasterConfig(FirstGscn=6246, LastGscn=6717, StepSize=3, Scs=15000, SsBlockPattern="Case A")
    BandN41C = BandRasterConfig(FirstGscn=6252, LastGscn=6714, StepSize=3, Scs=30000, SsBlockPattern="Case C")

    # n78 频段: 3500 MHz 范围
    BandN78 = BandRasterConfig(FirstGscn=7711, LastGscn=8051, StepSize=1, Scs=30000, SsBlockPattern="Case C")

    # n104 频段: 6000 MHz 范围
    BandN104 = BandRasterConfig(FirstGscn=9882, LastGscn=10358, StepSize=7, Scs=30000, SsBlockPattern="Case C")

    @staticmethod
    def getMatchingBands(scs: int) -> list:
        """
        根据子载波间隔筛选匹配的频段栅格条目

        依据 TS 38.101-1 Table 5.4.3.3-1, 每个频段条目关联了特定的 SCS 和 SS Block Pattern。
        给定 SCS 后, 仅保留 SCS 一致的条目。

        参数:
            scs: 子载波间隔 (Hz)

        返回:
            匹配的 BandRasterConfig 列表 (可能为空)
        """
        matching = []
        for attrName in dir(BandConstants):
            attr = getattr(BandConstants, attrName)
            if isinstance(attr, BandRasterConfig) and attr.Scs == scs:
                matching.append(attr)
        return matching


class OfdmConstants:
    """
    OFDM 基准常量 —— 依据 TS 38.211 Clause 5.3.1

    字段说明:
        BaseFftSize        : 基准 FFT 点数 2048 (对应 SCS=15kHz, fs=30.72MHz)
        BaseNormalCpLength : 基准常规 CP 长度 144 采样点 (对应 2048 FFT)
        SymbolsPerSlot     : Normal CP 下每时隙符号数 14
    """
    BaseFftSize = 2048
    BaseNormalCpLength = 144
    SymbolsPerSlot = 14


@dataclass
class SsbConfig:
    """
    SSB 配置 —— 运行时构造，参数由 main.py 根据协议推导后传入

    参数来源:
        - SampleRate        : 外部输入 (采样率)
        - SubcarrierSpacing : 外部输入 (子载波间隔)
        - FftSize           : 推导 = SampleRate / SubcarrierSpacing
        - NormalCpLength    : 推导 = BaseNormalCpLength × (FftSize / BaseFftSize)
        - SsbSymbolCount    : 协议固定值 4 (TS 38.211 Clause 7.4.3.1)
        - TargetBandRaster  : 根据目标频段选择
    """
    # 运行时参数 (必须传入)
    SampleRate: float
    SubcarrierSpacing: int
    FftSize: int
    NormalCpLength: int

    # 协议固定常量
    SsbSymbolCount: int = 4  # TS 38.211 Clause 7.4.3.1: SS/PBCH block 包含 4 个 OFDM 符号

    # 频段配置 (可选，默认 n104)
    TargetBandRaster: BandRasterConfig = None

    def __post_init__(self):
        # 根据 SCS 自动选择匹配的频段栅格 (TS 38.101-1 Table 5.4.3.3-1)
        if self.TargetBandRaster is None:
            matchingBands = BandConstants.getMatchingBands(self.SubcarrierSpacing)
            if len(matchingBands) == 1:
                self.TargetBandRaster = matchingBands[0]
            elif len(matchingBands) > 1:
                # 多个频段匹配时默认取第一个 (实际应由上层指定具体频段)
                self.TargetBandRaster = matchingBands[0]
            else:
                raise ValueError(f"未找到 SCS={self.SubcarrierSpacing} 对应的频段栅格条目")


def createSsbConfig(sampleRate: float, subcarrierSpacing: int) -> SsbConfig:
    """
    工厂函数: 根据采样率和子载波间隔创建 SsbConfig

    推导逻辑:
        1. FFT 点数 = 采样率 / 子载波间隔
        2. 常规 CP 长度 = 基准 CP × (FFT / 基准 FFT)

    参数:
        sampleRate:        采样率 (Hz)
        subcarrierSpacing: 子载波间隔 (Hz)

    返回:
        SsbConfig 实例
    """
    fftSize = int(sampleRate / subcarrierSpacing)
    normalCpLength = int(OfdmConstants.BaseNormalCpLength * fftSize / OfdmConstants.BaseFftSize)

    return SsbConfig(
        SampleRate=sampleRate,
        SubcarrierSpacing=subcarrierSpacing,
        FftSize=fftSize,
        NormalCpLength=normalCpLength
    )
