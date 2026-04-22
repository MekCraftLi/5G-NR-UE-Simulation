import logging

import numpy as np

from common.config import createSsbConfig, BandConstants
from common.cpManager import CpManager
from common.gscn import GscnRaster
from pss.pssDetector import PssDetector
from pss.pssVisualizer import PssVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

logger = logging.getLogger(__name__)


def main():
    """
    5G NR 小区搜索流水线总入口

    流程:
        1. 初始化系统基带配置
        2. 推导 OFDM 参数 (FFT 点数、CP 长度)
        3. 根据 SCS 确定目标频段 (TS 38.101-1 Table 5.4.3.3-1)
        4. 构建候选频点列表 (GSCN → 绝对频率)
        5. PSS 盲搜 (TS 38.211 Clause 7.4.2.2)
        6. 结果可视化与保存
        7. SSS 检测 (预留)
        8. PBCH 解码 (预留)
    """
    # =================================================================
    # 1. 初始化系统基带配置
    # =================================================================
    logger.info("1. 初始化系统基带配置...")

    # 【核心输入】全工程仅有这些参数和信号文件是根据实际情况给定的
    SAMPLE_RATE = 30.72e6   # 采样频率 fs (Hz)
    SCS = 30000             # 子载波间隔 (Hz)

    # 加载接收信号
    try:
        rxSignal = np.load('../data/rxSignal.npy')
        logger.info(f"成功加载接收信号，数据长度: {len(rxSignal)} 点")
    except FileNotFoundError:
        logger.error("未找到 npy 文件！")
        return

    # =================================================================
    # 2. 根据 3GPP 协议动态推导 OFDM 参数
    # =================================================================
    logger.info("\n=== 2. 根据 3GPP 协议动态推导 OFDM 参数 ===")

    ssbConfig = createSsbConfig(
        sampleRate=SAMPLE_RATE,
        subcarrierSpacing=SCS
    )
    cpManager = CpManager(ssbConfig)

    logger.info(f"FFT 点数: {ssbConfig.FftSize} = {SAMPLE_RATE/1e6:.2f} MHz / {SCS/1e3:.0f} kHz")
    logger.info(f"常规 CP 长度: {ssbConfig.NormalCpLength} 采样点")
    logger.info(f"长 CP 长度: {cpManager.longCpLength} 采样点")

    # =================================================================
    # 3. 根据 SCS 确定目标频段 (TS 38.101-1 Table 5.4.3.3-1)
    # =================================================================
    logger.info("\n=== 3. 根据 SCS 确定目标频段 ===")

    matchingBands = BandConstants.getMatchingBands(SCS)

    if not matchingBands:
        logger.error(f"未找到 SCS={SCS} 对应的频段栅格条目，请检查参数")
        return

    logger.info(f"SCS={SCS/1e3:.0f} kHz 匹配到 {len(matchingBands)} 个频段栅格条目:")
    for i, band in enumerate(matchingBands):
        gscnCount = len(range(band.FirstGscn, band.LastGscn + 1, band.StepSize))
        logger.info(f"  [{i}] Pattern={band.SsBlockPattern}, GSCN=[{band.FirstGscn}, {band.LastGscn}], "
                     f"Step={band.StepSize}, 候选数={gscnCount}")

    targetBand = matchingBands[0]
    logger.info(f"选定频段栅格: Pattern={targetBand.SsBlockPattern}, "
                f"GSCN=[{targetBand.FirstGscn}, {targetBand.LastGscn}], Step={targetBand.StepSize}")

    gscnList = list(range(targetBand.FirstGscn, targetBand.LastGscn + 1, targetBand.StepSize))
    logger.info(f"候选 GSCN 数量: {len(gscnList)}")

    # =================================================================
    # 4. 构建候选频点列表 (GSCN → 绝对频率)
    # =================================================================
    logger.info("\n=== 4. 构建候选频点列表 ===")

    # 盲搜假设接收信号为零中频 (中心频率 = DC = 0 Hz)
    # 每个 GSCN 的绝对频率即为相对 DC 的频偏
    # 参考: TS 38.101-1 Table 5.4.3.1-1
    validGscnList = []
    for gscn in gscnList:
        absFreq = GscnRaster.getAbsoluteFrequency(gscn)
        validGscnList.append((gscn, absFreq))

    logger.info(f"GSCN 频率范围: {validGscnList[0][1]/1e6:.2f} MHz ~ {validGscnList[-1][1]/1e6:.2f} MHz")

    # =================================================================
    # 5. 执行 PSS 盲搜 (TS 38.211 Clause 7.4.2.2)
    # =================================================================
    logger.info("\n=== 5. 执行 PSS 盲搜 ===")

    pssDetector = PssDetector(ssbConfig)
    logger.info("本地 PSS 时域模板已生成 (3 个扇区)")

    pssResult = pssDetector.detectPss(rxSignal, validGscnList)

    logger.info("盲搜完成:")
    logger.info(f"  GSCN={pssResult['gscn']}")
    logger.info(f"  SSREF={pssResult['freqOffset']/1e6:.4f} MHz")
    logger.info(f"  N_ID_2={pssResult['nId2']}")
    logger.info(f"  TimingOffset={pssResult['timingOffset']} 采样点")
    logger.info(f"  PeakValue={pssResult['peakValue']:.6f}")

    # =================================================================
    # 6. 结果可视化与保存
    # =================================================================
    logger.info("\n=== 6. 结果可视化与保存 ===")

    PssVisualizer.plotPssSearchValidation(pssResult, validGscnList, len(rxSignal))

    logger.info("PSS 盲搜流程全部完成")

    # =================================================================
    # 7. SSS 检测 (预留 — 待实现)
    # =================================================================
    # from sss.sssDetector import SssDetector
    # sssDetector = SssDetector(ssbConfig)
    # sssResult = sssDetector.detectSss(rxSignal, pssResult)

    # =================================================================
    # 8. PBCH 解码 (预留 — 待实现)
    # =================================================================
    # from pbch.pbchDecoder import PbchDecoder
    # pbchDecoder = PbchDecoder(ssbConfig)
    # pbchResult = pbchDecoder.decodePbch(rxSignal, sssResult, pssResult)


if __name__ == "__main__":
    main()
