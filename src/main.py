import numpy as np
import logging
from config import createSsbConfig, BandConstants
from gscn import GscnRaster
from synchronization import CellSearcher
from visualizer import VisualProbes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def main():
    logging.info("1. 初始化系统基带配置...")

    # =================================================================
    # 【核心输入】全工程仅有这 3 个参数和 1 个信号文件是根据实际情况给定的
    # =================================================================
    SAMPLE_RATE = 30.72e6       # 采样频率 fs (Hz)
    SCS = 30000                 # 子载波间隔 (Hz)
    N_RB = 51                   # 资源块数量 (PRBs)

    # =================================================================
    # 加载接收信号
    # =================================================================
    try:
        rxSignal = np.load('../data/rxSignal.npy')
        logging.info(f"成功加载接收信号，数据长度: {len(rxSignal)} 点")
    except FileNotFoundError:
        logging.error("未找到 npy 文件！")
        return

    # =================================================================
    # 2. 根据 3GPP 协议动态推导 OFDM 参数
    # =================================================================
    logging.info("\n=== 2. 根据 3GPP 协议动态推导 OFDM 参数 ===")

    # --- A. FFT 点数推导 ---
    fftSize = int(SAMPLE_RATE / SCS)
    logging.info(f"FFT 点数: {fftSize} = {SAMPLE_RATE/1e6:.2f} MHz / {SCS/1e3:.0f} kHz")

    # --- B. CP 长度推导 ---
    normalCpLength = int(144 * fftSize / 2048)
    logging.info(f"常规 CP 长度: {normalCpLength} 采样点")

    # --- C. 构造 SSB 配置 ---
    ssbConfig = createSsbConfig(
        sampleRate=SAMPLE_RATE,
        subcarrierSpacing=SCS
    )
    logging.info(f"SSB 配置: FFT={ssbConfig.FftSize}, CP={ssbConfig.NormalCpLength}")

    # =================================================================
    # 3. 根据 SCS 确定目标频段 (TS 38.101-1 Table 5.4.3.3-1)
    # =================================================================
    logging.info("\n=== 3. 根据 SCS 确定目标频段 ===")

    matchingBands = BandConstants.getMatchingBands(SCS)

    if not matchingBands:
        logging.error(f"未找到 SCS={SCS} 对应的频段栅格条目，请检查参数")
        return

    logging.info(f"SCS={SCS/1e3:.0f} kHz 匹配到 {len(matchingBands)} 个频段栅格条目:")
    for i, band in enumerate(matchingBands):
        gscnCount = len(range(band.FirstGscn, band.LastGscn + 1, band.StepSize))
        logging.info(f"  [{i}] Pattern={band.SsBlockPattern}, GSCN=[{band.FirstGscn}, {band.LastGscn}], "
                     f"Step={band.StepSize}, 候选数={gscnCount}")

    targetBand = matchingBands[0]
    logging.info(f"选定频段栅格: Pattern={targetBand.SsBlockPattern}, "
                 f"GSCN=[{targetBand.FirstGscn}, {targetBand.LastGscn}], Step={targetBand.StepSize}")

    gscnList = list(range(targetBand.FirstGscn, targetBand.LastGscn + 1, targetBand.StepSize))
    logging.info(f"候选 GSCN 数量: {len(gscnList)}")

    # =================================================================
    # 4. 构建候选频点列表 (GSCN → 绝对频率)
    # =================================================================
    logging.info("\n=== 4. 构建候选频点列表 ===")

    # 盲搜假设接收信号为零中频 (中心频率 = DC = 0 Hz)
    # 每个 GSCN 的绝对频率即为相对 DC 的频偏
    # 参考: TS 38.101-1 Table 5.4.3.1-1
    validGscnList = []
    for gscn in gscnList:
        absFreq = GscnRaster.getAbsoluteFrequency(gscn)
        validGscnList.append((gscn, absFreq))

    logging.info(f"GSCN 频率范围: {validGscnList[0][1]/1e6:.2f} MHz ~ {validGscnList[-1][1]/1e6:.2f} MHz")

    # =================================================================
    # 5. 执行 PSS 盲搜 (TS 38.211 Clause 7.4.2.2)
    # =================================================================
    logging.info("\n=== 5. 执行 PSS 盲搜 ===")

    searcher = CellSearcher(ssbConfig)
    logging.info("本地 PSS 时域模板已生成 (3 个扇区)")

    result = searcher.detectPss(rxSignal, validGscnList)

    logging.info(f"盲搜完成:")
    logging.info(f"  GSCN={result['gscn']}")
    logging.info(f"  SSREF={result['freqOffset']/1e6:.4f} MHz")
    logging.info(f"  N_ID_2={result['nId2']}")
    logging.info(f"  TimingOffset={result['timingOffset']} 采样点")
    logging.info(f"  PeakValue={result['peakValue']:.6f}")

    # =================================================================
    # 6. 结果可视化与保存
    # =================================================================
    logging.info("\n=== 6. 结果可视化与保存 ===")

    VisualProbes.plotPssSearchValidation(result, validGscnList, len(rxSignal))

    logging.info("PSS 盲搜流程全部完成")


if __name__ == "__main__":
    main()
