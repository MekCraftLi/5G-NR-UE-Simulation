import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class VisualProbes:
    """
    PSS 盲搜结果可视化与记录

    输出:
        - output/pss_search_heatmap.png : 2D 热力图 + 1D 相关峰剖面 (含多峰标注)
        - output/pss_search_result.json : 盲搜结果参数 (含多峰分析)
    """

    # 输出目录 (相对于 src/)
    OutputDir = os.path.join(os.path.dirname(__file__), '..', 'output')

    @classmethod
    def _ensureOutputDir(cls):
        """确保输出目录存在"""
        os.makedirs(cls.OutputDir, exist_ok=True)

    @staticmethod
    def plotPssSearchValidation(detectResult: dict, validGscnList: list, rxLength: int):
        """
        可视化 PSS 盲搜结果并保存到本地

        生成内容:
            1. 2D 热力图: 候选频点 (纵轴) × 时延采样点 (横轴)
            2. 1D 相关峰剖面: 最优频点处的互相关幅度曲线 (含多峰标注)
            3. JSON 结果文件: 盲搜检测参数

        参数:
            detectResult   : CellSearcher.detectPss() 的返回值
            validGscnList  : 候选频点列表 [(gscn, freqOffset), ...]
            rxLength       : 接收信号长度 (采样点数)
        """
        VisualProbes._ensureOutputDir()

        fig = plt.figure(figsize=(14, 8))

        # =================================================================
        # 1. 2D 搜索热力图 (纵轴: GSCN 频点, 横轴: 时延采样点)
        # =================================================================
        ax1 = plt.subplot(2, 1, 1)
        searchMatrix = detectResult["searchMatrix"]

        freqLabels = [f"{gscn} ({offset / 1e6:.2f} MHz)" for gscn, offset in validGscnList]

        cax = ax1.imshow(searchMatrix, aspect='auto', cmap='plasma', origin='lower')
        fig.colorbar(cax, ax=ax1, label='Correlation Magnitude')
        ax1.set_yticks(np.arange(len(freqLabels)))
        ax1.set_yticklabels(freqLabels)
        ax1.set_title(f'2D PSS Search Heatmap (Peak at GSCN: {detectResult["gscn"]})')
        ax1.set_ylabel('GSCN Candidate (Frequency Offset)')
        ax1.set_xlabel('Timing Offset (Samples)')

        # 标出最大峰值位置
        bestFreqIdx = [item[0] for item in validGscnList].index(detectResult["gscn"])
        ax1.plot(detectResult["timingOffset"], bestFreqIdx, 'ro', markersize=10, fillstyle='none', markeredgewidth=2)

        # =================================================================
        # 2. 1D 相关峰剖面 (最优频点处的互相关曲线 + 多峰标注)
        # =================================================================
        ax2 = plt.subplot(2, 1, 2)
        corrArray = detectResult["corrArray"]
        ax2.plot(corrArray, color='b', linewidth=1)

        # --- 标注主峰 ---
        peakX = detectResult["timingOffset"]
        peakY = detectResult["peakValue"]
        ax2.plot(peakX, peakY, 'r^', markersize=10)
        ax2.annotate(f'Main Peak\nOffset: {peakX}\nN_ID_2: {detectResult["nId2"]}',
                     xy=(peakX, peakY), xytext=(peakX + rxLength * 0.05, peakY * 0.9),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))

        # --- 标注所有检测到的显著峰 ---
        peakAnalysis = detectResult.get("peakAnalysis")
        if peakAnalysis and peakAnalysis.get("detectedPeaks"):
            for i, peak in enumerate(peakAnalysis["detectedPeaks"]):
                if peak["index"] == peakX:
                    continue  # 主峰已标注，跳过
                ax2.plot(peak["index"], peak["value"], 'g^', markersize=8)
                ax2.annotate(f'Peak {i}\nOffset: {peak["index"]}\nVal: {peak["value"]:.1f}',
                             xy=(peak["index"], peak["value"]),
                             xytext=(peak["index"] + rxLength * 0.03, peak["value"] * 0.85),
                             fontsize=7,
                             arrowprops=dict(facecolor='green', shrink=0.05, width=1, headwidth=4))

        # --- 标注 CP 回波对 ---
        if peakAnalysis and peakAnalysis.get("cpEchoPair"):
            cpPair = peakAnalysis["cpEchoPair"]
            mainIdx = cpPair["mainPeakIndex"]
            echoIdx = cpPair["cpEchoPeakIndex"]

            # 绘制主峰和回波峰之间的连线
            ax2.plot([echoIdx, mainIdx], [cpPair["cpEchoPeakValue"], cpPair["mainPeakValue"]],
                     'r--', linewidth=1.5, alpha=0.7)

            # 在连线中点标注 CP 长度信息
            midX = (mainIdx + echoIdx) / 2
            midY = (cpPair["mainPeakValue"] + cpPair["cpEchoPeakValue"]) / 2
            ax2.annotate(f'CP Echo\nΔ={cpPair["measuredCpLength"]} samples\n(expected: {cpPair["expectedCpLength"]})',
                         xy=(midX, midY), xytext=(midX, midY * 1.3),
                         fontsize=8, color='red',
                         ha='center',
                         arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=4))

        ax2.set_title('1D PSS Cross-Correlation Profile at Best Frequency Offset')
        ax2.set_ylabel('Magnitude')
        ax2.set_xlabel('Timing Offset (Samples)')
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

        # =================================================================
        # 保存图片到本地
        # =================================================================
        imgPath = os.path.join(VisualProbes.OutputDir, 'pss_search_heatmap.png')
        fig.savefig(imgPath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logging.info(f"热力图已保存: {os.path.abspath(imgPath)}")

        # =================================================================
        # 保存盲搜结果到 JSON
        # =================================================================
        VisualProbes._saveResult(detectResult, validGscnList)

    @staticmethod
    def _saveResult(detectResult: dict, validGscnList: list):
        """
        将 PSS 盲搜结果保存为 JSON 文件

        输出文件: output/pss_search_result.json
        """
        VisualProbes._ensureOutputDir()

        gscn = detectResult["gscn"]
        freqOffset = detectResult["freqOffset"]
        peakAnalysis = detectResult.get("peakAnalysis")

        resultData = {
            "检测结果": {
                "GSCN": gscn,
                "SSREF绝对频率Hz": freqOffset,
                "SSREF绝对频率MHz": round(freqOffset / 1e6, 4),
                "扇区ID_N_ID_2": detectResult["nId2"],
                "定时偏移采样点": detectResult["timingOffset"],
                "相关峰值幅度": round(float(detectResult["peakValue"]), 6)
            },
            "多峰分析": VisualProbes._serializePeakAnalysis(peakAnalysis),
            "搜索参数": {
                "候选GSCN数量": len(validGscnList),
                "GSCN范围": f"{validGscnList[0][0]} ~ {validGscnList[-1][0]}"
            }
        }

        jsonPath = os.path.join(VisualProbes.OutputDir, 'pss_search_result.json')
        with open(jsonPath, 'w', encoding='utf-8') as f:
            json.dump(resultData, f, ensure_ascii=False, indent=4)

        logging.info(f"盲搜结果已保存: {os.path.abspath(jsonPath)}")
        logging.info(f"  GSCN={gscn}, SSREF={freqOffset/1e6:.4f} MHz, "
                     f"N_ID_2={detectResult['nId2']}, "
                     f"TimingOffset={detectResult['timingOffset']}, "
                     f"PeakValue={detectResult['peakValue']:.6f}")

    @staticmethod
    def _serializePeakAnalysis(peakAnalysis: dict) -> dict:
        """序列化多峰分析结果为 JSON 可存储格式"""
        if not peakAnalysis:
            return None

        result = {
            "显著峰数量": peakAnalysis.get("peakCount", 0),
            "显著峰列表": peakAnalysis.get("detectedPeaks", []),
            "CP回波检测": None
        }

        cpPair = peakAnalysis.get("cpEchoPair")
        if cpPair:
            result["CP回波检测"] = {
                "主峰位置": cpPair["mainPeakIndex"],
                "主峰幅度": round(cpPair["mainPeakValue"], 6),
                "回波峰位置": cpPair["cpEchoPeakIndex"],
                "回波峰幅度": round(cpPair["cpEchoPeakValue"], 6),
                "测量CP长度": cpPair["measuredCpLength"],
                "预期CP长度": cpPair["expectedCpLength"],
                "CP长度误差": cpPair["cpLengthError"]
            }

        return result
