import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class PssVisualizer:
    OutputDir = os.path.join(os.path.dirname(__file__), '..', '..', 'output')
    TimeScanSamples = 15000

    @classmethod
    def _ensureOutputDir(cls):
        os.makedirs(cls.OutputDir, exist_ok=True)

    @staticmethod
    def _getNid2ResultMap(detectResult: dict) -> dict:
        resultMap = {}
        for item in detectResult.get("nId2BestResults", []):
            resultMap[int(item["nId2"])] = item
        return resultMap

    @staticmethod
    def plotPssSearchValidation(
        detectResult: dict,
        validGscnList: list,
        rxLength: int,
        outputPrefix: str = "",
    ):
        del validGscnList
        del rxLength

        PssVisualizer._ensureOutputDir()

        nid2Map = PssVisualizer._getNid2ResultMap(detectResult)
        for nid2 in [0, 1, 2]:
            if nid2 not in nid2Map:
                raise ValueError(f"Missing nId2 result for nId2={nid2}")

        scanSamples = PssVisualizer.TimeScanSamples
        scoreMode = str(detectResult.get("scoreMode", "raw")).lower()
        yLabel = "Normalized Correlation" if scoreMode == "ncc" else "Correlation Magnitude"
        fig = plt.figure(figsize=(18, 10))
        overlayPeak = {
            "nId2": 0,
            "sampleIndex": 0,
            "peakValue": -1.0,
        }
        colors = {0: "tab:blue", 1: "tab:orange", 2: "tab:green"}

        for plotIdx, nid2 in enumerate([0, 1, 2], start=1):
            result = nid2Map[nid2]
            corrArray = np.asarray(result["corrArray"])
            scanEnd = min(scanSamples, len(corrArray))
            x = np.arange(scanEnd)
            y = corrArray[:scanEnd]

            peakIdxLocal = int(np.argmax(y))
            peakValLocal = float(y[peakIdxLocal])
            if peakValLocal > overlayPeak["peakValue"]:
                overlayPeak = {
                    "nId2": nid2,
                    "sampleIndex": peakIdxLocal,
                    "peakValue": peakValLocal,
                }

            ax = plt.subplot(2, 2, plotIdx)
            ax.plot(x, y, color=colors[nid2], linewidth=1.0)
            ax.plot(peakIdxLocal, peakValLocal, "r^", markersize=8)
            ax.set_title(
                f"N_ID_2={nid2} | bestFreq={result['freqOffset']:.1f} Hz | "
                f"bestPeak={result['peakValue']:.3f}"
            )
            ax.set_xlabel("Timing Offset (samples)")
            ax.set_ylabel(yLabel)
            ax.grid(True, linestyle="--", alpha=0.5)

        axOverlay = plt.subplot(2, 2, 4)
        for nid2 in [0, 1, 2]:
            result = nid2Map[nid2]
            corrArray = np.asarray(result["corrArray"])
            scanEnd = min(scanSamples, len(corrArray))
            x = np.arange(scanEnd)
            y = corrArray[:scanEnd]
            axOverlay.plot(x, y, color=colors[nid2], linewidth=1.0, label=f"N_ID_2={nid2}")

        axOverlay.plot(overlayPeak["sampleIndex"], overlayPeak["peakValue"], "r^", markersize=10)
        axOverlay.annotate(
            f"Global Peak\nN_ID_2={overlayPeak['nId2']}\n"
            f"idx={overlayPeak['sampleIndex']}\n"
            f"val={overlayPeak['peakValue']:.3f}",
            xy=(overlayPeak["sampleIndex"], overlayPeak["peakValue"]),
            xytext=(
                overlayPeak["sampleIndex"] + scanSamples * 0.05,
                overlayPeak["peakValue"] * 0.9,
            ),
            arrowprops=dict(facecolor="red", shrink=0.05, width=1.2, headwidth=6),
        )
        axOverlay.set_title(f"Overlay (N_ID_2=0/1/2, first {scanSamples} samples)")
        axOverlay.set_xlabel("Timing Offset (samples)")
        axOverlay.set_ylabel(yLabel)
        axOverlay.grid(True, linestyle="--", alpha=0.5)
        axOverlay.legend()

        plt.tight_layout()

        prefix = f"{outputPrefix}_" if outputPrefix else ""
        imgPath = os.path.join(PssVisualizer.OutputDir, f"{prefix}pss_nid2_time_scan.png")
        fig.savefig(imgPath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"N_ID_2 time-scan figure saved: {os.path.abspath(imgPath)}")

        PssVisualizer._plotPssFreqOffsetEstimation(detectResult, outputPrefix=outputPrefix)
        PssVisualizer._plotCpFreqOffsetEstimation(detectResult, outputPrefix=outputPrefix)

        PssVisualizer._saveResult(
            detectResult=detectResult,
            overlayPeak=overlayPeak,
            scanSamples=scanSamples,
            outputPrefix=outputPrefix,
        )

    @staticmethod
    def _getPssLsqEstimation(detectResult: dict) -> dict | None:
        estimation = detectResult.get("pssFreqOffsetEstimation")
        if estimation is not None:
            return estimation
        selected = detectResult.get("freqOffsetEstimation")
        if isinstance(selected, dict) and str(selected.get("method", "")) == "pss_phase_lsq":
            return selected
        return None

    @staticmethod
    def _getCpEstimation(detectResult: dict) -> dict | None:
        estimation = detectResult.get("cpFreqOffsetEstimation")
        if estimation is not None:
            return estimation
        selected = detectResult.get("freqOffsetEstimation")
        if isinstance(selected, dict) and str(selected.get("method", "")) == "cp_phase":
            return selected
        return None

    @staticmethod
    def _plotPssFreqOffsetEstimation(detectResult: dict, outputPrefix: str = ""):
        estimation = PssVisualizer._getPssLsqEstimation(detectResult)
        if estimation is None:
            return

        sampleIndex = np.asarray(estimation.get("sampleIndex"))
        phaseSamples = np.asarray(estimation.get("phaseSamplesRad"))
        fittedPhase = np.asarray(estimation.get("fittedPhaseRad"))
        residualPhase = np.asarray(estimation.get("residualPhaseRad"))
        if len(sampleIndex) == 0:
            return

        fig = plt.figure(figsize=(14, 8))

        axFit = plt.subplot(2, 1, 1)
        axFit.scatter(sampleIndex, phaseSamples, s=8, alpha=0.55, color="tab:blue", label="Phase Samples")
        axFit.plot(sampleIndex, fittedPhase, color="tab:red", linewidth=2.0, label="LS Fitted Line")
        axFit.set_title(
            "PSS-assisted CFO LS Fit | "
            f"R^2={float(estimation.get('rSquared', 0.0)):.6f} | "
            f"base={float(estimation.get('baseFreqHz', 0.0)):.3f} Hz | "
            f"residual={float(estimation.get('residualFreqHz', 0.0)):.3f} Hz | "
            f"refined={float(estimation.get('refinedFreqHz', 0.0)):.3f} Hz"
        )
        axFit.set_xlabel("Sample Index (within PSS symbol)")
        axFit.set_ylabel("Unwrapped Phase (rad)")
        axFit.grid(True, linestyle="--", alpha=0.5)
        axFit.legend()

        axResidual = plt.subplot(2, 1, 2)
        axResidual.scatter(sampleIndex, residualPhase, s=8, alpha=0.65, color="tab:purple")
        axResidual.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axResidual.set_title("LS Residual Scatter")
        axResidual.set_xlabel("Sample Index (within PSS symbol)")
        axResidual.set_ylabel("Residual Phase (rad)")
        axResidual.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        prefix = f"{outputPrefix}_" if outputPrefix else ""
        imgPath = os.path.join(PssVisualizer.OutputDir, f"{prefix}pss_freq_offset_lsq.png")
        fig.savefig(imgPath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"PSS CFO LS fit figure saved: {os.path.abspath(imgPath)}")

    @staticmethod
    def _plotCpFreqOffsetEstimation(detectResult: dict, outputPrefix: str = ""):
        estimation = PssVisualizer._getCpEstimation(detectResult)
        if estimation is None:
            return

        windowIndex = np.asarray(estimation.get("windowIndex"))
        residualByWindow = np.asarray(estimation.get("residualFreqByWindowHz"))
        coherence = np.asarray(estimation.get("coherence"))
        pointIndex = np.asarray(estimation.get("pointIndex"))
        pointPhase = np.asarray(estimation.get("pointPhaseRad"))
        pointResidual = np.asarray(estimation.get("pointResidualHz"))
        pointWindow = np.asarray(estimation.get("pointWindowIndex"))
        pointWeight = np.asarray(estimation.get("pointWeight"))
        if len(windowIndex) == 0 or len(residualByWindow) == 0:
            return

        fig = plt.figure(figsize=(14, 11))

        axPhase = plt.subplot(3, 1, 1)
        if len(pointIndex) > 0 and len(pointPhase) == len(pointIndex):
            phaseSize = np.clip((pointWeight / (np.max(pointWeight) + 1e-12)) * 16.0, 3.0, 16.0)
            scPhase = axPhase.scatter(
                pointIndex,
                pointPhase,
                s=phaseSize,
                c=pointWindow if len(pointWindow) == len(pointIndex) else "tab:blue",
                cmap="tab10",
                alpha=0.7,
            )
            if len(pointWindow) == len(pointIndex):
                cbar = fig.colorbar(scPhase, ax=axPhase, pad=0.01)
                cbar.set_label("Window Index")
        axPhase.set_title("CP Point-wise Phase Difference Scatter")
        axPhase.set_xlabel("Point Index (across CP windows)")
        axPhase.set_ylabel("Phase Difference (rad)")
        axPhase.grid(True, linestyle="--", alpha=0.5)

        axFreq = plt.subplot(3, 1, 2)
        if len(pointIndex) > 0 and len(pointResidual) == len(pointIndex):
            freqSize = np.clip((pointWeight / (np.max(pointWeight) + 1e-12)) * 16.0, 3.0, 16.0)
            scFreq = axFreq.scatter(
                pointIndex,
                pointResidual,
                s=freqSize,
                c=pointWindow if len(pointWindow) == len(pointIndex) else "tab:blue",
                cmap="tab10",
                alpha=0.7,
            )
            if len(pointWindow) == len(pointIndex):
                cbar = fig.colorbar(scFreq, ax=axFreq, pad=0.01)
                cbar.set_label("Window Index")
        axFreq.plot(windowIndex * max(1, int(estimation.get("cpLength", 0))), residualByWindow, "k--", linewidth=1.2, label="Window CFO")
        axFreq.axhline(
            float(estimation.get("residualFreqHz", 0.0)),
            color="tab:red",
            linestyle="--",
            linewidth=1.5,
            label="Weighted Mean",
        )
        axFreq.set_title(
            "CP-based CFO Estimation | "
            f"base={float(estimation.get('baseFreqHz', 0.0)):.3f} Hz | "
            f"residual={float(estimation.get('residualFreqHz', 0.0)):.3f} Hz | "
            f"refined={float(estimation.get('refinedFreqHz', 0.0)):.3f} Hz | "
            f"std={float(estimation.get('residualStdHz', 0.0)):.3f} Hz"
        )
        axFreq.set_xlabel("Point Index (across CP windows)")
        axFreq.set_ylabel("Residual CFO by point (Hz)")
        axFreq.grid(True, linestyle="--", alpha=0.5)
        axFreq.legend()

        axCoh = plt.subplot(3, 1, 3)
        axCoh.plot(windowIndex, coherence, "s-", color="tab:green", linewidth=1.5, markersize=5)
        axCoh.set_ylim(0.0, 1.05)
        axCoh.set_title("CP Correlation Coherence")
        axCoh.set_xlabel("Window Index (SSB symbol order)")
        axCoh.set_ylabel("Coherence [0, 1]")
        axCoh.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        prefix = f"{outputPrefix}_" if outputPrefix else ""
        imgPath = os.path.join(PssVisualizer.OutputDir, f"{prefix}pss_freq_offset_cp.png")
        fig.savefig(imgPath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"CP CFO figure saved: {os.path.abspath(imgPath)}")

    @staticmethod
    def _summarizeFreqOffsetEstimation(estimation: dict) -> dict:
        method = str(estimation.get("method", "unknown"))
        summary = {
            "method": method,
            "baseFreqHz": float(estimation.get("baseFreqHz", 0.0)),
            "residualFreqHz": float(estimation.get("residualFreqHz", 0.0)),
            "refinedFreqHz": float(estimation.get("refinedFreqHz", 0.0)),
        }
        if method == "pss_phase_lsq":
            summary.update({
                "usefulOnly": bool(estimation.get("usefulOnly", False)),
                "slopeRadPerSample": float(estimation.get("slopeRadPerSample", 0.0)),
                "interceptRad": float(estimation.get("interceptRad", 0.0)),
                "rSquared": float(estimation.get("rSquared", 0.0)),
                "sampleStart": int(estimation.get("sampleStart", 0)),
                "sampleCount": int(estimation.get("sampleCount", 0)),
                "validSampleCount": int(estimation.get("validSampleCount", 0)),
            })
        elif method == "cp_phase":
            summary.update({
                "fftSize": int(estimation.get("fftSize", 0)),
                "cpLength": int(estimation.get("cpLength", 0)),
                "symbolCountUsed": int(estimation.get("symbolCountUsed", 0)),
                "residualStdHz": float(estimation.get("residualStdHz", 0.0)),
                "pointCount": int(estimation.get("pointCount", 0)),
            })
        return summary

    @staticmethod
    def _saveResult(detectResult: dict, overlayPeak: dict, scanSamples: int, outputPrefix: str = ""):
        PssVisualizer._ensureOutputDir()

        nId2ResultList = []
        for item in detectResult.get("nId2BestResults", []):
            nId2ResultList.append({
                "nId2": int(item["nId2"]),
                "gscn": int(item["gscn"]),
                "freqOffsetHz": float(item["freqOffset"]),
                "timingOffset": int(item["timingOffset"]),
                "peakValue": float(item["peakValue"]),
            })

        resultData = {
            "scoreMode": str(detectResult.get("scoreMode", "raw")),
            "finalBest": {
                "nId2": int(detectResult["nId2"]),
                "gscn": int(detectResult["gscn"]),
                "freqOffsetHz": float(detectResult["freqOffset"]),
                "freqOffsetParabolicHz": float(detectResult.get("freqOffsetParabolic", detectResult["freqOffset"])),
                "timingOffset": int(detectResult["timingOffset"]),
                "peakValue": float(detectResult["peakValue"]),
            },
            "coarseBest": detectResult.get("coarseBest"),
            "fineSearchConfig": detectResult.get("fineSearch"),
            "nId2BestResults": nId2ResultList,
            "overlayPeakInFirstSamples": {
                "sampleCount": int(scanSamples),
                "nId2": int(overlayPeak["nId2"]),
                "sampleIndex": int(overlayPeak["sampleIndex"]),
                "peakValue": float(overlayPeak["peakValue"]),
            },
        }
        parabolicRefinement = detectResult.get("parabolicRefinement")
        if parabolicRefinement is not None:
            resultData["parabolicRefinement"] = parabolicRefinement
        adaptiveRefinement = detectResult.get("adaptiveRefinement")
        if adaptiveRefinement is not None:
            resultData["adaptiveRefinement"] = adaptiveRefinement
        selectedEstimation = detectResult.get("freqOffsetEstimation")
        if selectedEstimation is not None:
            resultData["selectedFreqOffsetEstimation"] = PssVisualizer._summarizeFreqOffsetEstimation(selectedEstimation)
            resultData["freqOffsetEstimation"] = resultData["selectedFreqOffsetEstimation"]

        pssEstimation = detectResult.get("pssFreqOffsetEstimation")
        if pssEstimation is not None:
            resultData["pssFreqOffsetEstimation"] = PssVisualizer._summarizeFreqOffsetEstimation(pssEstimation)

        cpEstimation = detectResult.get("cpFreqOffsetEstimation")
        if cpEstimation is not None:
            resultData["cpFreqOffsetEstimation"] = PssVisualizer._summarizeFreqOffsetEstimation(cpEstimation)

        prefix = f"{outputPrefix}_" if outputPrefix else ""
        jsonPath = os.path.join(PssVisualizer.OutputDir, f"{prefix}pss_search_result.json")
        with open(jsonPath, "w", encoding="utf-8") as f:
            json.dump(resultData, f, ensure_ascii=False, indent=4)

        logger.info(f"PSS result JSON saved: {os.path.abspath(jsonPath)}")
