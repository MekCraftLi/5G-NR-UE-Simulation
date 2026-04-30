import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class SssVisualizer:
    OutputDir = os.path.join(os.path.dirname(__file__), "..", "..", "output")

    @classmethod
    def _ensureOutputDir(cls):
        os.makedirs(cls.OutputDir, exist_ok=True)

    @staticmethod
    def plotAndSave(sssResult: dict, outputPrefix: str = ""):
        SssVisualizer._ensureOutputDir()

        offsets = np.asarray(sssResult.get("offsetGridSamples"))
        corrMatrix = np.asarray(sssResult.get("corrMatrix"))
        corrMax = np.asarray(sssResult.get("corrMaxByOffset"))
        bestNid1ByOffset = np.asarray(sssResult.get("bestNid1ByOffset"))
        if len(offsets) == 0 or corrMatrix.size == 0:
            logger.warning("Skip SSS plotting: empty correlation result")
            return

        bestOffset = int(sssResult["bestOffsetSamples"])
        bestNid1 = int(sssResult["nId1"])
        bestScore = float(sssResult["bestScore"])
        expectedOffset = int(sssResult.get("expectedSssOffsetSamples", 0))

        fig = plt.figure(figsize=(16, 11))

        axHeat = plt.subplot(2, 2, 1)
        im = axHeat.imshow(
            corrMatrix.T,
            aspect="auto",
            origin="lower",
            extent=[int(offsets[0]), int(offsets[-1]), 0, corrMatrix.shape[1] - 1],
            cmap="viridis",
        )
        axHeat.axvline(bestOffset, color="red", linestyle="--", linewidth=1.2)
        axHeat.axhline(bestNid1, color="white", linestyle="--", linewidth=1.0)
        axHeat.set_title("SSS Sliding Correlation Heatmap")
        axHeat.set_xlabel("Offset from PSS Timing (samples)")
        axHeat.set_ylabel("N_ID_1")
        plt.colorbar(im, ax=axHeat, pad=0.01, label="Normalized Correlation")

        axOffset = plt.subplot(2, 2, 2)
        axOffset.plot(offsets, corrMax, color="tab:blue", linewidth=1.2, label="Max corr per offset")
        axOffset.axvline(bestOffset, color="red", linestyle="--", linewidth=1.2, label="Best offset")
        axOffset.axvline(expectedOffset, color="tab:green", linestyle="--", linewidth=1.2, label="Expected ~2-symbol offset")
        axOffset.plot(bestOffset, bestScore, "r^", markersize=8)
        axOffset.set_title(
            f"Offset Scan Envelope | bestOffset={bestOffset}, bestNID1={bestNid1}, score={bestScore:.6f}"
        )
        axOffset.set_xlabel("Offset from PSS Timing (samples)")
        axOffset.set_ylabel("Normalized Correlation")
        axOffset.grid(True, linestyle="--", alpha=0.5)
        axOffset.legend()

        axNid1 = plt.subplot(2, 2, 3)
        axNid1.plot(offsets, bestNid1ByOffset, color="tab:orange", linewidth=1.0)
        axNid1.axvline(bestOffset, color="red", linestyle="--", linewidth=1.2)
        axNid1.plot(bestOffset, bestNid1, "r^", markersize=8)
        axNid1.set_title("Best N_ID_1 per Offset")
        axNid1.set_xlabel("Offset from PSS Timing (samples)")
        axNid1.set_ylabel("N_ID_1")
        axNid1.grid(True, linestyle="--", alpha=0.5)

        axNidProfile = plt.subplot(2, 2, 4)
        rowBest = int(np.argmin(np.abs(offsets - bestOffset)))
        profile = corrMatrix[rowBest, :]
        axNidProfile.plot(np.arange(len(profile)), profile, color="tab:purple", linewidth=1.2)
        axNidProfile.plot(bestNid1, profile[bestNid1], "r^", markersize=8)
        axNidProfile.set_title("N_ID_1 Correlation Profile at Best Offset")
        axNidProfile.set_xlabel("N_ID_1")
        axNidProfile.set_ylabel("Normalized Correlation")
        axNidProfile.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()

        prefix = f"{outputPrefix}_" if outputPrefix else ""
        figPath = os.path.join(SssVisualizer.OutputDir, f"{prefix}sss_sliding_search.png")
        fig.savefig(figPath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SSS sliding-search figure saved: {os.path.abspath(figPath)}")

        npzPath = os.path.join(SssVisualizer.OutputDir, f"{prefix}sss_sliding_correlation.npz")
        np.savez_compressed(
            npzPath,
            offsetGridSamples=offsets.astype(np.int32),
            bestNid1ByOffset=bestNid1ByOffset.astype(np.int32),
            corrMaxByOffset=corrMax.astype(np.float32),
            corrMatrix=corrMatrix.astype(np.float32),
        )
        logger.info(f"SSS full correlation matrix saved: {os.path.abspath(npzPath)}")

        summary = {
            "method": str(sssResult.get("method", "sss_sliding_symbol_correlation")),
            "nId2": int(sssResult["nId2"]),
            "nId1": int(sssResult["nId1"]),
            "nIdCell": int(sssResult["nIdCell"]),
            "timingBase": int(sssResult["timingBase"]),
            "freqCompHz": float(sssResult.get("freqCompHz", 0.0)),
            "searchStartSymbol": float(sssResult["searchStartSymbol"]),
            "searchEndSymbol": float(sssResult["searchEndSymbol"]),
            "searchStepSamples": int(sssResult["searchStepSamples"]),
            "symbolLengthSamples": int(sssResult["symbolLengthSamples"]),
            "expectedSssOffsetSamples": int(sssResult.get("expectedSssOffsetSamples", 0)),
            "bestOffsetSamples": int(sssResult["bestOffsetSamples"]),
            "bestSymbolStart": int(sssResult["bestSymbolStart"]),
            "bestScore": float(sssResult["bestScore"]),
            "windowCount": int(len(offsets)),
            "topCandidates": sssResult.get("topCandidates", []),
            "savedFiles": {
                "figure": os.path.abspath(figPath),
                "correlationNpz": os.path.abspath(npzPath),
            },
        }

        jsonPath = os.path.join(SssVisualizer.OutputDir, f"{prefix}sss_search_result.json")
        with open(jsonPath, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=4)
        logger.info(f"SSS result JSON saved: {os.path.abspath(jsonPath)}")
