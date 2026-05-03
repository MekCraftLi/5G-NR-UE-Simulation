import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager
from common.ofdm import OfdmDemodulator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PbchRe:
    k: int
    l: int


class PbchDecoder:
    """PBCH DM-RS based equalizer, QPSK demodulator, and EVM estimator.

    This class stops at PBCH symbol demodulation. BCH polar decoding/MIB parsing is
    intentionally not included here.
    """

    SsbSubcarriers = 240
    OutputDir = os.path.join(os.path.dirname(__file__), "..", "..", "output")

    def __init__(self, config: SsbConfig, ssbIndexCandidates: range | list[int] | tuple[int, ...] | None = None):
        self.config = config
        self.fftSize = int(config.FftSize)
        self.sampleRate = float(config.SampleRate)
        self.cpManager = CpManager(config)
        self.ofdm = OfdmDemodulator(config)
        self.ssbSubcarrierOffset = int(getattr(config, "SsbSubcarrierOffset", 0))
        self.ssbIndexCandidates = list(range(8)) if ssbIndexCandidates is None else [int(x) for x in ssbIndexCandidates]

    @staticmethod
    def _goldSequence(cInit: int, length: int) -> np.ndarray:
        nc = 1600
        total = nc + int(length) + 31
        x1 = np.zeros(total, dtype=np.uint8)
        x2 = np.zeros(total, dtype=np.uint8)
        x1[0] = 1
        for i in range(31):
            x2[i] = (int(cInit) >> i) & 1
        for n in range(total - 31):
            x1[n + 31] = (x1[n + 3] + x1[n]) & 1
            x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
        return ((x1[nc:nc + length] + x2[nc:nc + length]) & 1).astype(np.uint8)

    @classmethod
    def generatePbchDmrs(cls, nIdCell: int, iSsbBar: int) -> np.ndarray:
        """Generate r(0)..r(143), TS 38.211 7.4.1.4.1."""
        nIdCell = int(nIdCell)
        iSsbBar = int(iSsbBar)
        cInit = (2 ** 11) * (iSsbBar + 1) * (nIdCell // 4 + 1) + (2 ** 6) * (iSsbBar + 1) + (nIdCell % 4)
        c = cls._goldSequence(cInit, 2 * 144)
        r = ((1.0 - 2.0 * c[0::2].astype(np.float32)) + 1j * (1.0 - 2.0 * c[1::2].astype(np.float32))) / np.sqrt(2.0)
        return r.astype(np.complex64)

    @staticmethod
    def _dmrsReList(nIdCell: int) -> list[PbchRe]:
        v = int(nIdCell) % 4
        items: list[PbchRe] = []
        for l in (1, 3):
            for k in range(v, 240, 4):
                items.append(PbchRe(k=k, l=l))
        for k in list(range(v, 48, 4)) + list(range(192 + v, 240, 4)):
            items.append(PbchRe(k=k, l=2))
        return sorted(items, key=lambda item: (item.l, item.k))

    @classmethod
    def _pbchDataReList(cls, nIdCell: int) -> list[PbchRe]:
        dmrs = {(item.k, item.l) for item in cls._dmrsReList(nIdCell)}
        items: list[PbchRe] = []
        for l in (1, 3):
            for k in range(240):
                if (k, l) not in dmrs:
                    items.append(PbchRe(k=k, l=l))
        for k in list(range(0, 48)) + list(range(192, 240)):
            if (k, 2) not in dmrs:
                items.append(PbchRe(k=k, l=2))
        return sorted(items, key=lambda item: (item.l, item.k))

    def _cpLengths(self) -> list[int]:
        return [int(self.cpManager.getCpLength(i)) for i in range(4)]

    def _cpLengthProfiles(self) -> list[tuple[str, list[int]]]:
        normal = int(self.cpManager.normalCpLength)
        long = int(self.cpManager.longCpLength)
        profiles: list[tuple[str, list[int]]] = [
            ("all_normal", [normal, normal, normal, normal]),
            ("slot_head", [long, normal, normal, normal]),
        ]
        return profiles

    def _compensate(self, rxSignal: np.ndarray, freqCompHz: float) -> np.ndarray:
        n = np.arange(len(rxSignal), dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * n / self.sampleRate).astype(np.complex64)
        return (np.asarray(rxSignal, dtype=np.complex64) * phase).astype(np.complex64)

    def _extractSsbGrid(self, rxSignal: np.ndarray, ssbStart: int, freqCompHz: float, cpLengths: list[int] | None = None) -> np.ndarray:
        compensated = self._compensate(rxSignal, freqCompHz)
        useCp = self._cpLengths() if cpLengths is None else [int(v) for v in cpLengths]
        symbols = self.ofdm.extractSsbSymbols(compensated, int(ssbStart), cpLengths=useCp)
        start = self.fftSize // 2 - self.SsbSubcarriers // 2 + self.ssbSubcarrierOffset
        grid = np.zeros((self.SsbSubcarriers, 4), dtype=np.complex64)
        for l, symbol in enumerate(symbols):
            grid[:, l] = np.asarray(symbol[start:start + self.SsbSubcarriers], dtype=np.complex64)
        return grid

    @staticmethod
    def _extract(grid: np.ndarray, items: list[PbchRe]) -> np.ndarray:
        return np.asarray([grid[item.k, item.l] for item in items], dtype=np.complex64)

    @staticmethod
    def _interpolateChannel(dmrsItems: list[PbchRe], hDmrs: np.ndarray, dataItems: list[PbchRe]) -> np.ndarray:
        hBySymbol: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for l in (1, 2, 3):
            idx = [i for i, item in enumerate(dmrsItems) if item.l == l]
            k = np.asarray([dmrsItems[i].k for i in idx], dtype=np.float64)
            h = np.asarray([hDmrs[i] for i in idx], dtype=np.complex64)
            order = np.argsort(k)
            hBySymbol[l] = (k[order], h[order])

        hData = np.zeros(len(dataItems), dtype=np.complex64)
        for i, item in enumerate(dataItems):
            kRef, hRef = hBySymbol[item.l]
            magRef = np.abs(hRef).astype(np.float64)
            phaseRef = np.unwrap(np.angle(hRef)).astype(np.float64)
            mag = float(np.interp(float(item.k), kRef, magRef))
            phase = float(np.interp(float(item.k), kRef, phaseRef))
            hData[i] = np.complex64(mag * np.exp(1j * phase))
        return hData

    @staticmethod
    def _nearestQpsk(symbols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        realBits = (np.real(symbols) < 0).astype(np.uint8)
        imagBits = (np.imag(symbols) < 0).astype(np.uint8)
        ref = ((1.0 - 2.0 * realBits.astype(np.float32)) + 1j * (1.0 - 2.0 * imagBits.astype(np.float32))) / np.sqrt(2.0)
        bits = np.empty(2 * len(symbols), dtype=np.uint8)
        bits[0::2] = realBits
        bits[1::2] = imagBits
        return ref.astype(np.complex64), bits

    @staticmethod
    def _evmPercent(symbols: np.ndarray, reference: np.ndarray) -> float:
        num = np.mean(np.abs(symbols - reference) ** 2)
        den = np.mean(np.abs(reference) ** 2)
        return float(100.0 * np.sqrt(num / max(den, 1e-12)))

    def _evaluateCandidate(
        self,
        rxSignal: np.ndarray,
        nIdCell: int,
        ssbStart: int,
        freqCompHz: float,
        iSsbBar: int,
        cpProfileName: str,
        cpLengths: list[int],
    ) -> dict:
        grid = self._extractSsbGrid(rxSignal, ssbStart=ssbStart, freqCompHz=freqCompHz, cpLengths=cpLengths)
        dmrsItems = self._dmrsReList(nIdCell)
        dataItems = self._pbchDataReList(nIdCell)
        dmrsRx = self._extract(grid, dmrsItems)
        dmrsRef = self.generatePbchDmrs(nIdCell, iSsbBar)
        hDmrs = dmrsRx / dmrsRef

        dataRx = self._extract(grid, dataItems)
        hData = self._interpolateChannel(dmrsItems, hDmrs, dataItems)
        dataEq = dataRx / np.where(np.abs(hData) > 1e-9, hData, 1.0 + 0j)

        # Remove one residual common gain/phase so constellation EVM measures scatter, not arbitrary scaling.
        qpskRef0, _ = self._nearestQpsk(dataEq)
        gain = np.vdot(qpskRef0, dataEq) / max(float(np.vdot(qpskRef0, qpskRef0).real), 1e-12)
        dataEqNorm = dataEq / (gain if abs(gain) > 1e-12 else 1.0 + 0j)
        qpskRef, bits = self._nearestQpsk(dataEqNorm)
        pbchEvm = self._evmPercent(dataEqNorm, qpskRef)

        dmrsPower = float(np.mean(np.abs(dmrsRx) ** 2))
        dataPower = float(np.mean(np.abs(dataRx) ** 2))
        channelMean = complex(np.mean(hDmrs))
        channelStd = float(np.std(hDmrs))
        return {
            "nIdCell": int(nIdCell),
            "iSsbBar": int(iSsbBar),
            "ssbStart": int(ssbStart),
            "freqCompHz": float(freqCompHz),
            "cpProfile": str(cpProfileName),
            "grid": grid,
            "dmrsItems": dmrsItems,
            "dataItems": dataItems,
            "dmrsRx": dmrsRx,
            "dmrsRef": dmrsRef,
            "hDmrs": hDmrs,
            "dataRx": dataRx,
            "hData": hData,
            "pbchEq": dataEqNorm.astype(np.complex64),
            "pbchHardRef": qpskRef.astype(np.complex64),
            "hardBits": bits,
            "evmPercent": float(pbchEvm),
            "dmrsPower": dmrsPower,
            "dataPower": dataPower,
            "channelMean": channelMean,
            "channelStd": channelStd,
        }

    @staticmethod
    def _candidateStarts(sssResult: dict, pssResult: dict, cpLengths: list[int], fftSize: int) -> list[int]:
        starts: list[int] = []
        if "bestSymbolStart" in sssResult:
            sssOffset = 2 * fftSize + cpLengths[0] + cpLengths[1]
            starts.append(int(sssResult["bestSymbolStart"]) - int(sssOffset))
        if "verifiedSymbolStart" in sssResult:
            sssOffset = 2 * fftSize + cpLengths[0] + cpLengths[1]
            starts.append(int(sssResult["verifiedSymbolStart"]) - int(sssOffset))
        if "ssbStart" in pssResult:
            starts.append(int(pssResult["ssbStart"]))
        if "timingOffset" in pssResult:
            starts.append(int(pssResult["timingOffset"]))
        unique = []
        for value in starts:
            for delta in (-2, -1, 0, 1, 2):
                candidate = int(value + delta)
                if candidate >= 0 and candidate not in unique:
                    unique.append(candidate)
        return unique

    def decodePbch(
        self,
        rxSignal: np.ndarray,
        sssResult: dict,
        pssResult: dict,
        freqCompHz: float | None = None,
        residualFreqSearchHz: float = 500.0,
        residualFreqStepHz: float = 50.0,
    ) -> dict:
        nIdCell = int(sssResult["nIdCell"])
        if freqCompHz is None:
            freqCompHz = float(sssResult.get("verifiedFreqCompHz", sssResult.get("freqCompHz", pssResult.get("freqOffsetParabolic", 0.0))))

        cpLengths = self._cpLengths()
        starts = self._candidateStarts(sssResult, pssResult, cpLengths, self.fftSize)
        if not starts:
            raise ValueError("No valid PBCH SSB-start candidates")

        if residualFreqStepHz <= 0:
            freqGrid = np.asarray([float(freqCompHz)], dtype=np.float64)
        else:
            residual = np.arange(-abs(float(residualFreqSearchHz)), abs(float(residualFreqSearchHz)) + residualFreqStepHz / 2, abs(float(residualFreqStepHz)))
            freqGrid = float(freqCompHz) + residual

        candidates = []
        best = None
        cpProfiles = self._cpLengthProfiles()
        for ssbStart in starts:
            for freqHz in freqGrid:
                for iSsbBar in self.ssbIndexCandidates:
                    for cpProfileName, cpLengths in cpProfiles:
                        try:
                            item = self._evaluateCandidate(
                                rxSignal,
                                nIdCell=nIdCell,
                                ssbStart=ssbStart,
                                freqCompHz=float(freqHz),
                                iSsbBar=iSsbBar,
                                cpProfileName=cpProfileName,
                                cpLengths=cpLengths,
                            )
                        except Exception:
                            continue
                        summary = {
                            "iSsbBar": int(iSsbBar),
                            "ssbStart": int(ssbStart),
                            "freqCompHz": float(freqHz),
                            "cpProfile": str(cpProfileName),
                            "evmPercent": float(item["evmPercent"]),
                            "dmrsPower": float(item["dmrsPower"]),
                            "channelStd": float(item["channelStd"]),
                        }
                        candidates.append(summary)
                        if best is None or item["evmPercent"] < best["evmPercent"]:
                            best = item

        if best is not None and residualFreqStepHz > 1.0:
            fineStepHz = max(1.0, float(residualFreqStepHz) / 10.0)
            fineSearchHz = max(float(residualFreqStepHz), float(residualFreqStepHz) * 3.0)
            fineResidual = np.arange(
                -abs(fineSearchHz),
                abs(fineSearchHz) + fineStepHz / 2.0,
                abs(fineStepHz),
            )
            fineGrid = float(best["freqCompHz"]) + fineResidual
            for freqHz in fineGrid:
                try:
                    item = self._evaluateCandidate(
                        rxSignal,
                        nIdCell=nIdCell,
                        ssbStart=int(best["ssbStart"]),
                        freqCompHz=float(freqHz),
                        iSsbBar=int(best["iSsbBar"]),
                        cpProfileName=str(best.get("cpProfile", "slot_head")),
                        cpLengths=[int(v) for v in dict(cpProfiles).get(str(best.get("cpProfile", "slot_head")), self._cpLengths())],
                    )
                except Exception:
                    continue
                summary = {
                    "iSsbBar": int(item["iSsbBar"]),
                    "ssbStart": int(item["ssbStart"]),
                    "freqCompHz": float(item["freqCompHz"]),
                    "cpProfile": str(item.get("cpProfile", "slot_head")),
                    "evmPercent": float(item["evmPercent"]),
                    "dmrsPower": float(item["dmrsPower"]),
                    "channelStd": float(item["channelStd"]),
                }
                candidates.append(summary)
                if item["evmPercent"] < best["evmPercent"]:
                    best = item

        if best is not None:
            cpProfileName = str(best.get("cpProfile", "all_normal"))
            cpProfileMap = {name: vals for name, vals in cpProfiles}
            cpLengths = [int(v) for v in cpProfileMap.get(cpProfileName, self._cpLengths())]
            startCenter = int(best["ssbStart"])
            startGrid = np.arange(startCenter - 16, startCenter + 16 + 1, dtype=np.int32)
            step = max(1.0, float(residualFreqStepHz) / 10.0)
            freqCenter = float(best["freqCompHz"])
            freqGridLocal = np.arange(freqCenter - 40.0, freqCenter + 40.0 + step / 2.0, step, dtype=np.float64)
            for ssbStartLocal in startGrid:
                if ssbStartLocal < 0:
                    continue
                for freqHz in freqGridLocal:
                    try:
                        item = self._evaluateCandidate(
                            rxSignal,
                            nIdCell=nIdCell,
                            ssbStart=int(ssbStartLocal),
                            freqCompHz=float(freqHz),
                            iSsbBar=int(best["iSsbBar"]),
                            cpProfileName=cpProfileName,
                            cpLengths=cpLengths,
                        )
                    except Exception:
                        continue
                    summary = {
                        "iSsbBar": int(item["iSsbBar"]),
                        "ssbStart": int(item["ssbStart"]),
                        "freqCompHz": float(item["freqCompHz"]),
                        "cpProfile": str(item.get("cpProfile", cpProfileName)),
                        "evmPercent": float(item["evmPercent"]),
                        "dmrsPower": float(item["dmrsPower"]),
                        "channelStd": float(item["channelStd"]),
                    }
                    candidates.append(summary)
                    if item["evmPercent"] < best["evmPercent"]:
                        best = item

        if best is None:
            raise RuntimeError("PBCH candidate scan failed")

        bestSummary = {
            "method": "pbch_dmrs_equalize_qpsk_demod",
            "nIdCell": int(nIdCell),
            "iSsbBar": int(best["iSsbBar"]),
            "ssbStart": int(best["ssbStart"]),
            "freqCompHz": float(best["freqCompHz"]),
            "cpProfile": str(best.get("cpProfile", "slot_head")),
            "evmPercent": float(best["evmPercent"]),
            "evmPass10Percent": bool(float(best["evmPercent"]) < 10.0),
            "dmrsPower": float(best["dmrsPower"]),
            "dataPower": float(best["dataPower"]),
            "dmrsCount": int(len(best["dmrsItems"])),
            "pbchSymbolCount": int(len(best["dataItems"])),
            "hardBitCount": int(len(best["hardBits"])),
            "channelMeanReal": float(np.real(best["channelMean"])),
            "channelMeanImag": float(np.imag(best["channelMean"])),
            "channelStd": float(best["channelStd"]),
            "candidateCount": int(len(candidates)),
            "topCandidates": sorted(candidates, key=lambda x: x["evmPercent"])[:10],
            "pbchEq": best["pbchEq"],
            "pbchHardRef": best["pbchHardRef"],
            "hardBits": best["hardBits"],
            "dataRe": np.asarray([[item.k, item.l] for item in best["dataItems"]], dtype=np.int16),
            "dmrsRe": np.asarray([[item.k, item.l] for item in best["dmrsItems"]], dtype=np.int16),
        }
        logger.info(
            "PBCH demod done: N_ID_cell=%s, i_SSB_bar=%s, ssbStart=%s, freq=%.2f Hz, EVM=%.2f%%",
            bestSummary["nIdCell"],
            bestSummary["iSsbBar"],
            bestSummary["ssbStart"],
            bestSummary["freqCompHz"],
            bestSummary["evmPercent"],
        )
        return bestSummary

    @classmethod
    def saveArtifacts(cls, pbchResult: dict, outputPrefix: str = "") -> None:
        os.makedirs(cls.OutputDir, exist_ok=True)
        prefix = f"{outputPrefix}_" if outputPrefix else ""
        eq = np.asarray(pbchResult["pbchEq"], dtype=np.complex64)
        ref = np.asarray(pbchResult["pbchHardRef"], dtype=np.complex64)

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(np.real(eq), np.imag(eq), s=12, alpha=0.65, label="Equalized PBCH")
        q = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
        ax.scatter(np.real(q), np.imag(q), marker="x", s=120, c="red", linewidths=2, label="QPSK decisions")
        lim = max(1.2, float(np.max(np.abs(np.concatenate([np.real(eq), np.imag(eq)])))) * 1.1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(f"PBCH Equalized Constellation | EVM={pbchResult['evmPercent']:.2f}%")
        ax.legend(loc="best")
        fig.tight_layout()
        figPath = Path(cls.OutputDir) / f"{prefix}pbch_constellation.png"
        fig.savefig(figPath, dpi=160, bbox_inches="tight")
        plt.close(fig)

        npzPath = Path(cls.OutputDir) / f"{prefix}pbch_demod_symbols.npz"
        np.savez_compressed(
            npzPath,
            pbchEq=eq,
            pbchHardRef=ref,
            hardBits=np.asarray(pbchResult["hardBits"], dtype=np.uint8),
            dataRe=np.asarray(pbchResult["dataRe"], dtype=np.int16),
            dmrsRe=np.asarray(pbchResult["dmrsRe"], dtype=np.int16),
        )

        summary = {k: v for k, v in pbchResult.items() if k not in ("pbchEq", "pbchHardRef", "hardBits", "dataRe", "dmrsRe")}
        summary["savedFiles"] = {
            "constellation": str(figPath.resolve()),
            "symbols": str(npzPath.resolve()),
        }
        jsonPath = Path(cls.OutputDir) / f"{prefix}pbch_demod_result.json"
        jsonPath.write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
        logger.info("PBCH constellation saved: %s", figPath.resolve())
        logger.info("PBCH demod result saved: %s", jsonPath.resolve())
