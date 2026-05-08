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
    """240x4 本地 SS/PBCH 网格中的一个 PBCH/DM-RS 资源元素。"""

    k: int
    l: int


class PbchDecoder:
    """基于 PBCH DM-RS 的均衡器、QPSK 解调器和 EVM 估计器。

    本类停在 PBCH 符号解调层；BCH polar 解码和 MIB 解析不放在这里。

    规范锚点：
    - TS 38.211 Clause 7.4.3.1：SS/PBCH block 占 4 个 OFDM 符号和
      240 个连续子载波。PSS 位于 l=0，SSS 位于 l=2，PBCH data 位于
      l=1/3 全带以及 l=2 两侧边带。
    - TS 38.211 Clause 7.4.1.4：PBCH DM-RS 由 `N_ID_cell` 和
      `i_SSB_bar` 生成；DM-RS RE 由 `v = N_ID_cell mod 4` 决定频域
      平移，并且每 4 个子载波间隔映射。

    数据交接：
    `decodePbch(rxSignal, sssResult, pssResult, freqCompHz)` 接收：
    - 来自 `main.py` 的原始 IQ 采样 `rxSignal`；
    - 来自 SSS 的 `nIdCell`、`verifiedSymbolStart` 和
      `ssbSubcarrierOffset`；
    - 作为 SSB 起点后备锚点的 PSS 定时；
    - 单位为 Hz 的 `freqCompHz`，优先使用 `sssResult["verifiedFreqCompHz"]`。

    返回：
    - `pbchEq`：交给 BCH 软/硬解码的均衡 PBCH QPSK 符号；
    - `hardBits`：诊断用硬判决比特；
    - `nIdCell`, `iSsbBar`, `ssbStart`, `freqCompHz`, `cpProfile`;
    - 用于候选排序和任务交接记录的 DM-RS/EVM/噪声指标。
    """

    SsbSubcarriers = 240
    OutputDir = os.path.join(os.path.dirname(__file__), "..", "..", "output")

    def __init__(
        self,
        config: SsbConfig,
        ssbIndexCandidates: range | list[int] | tuple[int, ...] | None = None,
        cpProfileNames: list[str] | tuple[str, ...] | None = None,
    ):
        self.config = config
        self.fftSize = int(config.FftSize)
        self.sampleRate = float(config.SampleRate)
        self.cpManager = CpManager(config)
        self.ofdm = OfdmDemodulator(config)
        self.ssbSubcarrierOffset = int(getattr(config, "SsbSubcarrierOffset", 0))
        self.ssbIndexCandidates = list(range(8)) if ssbIndexCandidates is None else [int(x) for x in ssbIndexCandidates]
        self.cpProfileNames = None if cpProfileNames is None else [str(x) for x in cpProfileNames]

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
        """生成 PBCH DM-RS `r(0)..r(143)`。

        输入来自上游同步：
        - `nIdCell` = 3*N_ID_1 + N_ID_2，由 SSS/PSS 给出。
        - `iSsbBar` 会被扫描，因为它会影响 PBCH DM-RS 序列。

        规范依据：TS 38.211 Clause 7.4.1.4.1。
        """
        nIdCell = int(nIdCell)
        iSsbBar = int(iSsbBar)
        cInit = (2 ** 11) * (iSsbBar + 1) * (nIdCell // 4 + 1) + (2 ** 6) * (iSsbBar + 1) + (nIdCell % 4)
        c = cls._goldSequence(cInit, 2 * 144)
        r = ((1.0 - 2.0 * c[0::2].astype(np.float32)) + 1j * (1.0 - 2.0 * c[1::2].astype(np.float32))) / np.sqrt(2.0)
        return r.astype(np.complex64)

    @staticmethod
    def _dmrsReList(nIdCell: int) -> list[PbchRe]:
        """列出本地 240 子载波 SSB 网格中的 PBCH DM-RS RE。

        DM-RS 频域偏移为 `v = N_ID_cell mod 4`。符号 l=1 和 l=3 覆盖
        PBCH 的完整 240 子载波范围；l=2 只使用中央 SSS 之外的两侧
        PBCH 区域。
        """
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
        """列出从 PBCH 映射中去除 DM-RS 后的数据 RE。

        输出顺序就是 `pbchEq`、`pbchHardRef` 和 `hardBits` 的解调顺序；
        下游 BCH 按这个顺序消费符号。
        """
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
        if self.cpProfileNames is not None:
            allowed = set(self.cpProfileNames)
            profiles = [item for item in profiles if item[0] in allowed]
        return profiles

    def _compensate(self, rxSignal: np.ndarray, freqCompHz: float) -> np.ndarray:
        n = np.arange(len(rxSignal), dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * n / self.sampleRate).astype(np.complex64)
        return (np.asarray(rxSignal, dtype=np.complex64) * phase).astype(np.complex64)

    def _extractSsbGrid(self, rxSignal: np.ndarray, ssbStart: int, freqCompHz: float, cpLengths: list[int] | None = None) -> np.ndarray:
        """提取 240x4 的 SS/PBCH 频域网格。

        数据流：
        `rxSignal` -> 对 l=0..3：
        `[当前符号 CP 起点 = current]` -> 跳过 CP -> 用 `freqCompHz` 做频偏补偿
        -> FFT/fftshift -> 按 `ssbSubcarrierOffset` 切出配置的 240 个 SSB
        子载波。

        `ssbStart` 是 SS/PBCH 符号 0 的 CP 起点，主要由 `_candidateStarts()`
        根据 SSS 符号 2 定时反推得到。
        """
        useCp = self._cpLengths() if cpLengths is None else [int(v) for v in cpLengths]
        start = self.fftSize // 2 - self.SsbSubcarriers // 2 + self.ssbSubcarrierOffset
        grid = np.zeros((self.SsbSubcarriers, 4), dtype=np.complex64)
        current = int(ssbStart)
        for l in range(4):
            cpLength = int(useCp[l])
            usefulStart = current + cpLength
            usefulEnd = usefulStart + self.fftSize
            if current < 0 or usefulEnd > len(rxSignal):
                logger.warning(
                    "SSB symbol extraction out of range: start=%s, cp=%s, end=%s, signalLen=%s",
                    current,
                    cpLength,
                    usefulEnd,
                    len(rxSignal),
                )
                symbol = np.zeros(self.fftSize, dtype=np.complex64)
            else:
                useful = np.asarray(rxSignal[usefulStart:usefulEnd], dtype=np.complex64)
                if abs(float(freqCompHz)) > 0.0:
                    sampleIndex = np.arange(usefulStart, usefulEnd, dtype=np.float64)
                    phase = np.exp(-1j * 2.0 * np.pi * float(freqCompHz) * sampleIndex / self.sampleRate).astype(np.complex64)
                    useful = useful * phase
                symbol = np.fft.fftshift(np.fft.fft(useful)).astype(np.complex64)
            grid[:, l] = np.asarray(symbol[start:start + self.SsbSubcarriers], dtype=np.complex64)
            current += self.fftSize + cpLength
        return grid

    @staticmethod
    def _extract(grid: np.ndarray, items: list[PbchRe]) -> np.ndarray:
        return np.asarray([grid[item.k, item.l] for item in items], dtype=np.complex64)

    @staticmethod
    def _fitChannelFromDmrs(
        dmrsItems: list[PbchRe],
        hDmrs: np.ndarray,
        dataItems: list[PbchRe],
    ) -> np.ndarray:
        bySymDmrs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for l in (1, 2, 3):
            idx = [i for i, item in enumerate(dmrsItems) if item.l == l]
            k = np.asarray([dmrsItems[i].k for i in idx], dtype=np.float64)
            h = np.asarray([hDmrs[i] for i in idx], dtype=np.complex64)
            order = np.argsort(k)
            k = k[order]
            h = h[order]
            # New method: parametric channel model in frequency domain.
            # 1) smooth phase/magnitude
            phase = np.unwrap(np.angle(h)).astype(np.float64)
            mag = np.abs(h).astype(np.float64)
            if len(k) >= 7:
                w = np.asarray([1.0, 6.0, 16.0, 30.0, 45.0, 56.0, 60.0, 56.0, 45.0, 30.0, 16.0, 6.0, 1.0], dtype=np.float64)
                w = w / np.sum(w)
                phase = np.convolve(np.pad(phase, (6, 6), mode="edge"), w, mode="valid")
                mag = np.convolve(np.pad(mag, (6, 6), mode="edge"), w, mode="valid")

            # 2) polynomial fit to suppress noise/outliers
            degPhase = int(min(2, max(len(k) - 1, 1)))
            degMag = int(min(1, max(len(k) - 1, 1)))
            # Robust weighted LS fit: down-weight local outliers in pilot LS estimates.
            medMag = np.median(mag)
            madMag = np.median(np.abs(mag - medMag)) + 1e-9
            rMag = np.abs(mag - medMag) / (1.4826 * madMag)
            wMag = 1.0 / (1.0 + (rMag / 2.1) ** 2)

            medPha = np.median(phase)
            madPha = np.median(np.abs(phase - medPha)) + 1e-9
            rPha = np.abs(phase - medPha) / (1.4826 * madPha)
            wPha = 1.0 / (1.0 + (rPha / 2.4) ** 2)

            phaseCoef = np.polyfit(k, phase, deg=degPhase, w=wPha)
            magCoef = np.polyfit(k, np.log(np.maximum(mag, 1e-9)), deg=degMag, w=wMag)
            phaseFit = np.polyval(phaseCoef, k)
            magFit = np.exp(np.polyval(magCoef, k))
            hFit = (magFit * np.exp(1j * phaseFit)).astype(np.complex64)
            bySymDmrs[l] = (k, hFit)

        hData = np.zeros(len(dataItems), dtype=np.complex64)
        for i, item in enumerate(dataItems):
            kRef, hRef = bySymDmrs[item.l]
            magRef = np.abs(hRef).astype(np.float64)
            phaseRef = np.unwrap(np.angle(hRef)).astype(np.float64)
            mag = float(np.interp(float(item.k), kRef, magRef))
            phase = float(np.interp(float(item.k), kRef, phaseRef))
            hData[i] = np.complex64(mag * np.exp(1j * phase))
        return hData

    @staticmethod
    def _decisionDirectedPhaseRefine(dataEq: np.ndarray, dataItems: list[PbchRe], iterations: int = 2) -> np.ndarray:
        out = np.asarray(dataEq, dtype=np.complex64).copy()
        for _ in range(max(0, int(iterations))):
            for l in (1, 2, 3):
                idx = [i for i, item in enumerate(dataItems) if item.l == l]
                if not idx:
                    continue
                sym = out[np.asarray(idx, dtype=np.int32)]
                ref, _ = PbchDecoder._nearestQpsk(sym)
                metric = np.vdot(ref, sym)
                cpe = np.angle(metric) if abs(metric) > 1e-12 else 0.0
                out[np.asarray(idx, dtype=np.int32)] = sym * np.exp(-1j * cpe).astype(np.complex64)
        return out

    @staticmethod
    def _decisionDirectedChannelRefine(
        dataRx: np.ndarray,
        dataItems: list[PbchRe],
        dmrsItems: list[PbchRe],
        hDmrs: np.ndarray,
        initialEq: np.ndarray,
        iterations: int = 2,
    ) -> np.ndarray:
        """Refine data-channel estimate with decision-directed pseudo-pilots.

        Per PBCH symbol, combine:
        - DM-RS LS channel (strong anchors)
        - hard-decision-based data channel (weak pseudo-pilots)
        then refit a smooth magnitude/phase model on subcarrier index.
        """
        out = np.asarray(initialEq, dtype=np.complex64).copy()
        dataRxUse = np.asarray(dataRx, dtype=np.complex64).reshape(-1)
        hDmrsUse = np.asarray(hDmrs, dtype=np.complex64).reshape(-1)

        for _ in range(max(0, int(iterations))):
            hardRef, _ = PbchDecoder._nearestQpsk(out)
            hDataDd = dataRxUse / np.where(np.abs(hardRef) > 1e-9, hardRef, 1.0 + 0j)
            hDataNew = np.zeros_like(hDataDd)

            for l in (1, 2, 3):
                dataIdx = [i for i, item in enumerate(dataItems) if item.l == l]
                dmrsIdx = [i for i, item in enumerate(dmrsItems) if item.l == l]
                if not dataIdx or not dmrsIdx:
                    continue

                kData = np.asarray([dataItems[i].k for i in dataIdx], dtype=np.float64)
                hDataSym = np.asarray([hDataDd[i] for i in dataIdx], dtype=np.complex64)
                kDmrs = np.asarray([dmrsItems[i].k for i in dmrsIdx], dtype=np.float64)
                hDmrsSym = np.asarray([hDmrsUse[i] for i in dmrsIdx], dtype=np.complex64)

                # DM-RS anchors keep stronger weights than pseudo-pilots.
                kAll = np.concatenate([kData, kDmrs])
                hAll = np.concatenate([hDataSym, hDmrsSym]).astype(np.complex64)
                wAll = np.concatenate(
                    [
                        np.full(len(kData), 0.25, dtype=np.float64),
                        np.ones(len(kDmrs), dtype=np.float64),
                    ]
                )
                order = np.argsort(kAll)
                kAll = kAll[order]
                hAll = hAll[order]
                wAll = wAll[order]

                mag = np.abs(hAll).astype(np.float64)
                phase = np.unwrap(np.angle(hAll)).astype(np.float64)

                if len(kAll) >= 7:
                    kernel = np.asarray([1.0, 4.0, 7.0, 10.0, 13.0, 10.0, 7.0, 4.0, 1.0], dtype=np.float64)
                    kernel = kernel / np.sum(kernel)
                    mag = np.convolve(np.pad(mag, (4, 4), mode="edge"), kernel, mode="valid")
                    phase = np.convolve(np.pad(phase, (4, 4), mode="edge"), kernel, mode="valid")

                # Robust down-weight on top of pilot/data prior weights.
                magMad = np.median(np.abs(mag - np.median(mag))) + 1e-9
                magR = np.abs(mag - np.median(mag)) / (1.4826 * magMad)
                wMag = wAll / (1.0 + (magR / 2.4) ** 2)

                phaMad = np.median(np.abs(phase - np.median(phase))) + 1e-9
                phaR = np.abs(phase - np.median(phase)) / (1.4826 * phaMad)
                wPha = wAll / (1.0 + (phaR / 2.4) ** 2)

                degPha = int(min(2, max(len(kAll) - 1, 1)))
                degMag = int(min(1, max(len(kAll) - 1, 1)))
                phaCoef = np.polyfit(kAll, phase, deg=degPha, w=wPha)
                magCoef = np.polyfit(kAll, np.log(np.maximum(mag, 1e-9)), deg=degMag, w=wMag)
                phaFit = np.polyval(phaCoef, kData)
                magFit = np.exp(np.polyval(magCoef, kData))
                hDataNew[np.asarray(dataIdx, dtype=np.int32)] = (magFit * np.exp(1j * phaFit)).astype(np.complex64)

            out = dataRxUse / np.where(np.abs(hDataNew) > 1e-9, hDataNew, 1.0 + 0j)
            out = PbchDecoder._decisionDirectedPhaseRefine(out, dataItems, iterations=1)

        return np.asarray(out, dtype=np.complex64)

    @staticmethod
    def _periodicGainRefine(
        dataEq: np.ndarray,
        dataItems: list[PbchRe],
        period: int = 4,
        iterations: int = 1,
    ) -> np.ndarray:
        """Reduce periodic constellation distortion (e.g. k mod 4 groups).

        PBCH DM-RS is spaced every 4 subcarriers. In this capture, residual
        periodic mismatch often appears on data RE groups by subcarrier modulo 4.
        This step estimates one complex gain per (symbol, k mod period) group
        from hard decisions and removes it.
        """
        out = np.asarray(dataEq, dtype=np.complex64).copy()
        p = max(1, int(period))
        for _ in range(max(0, int(iterations))):
            ref, _ = PbchDecoder._nearestQpsk(out)
            for l in (1, 2, 3):
                symIdx = [i for i, item in enumerate(dataItems) if item.l == l]
                if not symIdx:
                    continue
                for modK in range(p):
                    grp = [i for i in symIdx if int(dataItems[i].k) % p == modK]
                    if not grp:
                        continue
                    idx = np.asarray(grp, dtype=np.int32)
                    y = out[idx]
                    r = ref[idx]
                    denom = max(float(np.vdot(r, r).real), 1e-12)
                    gain = np.vdot(r, y) / denom
                    if abs(gain) > 1e-12:
                        out[idx] = y / gain
        return out

    @staticmethod
    def _interpolateChannel(dmrsItems: list[PbchRe], hDmrs: np.ndarray, dataItems: list[PbchRe]) -> np.ndarray:
        hBySymbol: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for l in (1, 2, 3):
            idx = [i for i, item in enumerate(dmrsItems) if item.l == l]
            k = np.asarray([dmrsItems[i].k for i in idx], dtype=np.float64)
            h = np.asarray([hDmrs[i] for i in idx], dtype=np.complex64)
            if len(h) >= 5:
                # Smooth DMRS channel estimates per symbol to reduce interpolation noise.
                kernel = np.asarray([1.0, 7.0, 20.0, 40.0, 65.0, 90.0, 110.0, 118.0, 110.0, 90.0, 65.0, 40.0, 20.0, 7.0, 1.0], dtype=np.float32)
                kernel = kernel / np.sum(kernel)
                mag = np.abs(h).astype(np.float64)
                pha = np.unwrap(np.angle(h)).astype(np.float64)
                magPad = np.pad(mag, (7, 7), mode="edge")
                phaPad = np.pad(pha, (7, 7), mode="edge")
                magSm = np.convolve(magPad, kernel.astype(np.float64), mode="valid")
                phaSm = np.convolve(phaPad, kernel.astype(np.float64), mode="valid")
                h = (magSm * np.exp(1j * phaSm)).astype(np.complex64)
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
        """评估一个 PBCH 定时/频偏/iSSB/CP 候选假设。

        输入组合：
        - `ssbStart`：候选 SS/PBCH 符号 0 CP 起点采样。
        - `freqCompHz`：单位为 Hz 的频偏补偿。
        - `iSsbBar`：PBCH DM-RS 序列假设。
        - `cpLengths`：推进四个 OFDM 符号时使用的 CP profile。

        处理步骤：
        1. 从原始采样提取 240x4 SSB 网格。
        2. 根据 `nIdCell` 构造 DM-RS 和 PBCH data RE 列表。
        3. 生成本地 PBCH DM-RS，并在导频 RE 上估计信道。
        4. 对 PBCH data RE 插值/细化信道。
        5. 均衡 QPSK 符号并生成硬判决比特。

        输出是完整候选对象；`decodePbch()` 会保存紧凑摘要用于排序，并把
        选中的完整对象交给 BCH。
        """
        grid = self._extractSsbGrid(rxSignal, ssbStart=ssbStart, freqCompHz=freqCompHz, cpLengths=cpLengths)
        dmrsItems = self._dmrsReList(nIdCell)
        dataItems = self._pbchDataReList(nIdCell)
        dmrsRx = self._extract(grid, dmrsItems)
        dmrsRef = self.generatePbchDmrs(nIdCell, iSsbBar)
        hDmrs = dmrsRx / dmrsRef

        # DM-RS 相关值: 衡量接收 DM-RS 与本地生成 DM-RS 的相似度
        dmrsCorr = float(np.abs(np.vdot(dmrsRef, dmrsRx)))
        dmrsCorrNorm = dmrsCorr / max(float(np.linalg.norm(dmrsRef) * np.linalg.norm(dmrsRx)), 1e-12)

        # DM-RS EVM: 信道补偿后的 DM-RS 残差
        hMean = np.mean(hDmrs)
        if np.abs(hMean) > 1e-9:
            dmrsEq = dmrsRx / hMean
            dmrsEvm = self._evmPercent(dmrsEq, dmrsRef)
        else:
            dmrsEvm = 100.0

        dataRx = self._extract(grid, dataItems)
        hData = self._fitChannelFromDmrs(dmrsItems, hDmrs, dataItems)
        dataEq = dataRx / np.where(np.abs(hData) > 1e-9, hData, 1.0 + 0j)
        dataEq = self._decisionDirectedPhaseRefine(dataEq, dataItems, iterations=2)
        dataEq = self._decisionDirectedChannelRefine(
            dataRx=dataRx,
            dataItems=dataItems,
            dmrsItems=dmrsItems,
            hDmrs=hDmrs,
            initialEq=dataEq,
            iterations=2,
        )
        dataEq = self._periodicGainRefine(dataEq, dataItems, period=4, iterations=1)

        # Remove one residual common gain/phase so constellation EVM measures scatter, not arbitrary scaling.
        qpskRef0, _ = self._nearestQpsk(dataEq)
        gain = np.vdot(qpskRef0, dataEq) / max(float(np.vdot(qpskRef0, qpskRef0).real), 1e-12)
        dataEqNorm = dataEq / (gain if abs(gain) > 1e-12 else 1.0 + 0j)
        qpskRef, bits = self._nearestQpsk(dataEqNorm)
        pbchEvm = self._evmPercent(dataEqNorm, qpskRef)

        # 噪声方差估计: 使用数据 RE 的均衡后残差作为噪声估计
        noiseVarEstimate = float(np.mean(np.abs(dataEqNorm - qpskRef) ** 2))

        dmrsPower = float(np.mean(np.abs(dmrsRx) ** 2))
        dataPower = float(np.mean(np.abs(dataRx) ** 2))
        channelMean = complex(np.mean(hDmrs))
        channelStd = float(np.std(hDmrs))

        # DM-RS SNR: DM-RS 相关值可作为 SNR 估计的上限
        dmrsSnrDb = float(10.0 * np.log10(max(dmrsCorrNorm, 1e-12) / (1.0 - max(dmrsCorrNorm, 1e-12) + 1e-12)))

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
            # 新增 DM-RS 相关指标
            "dmrsCorr": dmrsCorr,
            "dmrsCorrNorm": dmrsCorrNorm,
            "dmrsEvmPercent": dmrsEvm,
            "noiseVarEstimate": noiseVarEstimate,
            "dmrsSnrDb": dmrsSnrDb,
        }

    @staticmethod
    def _candidateRankKey(candidate: dict, mode: str = "evm_guarded") -> tuple:
        """返回可排序键，值越大表示候选越好。

        候选排序是任务交接中的关键策略。`dmrs_only` 最直接遵循任务指导：
        DM-RS 相关高、DM-RS EVM 低、PBCH hard EVM 低。默认
        `evm_guarded` 先保留 DM-RS 可用性门限，再优先比较 PBCH EVM，
        同时用 DM-RS 质量作为并列候选的裁决依据。
        """
        dmrsCorrNorm = float(candidate.get("dmrsCorrNorm", 0.0))
        dmrsEvmPercent = float(candidate.get("dmrsEvmPercent", 1e9))
        evmPercent = float(candidate.get("evmPercent", candidate.get("pbchHardEvmPercent", 1e9)))

        if mode == "dmrs_only":
            return (dmrsCorrNorm, -dmrsEvmPercent, -evmPercent)
        if mode == "hard_evm":
            return (-evmPercent, dmrsCorrNorm, -dmrsEvmPercent)

        dmrsUsable = 1 if dmrsEvmPercent < 250.0 else 0
        return (dmrsUsable, -evmPercent, dmrsCorrNorm, -dmrsEvmPercent)

    @staticmethod
    def _anchorSsbStart(sssResult: dict, pssResult: dict, cpLengths: list[int], fftSize: int) -> int | None:
        """根据上游定时信息构造一个确定性的 SSB 起点锚点。

        `verifiedSymbolStart` 是 SSS 符号 2 的 CP 起点。减去符号 0 和符号
        1 的长度即可得到 SS/PBCH block 起点。PSS 的 `timingOffset` 是后备
        锚点，因为 PSS 本身位于符号 0。
        """
        sssOffset = 2 * int(fftSize) + int(cpLengths[0]) + int(cpLengths[1])
        if "verifiedSymbolStart" in sssResult:
            return int(sssResult["verifiedSymbolStart"]) - sssOffset
        if "bestSymbolStart" in sssResult:
            return int(sssResult["bestSymbolStart"]) - sssOffset
        if "ssbStart" in pssResult:
            return int(pssResult["ssbStart"])
        if "timingOffset" in pssResult:
            return int(pssResult["timingOffset"])
        return None

    @staticmethod
    def _selectStableCandidateSummary(
        candidates: list[dict],
        rankingMode: str,
        anchorSsbStart: int | None,
        anchorFreqHz: float | None,
        evmTolerancePercent: float = 0.12,
        maxPool: int = 24,
    ) -> tuple[dict | None, dict]:
        """Pick a stable winner from a near-optimal pool.

        1) keep candidates near best EVM;
        2) prefer closeness to anchor timing/frequency;
        3) break ties using DM-RS quality.
        """
        if not candidates:
            return None, {"poolSize": 0}

        ranked = sorted(candidates, key=lambda x: PbchDecoder._candidateRankKey(x, mode=rankingMode), reverse=True)
        bestByRank = ranked[0]
        bestEvm = float(bestByRank.get("evmPercent", 1e9))
        tol = max(0.0, float(evmTolerancePercent))
        pool = [x for x in ranked if float(x.get("evmPercent", 1e9)) <= bestEvm + tol][: max(1, int(maxPool))]
        if not pool:
            pool = [bestByRank]

        def stableKey(item: dict) -> tuple:
            startDist = abs(int(item.get("ssbStart", 0)) - int(anchorSsbStart)) if anchorSsbStart is not None else 0
            freqDist = (
                abs(float(item.get("freqCompHz", 0.0)) - float(anchorFreqHz))
                if anchorFreqHz is not None
                else 0.0
            )
            return (
                int(startDist),
                float(freqDist),
                float(item.get("evmPercent", 1e9)),
                -float(item.get("dmrsCorrNorm", 0.0)),
                float(item.get("dmrsEvmPercent", 1e9)),
                int(item.get("iSsbBar", 0)),
                int(item.get("ssbStart", 0)),
                float(item.get("freqCompHz", 0.0)),
            )

        chosen = min(pool, key=stableKey)
        return chosen, {
            "poolSize": int(len(pool)),
            "evmTolerancePercent": float(tol),
            "anchorSsbStart": None if anchorSsbStart is None else int(anchorSsbStart),
            "anchorFreqHz": None if anchorFreqHz is None else float(anchorFreqHz),
            "bestRankEvmPercent": float(bestEvm),
            "chosenEvmPercent": float(chosen.get("evmPercent", bestEvm)),
        }

    @staticmethod
    def _candidateStarts(sssResult: dict, pssResult: dict, cpLengths: list[int], fftSize: int) -> list[int]:
        """由 SSS 和 PSS 定时构造 PBCH `ssbStart` 候选。

        数据交接：
        - 主种子：`sssResult["verifiedSymbolStart"] - (len(sym0)+len(sym1))`。
        - 后备种子：`pssResult["timingOffset"]`。

        局部正负采样扩展用于吸收小的 FFT 窗口偏差。PBCH 比 PSS/SSS 更敏感，
        因为三个 PBCH 符号和 DM-RS 相位插值都依赖精确 FFT 窗口。
        """
        sssStarts: list[int] = []
        pssStarts: list[int] = []
        sssOffset = 2 * fftSize + cpLengths[0] + cpLengths[1]
        if "bestSymbolStart" in sssResult:
            sssStarts.append(int(sssResult["bestSymbolStart"]) - int(sssOffset))
        if "verifiedSymbolStart" in sssResult:
            sssStarts.append(int(sssResult["verifiedSymbolStart"]) - int(sssOffset))
        if "ssbStart" in pssResult:
            pssStarts.append(int(pssResult["ssbStart"]))
        if "timingOffset" in pssResult:
            pssStarts.append(int(pssResult["timingOffset"]))

        unique: list[int] = []

        def appendAround(center: int, radius: int) -> None:
            for delta in range(-int(radius), int(radius) + 1):
                candidate = int(center + delta)
                if candidate >= 0 and candidate not in unique:
                    unique.append(candidate)

        # SSS-derived starts are closer to the PBCH symbol geometry, but can still
        # be biased by a few samples. Use a wider local span so points like ssbStart=6
        # remain reachable from a bestSymbolStart-derived seed around -1/0.
        for value in sssStarts:
            appendAround(int(value), radius=8)

        # PSS timing is a weaker PBCH start anchor; keep a narrower expansion.
        for value in pssStarts:
            appendAround(int(value), radius=2)

        return unique

    def decodePbch(
        self,
        rxSignal: np.ndarray,
        sssResult: dict,
        pssResult: dict,
        freqCompHz: float | None = None,
        residualFreqSearchHz: float = 500.0,
        residualFreqStepHz: float = 50.0,
        rankingMode: str = "evm_guarded",
    ) -> dict:
        """运行 PBCH DM-RS 候选扫描，并返回选中的解调结果。

        主要候选维度：
        - `starts`：由 `_candidateStarts()` 得到的 SS/PBCH block 采样起点。
        - `freqGrid`：围绕 SSS 复核 `freqCompHz` 的残余频偏值。
        - `cpProfiles`：全 normal CP 或 slot-head 长 CP 假设。
        - `ssbIndexCandidates`：`iSsbBar` 假设，通常为 0..7。

        选中结果包含 `PbchBchDecoder.decode()` 消费的等化 PBCH 符号；
        原始 `grid/dmrs/data` 数组只在内存中保留，用于绘图/NPZ 工件，
        不写入主 JSON。
        """
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

        def evaluateAndRecord(ssbStart: int, freqHz: float, iSsbBar: int, cpProfileName: str, cpLens: list[int]) -> dict | None:
            nonlocal best
            try:
                item = self._evaluateCandidate(
                    rxSignal,
                    nIdCell=nIdCell,
                    ssbStart=int(ssbStart),
                    freqCompHz=float(freqHz),
                    iSsbBar=int(iSsbBar),
                    cpProfileName=str(cpProfileName),
                    cpLengths=[int(v) for v in cpLens],
                )
            except Exception:
                return None
            summary = {
                "iSsbBar": int(item["iSsbBar"]),
                "ssbStart": int(item["ssbStart"]),
                "freqCompHz": float(item["freqCompHz"]),
                "cpProfile": str(item.get("cpProfile", cpProfileName)),
                "evmPercent": float(item["evmPercent"]),
                "dmrsPower": float(item["dmrsPower"]),
                "channelStd": float(item["channelStd"]),
                "dmrsCorr": float(item["dmrsCorr"]),
                "dmrsCorrNorm": float(item["dmrsCorrNorm"]),
                "dmrsEvmPercent": float(item["dmrsEvmPercent"]),
                "noiseVarEstimate": float(item["noiseVarEstimate"]),
                "dmrsSnrDb": float(item["dmrsSnrDb"]),
                "pbchHardEvmPercent": float(item["evmPercent"]),
            }
            candidates.append(summary)
            # 排序优先级: DM-RS 相关最大 -> DM-RS EVM 最小 -> PBCH hard EVM 最小
            if best is None:
                best = item
            else:
                # 比较逻辑: 优先 dmrsCorrNorm 大，其次 dmrsEvmPercent 小，最后 evmPercent 小
                bestScore = self._candidateRankKey(best, mode=rankingMode)
                itemScore = self._candidateRankKey(item, mode=rankingMode)
                if itemScore > bestScore:
                    best = item
            return item

        # Stage-1: 始终扫描所有 iSsbBar=0..7，不优先固定某个值
        for ssbStart in starts:
            for freqHz in freqGrid:
                for cpProfileName, cpLens in cpProfiles:
                    for ib in self.ssbIndexCandidates:
                        evaluateAndRecord(ssbStart, float(freqHz), int(ib), cpProfileName, cpLens)

        # Stage-2 Top-K refinement seeds (基于 DM-RS 相关排序).
        rankedSeeds = sorted(candidates, key=lambda x: self._candidateRankKey(x, mode=rankingMode), reverse=True)[:3]
        for seed in rankedSeeds:
            cpProfileName = str(seed.get("cpProfile", "slot_head"))
            cpMap = {name: vals for name, vals in cpProfiles}
            cpLens = [int(v) for v in cpMap.get(cpProfileName, self._cpLengths())]
            fineStepHz = max(1.0, float(residualFreqStepHz) / 10.0)
            fineSearchHz = max(float(residualFreqStepHz), float(residualFreqStepHz) * 3.0)
            fineResidual = np.arange(-abs(fineSearchHz), abs(fineSearchHz) + fineStepHz / 2.0, abs(fineStepHz))
            fineGrid = float(seed["freqCompHz"]) + fineResidual
            for freqHz in fineGrid:
                evaluateAndRecord(int(seed["ssbStart"]), float(freqHz), int(seed["iSsbBar"]), cpProfileName, cpLens)

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
                evaluateAndRecord(
                    int(best["ssbStart"]),
                    float(freqHz),
                    int(best["iSsbBar"]),
                    str(best.get("cpProfile", "slot_head")),
                    [int(v) for v in dict(cpProfiles).get(str(best.get("cpProfile", "slot_head")), self._cpLengths())],
                )

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
                    evaluateAndRecord(
                        int(ssbStartLocal),
                        float(freqHz),
                        int(best["iSsbBar"]),
                        cpProfileName,
                        cpLengths,
                    )

        if best is not None:
            cpProfileName = str(best.get("cpProfile", "all_normal"))
            cpProfileMap = {name: vals for name, vals in cpProfiles}
            cpLengths = [int(v) for v in cpProfileMap.get(cpProfileName, self._cpLengths())]
            startCenter = int(best["ssbStart"])
            localStarts = np.arange(startCenter - 2, startCenter + 2 + 1, dtype=np.int32)
            freqCenter = float(best["freqCompHz"])
            ultraGrid = np.arange(freqCenter - 8.0, freqCenter + 8.0 + 1.0, 2.0, dtype=np.float64)
            for ssbStartLocal in localStarts:
                if ssbStartLocal < 0:
                    continue
                for freqHz in ultraGrid:
                    evaluateAndRecord(
                        int(ssbStartLocal),
                        float(freqHz),
                        int(best["iSsbBar"]),
                        cpProfileName,
                        cpLengths,
                    )

        if best is None:
            raise RuntimeError("PBCH 候选扫描失败")

        # 当多个近邻候选分数接近时，再做一次稳定性选择，降低偶然最优点的抖动。
        anchorSsbStart = self._anchorSsbStart(sssResult, pssResult, cpLengths, self.fftSize)
        chosenSummary, stableMeta = self._selectStableCandidateSummary(
            candidates=candidates,
            rankingMode=str(rankingMode),
            anchorSsbStart=anchorSsbStart,
            anchorFreqHz=float(freqCompHz),
            evmTolerancePercent=0.12,
            maxPool=24,
        )
        if chosenSummary is not None:
            cpProfileName = str(chosenSummary.get("cpProfile", "slot_head"))
            cpProfileMap = {name: vals for name, vals in cpProfiles}
            chosenCp = [int(v) for v in cpProfileMap.get(cpProfileName, self._cpLengths())]
            best = self._evaluateCandidate(
                rxSignal=rxSignal,
                nIdCell=nIdCell,
                ssbStart=int(chosenSummary["ssbStart"]),
                freqCompHz=float(chosenSummary["freqCompHz"]),
                iSsbBar=int(chosenSummary["iSsbBar"]),
                cpProfileName=cpProfileName,
                cpLengths=chosenCp,
            )
        else:
            stableMeta = {"poolSize": 0}

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
            "rankingMode": str(rankingMode),
            "dmrsCorr": float(best["dmrsCorr"]),
            "dmrsCorrNorm": float(best["dmrsCorrNorm"]),
            "dmrsEvmPercent": float(best["dmrsEvmPercent"]),
            "noiseVarEstimate": float(best["noiseVarEstimate"]),
            "dmrsSnrDb": float(best["dmrsSnrDb"]),
            "pbchHardEvmPercent": float(best["evmPercent"]),
            "selectionStability": stableMeta,
            # Top 10 候选按 DM-RS 相关排序
            "topCandidates": sorted(candidates, key=lambda x: self._candidateRankKey(x, mode=rankingMode), reverse=True)[:10],
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
