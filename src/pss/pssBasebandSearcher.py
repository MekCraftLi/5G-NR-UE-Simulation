"""
基带 PSS 自适应步进频率搜索模块

职责：在信号带宽范围内以多层自适应步进搜索最佳 PSS 频偏和定时。
      支持 CPU 多进程并行和 GPU (PyTorch) 加速两种模式。
"""
import logging
import multiprocessing
import os
import time

import numpy as np

from common.config import SsbConfig
from pss.pssTemplateFactory import buildPssTimeDomainTemplates

logger = logging.getLogger(__name__)

# ── CPU Worker 全局变量 ──────────────────────────────────────
_WORKER_RX_SIGNAL = None
_WORKER_TEMPLATE_FFTS = None
_WORKER_CORR_NFFT = None
_WORKER_SAMPLE_RATE = None
_WORKER_TEMPLATE_LENGTH = None
_WORKER_SAMPLE_INDEX = None


def _initWorker(rxSignal, templateFfts, corrNfft, sampleRate, templateLength):
    global _WORKER_RX_SIGNAL, _WORKER_TEMPLATE_FFTS, _WORKER_CORR_NFFT
    global _WORKER_SAMPLE_RATE, _WORKER_TEMPLATE_LENGTH, _WORKER_SAMPLE_INDEX
    _WORKER_RX_SIGNAL = rxSignal
    _WORKER_TEMPLATE_FFTS = templateFfts
    _WORKER_CORR_NFFT = corrNfft
    _WORKER_SAMPLE_RATE = sampleRate
    _WORKER_TEMPLATE_LENGTH = templateLength
    _WORKER_SAMPLE_INDEX = np.arange(len(rxSignal), dtype=np.float64)


def _searchSingleFrequency(task):
    freqHz, nId2 = task
    phase = np.exp(
        -1j * 2.0 * np.pi * freqHz * _WORKER_SAMPLE_INDEX / _WORKER_SAMPLE_RATE
    ).astype(np.complex64)
    compensated = (_WORKER_RX_SIGNAL * phase).astype(np.complex64)
    signalFft = np.fft.fft(compensated, n=_WORKER_CORR_NFFT).astype(np.complex64)
    corrFull = np.fft.ifft(signalFft * _WORKER_TEMPLATE_FFTS[nId2], n=_WORKER_CORR_NFFT)
    validStart = _WORKER_TEMPLATE_LENGTH - 1
    validEnd = len(_WORKER_RX_SIGNAL) - _WORKER_TEMPLATE_LENGTH + 1 + validStart
    corr = np.abs(corrFull[validStart:validEnd]).astype(np.float32)
    peakIdx = int(np.argmax(corr))
    peakValue = float(corr[peakIdx])
    return (freqHz, nId2, peakValue, peakIdx)


# ── Rich 进度条 ────────────────────────────────────────────────

class _ProgressReporter:
    """Rich 进度条，不可用时自动回退到 logger"""

    def __init__(self):
        self._rich = None
        self._taskId = None
        self._descPrefix = ""
        try:
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress, SpinnerColumn,
                TaskProgressColumn, TextColumn, TimeElapsedColumn,
                TimeRemainingColumn,
            )
            self._rich = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TaskProgressColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                TextColumn("• peak [yellow]{task.fields[peak]:.2f}"),
                transient=False,
                refresh_per_second=10,
            )
        except ImportError:
            pass

    def __enter__(self):
        if self._rich is not None:
            self._rich.start()
        return self

    def __exit__(self, *args):
        if self._rich is not None:
            self._rich.stop()

    def startStage(self, label: str, total: int):
        if self._rich is not None:
            self._taskId = self._rich.add_task(
                f"[bold cyan]{label}[/bold cyan]", total=total, peak=0.0,
            )
        else:
            logger.info(f"{label}: 0/{total}")

    def advance(self, increment: int = 1, currentPeak: float = 0.0):
        if self._rich is not None:
            self._rich.update(
                self._taskId, advance=increment,
                peak=max(currentPeak, 0.0),
            )
        else:
            pass  # log 版本在调用侧处理

    def endStage(self):
        if self._rich is not None and self._taskId is not None:
            self._rich.remove_task(self._taskId)
            self._taskId = None


# ── 搜索器 ────────────────────────────────────────────────────

class PssBasebandSearcher:
    """
    基带 PSS 自适应步进频率搜索器

    CPU 模式：multiprocessing.Pool 并行 FFT 卷积
    GPU 模式：PyTorch cuFFT 串行处理（单张 GPU 上 33M 点 FFT ~10-20ms）
    """

    def __init__(
        self,
        config: SsbConfig,
        freqMinHz: float,
        freqMaxHz: float,
        workers: int | None = None,
        useGpu: bool = True,
    ):
        self.config = config
        self.sampleRate = float(config.SampleRate)
        self.freqMinHz = float(freqMinHz)
        self.freqMaxHz = float(freqMaxHz)
        self.workers = workers or os.cpu_count() or 4

        self.pssTemplates = buildPssTimeDomainTemplates(config)
        self.templateLength = len(next(iter(self.pssTemplates.values())))

        # 时域反转核（FFT 在 _executeSearch 中按信号长度计算）
        self._pssKernels = {}
        for nId2, template in self.pssTemplates.items():
            self._pssKernels[nId2] = np.conjugate(template[::-1]).astype(np.complex64)

        # GPU 状态
        self._gpu = self._detectGpu(useGpu)
        self._gpuState = None

    # ── GPU 检测 ────────────────────────────────────────────

    @staticmethod
    def _detectGpu(requested: bool) -> str | None:
        if not requested:
            logger.info("GPU: disabled by config")
            return None
        try:
            import torch
        except ImportError:
            logger.info("GPU: torch not installed, using CPU")
            return None
        if torch.cuda.is_available():
            gpuName = torch.cuda.get_device_name(0)
            memGb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"GPU: {gpuName} ({memGb:.1f} GB), using CUDA")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("GPU: Apple MPS, using MPS")
            return "mps"
        logger.info("GPU: no CUDA/MPS device found, using CPU")
        return None

    # ── 内部工具 ────────────────────────────────────────────

    @staticmethod
    def _nextPow2(value: int) -> int:
        if value <= 1:
            return 1
        return 1 << (value - 1).bit_length()

    def _computeTemplateFftsNp(self, corrNfft: int) -> dict:
        return {
            nId2: np.fft.fft(kernel, n=corrNfft).astype(np.complex64)
            for nId2, kernel in self._pssKernels.items()
        }

    @staticmethod
    def _buildFreqGrid(freqMin: float, freqMax: float, step: float) -> np.ndarray:
        return np.arange(float(freqMin), float(freqMax) + step / 2, float(step))

    @staticmethod
    def _buildTaskList(freqGrid: np.ndarray) -> list:
        tasks = []
        for freqHz in freqGrid:
            for nId2 in range(3):
                tasks.append((float(freqHz), int(nId2)))
        return tasks

    # ── GPU 状态管理 ────────────────────────────────────────

    def _prepareGpuState(self, signal: np.ndarray):
        """将信号及相关数据加载到 GPU 并缓存"""
        import torch
        device = torch.device(self._gpu)
        signalLen = len(signal)
        convLength = signalLen + self.templateLength - 1
        corrNfft = self._nextPow2(convLength)

        signalGpu = torch.from_numpy(np.asarray(signal, dtype=np.complex64)).to(device)
        sampleIdxGpu = torch.arange(signalLen, dtype=torch.float64, device=device)

        templateFftsGpu = {}
        for nId2, kernel in self._pssKernels.items():
            k = torch.from_numpy(kernel).to(device)
            templateFftsGpu[nId2] = torch.fft.fft(k, n=corrNfft).type(torch.complex64)

        validStart = self.templateLength - 1
        validEnd = signalLen - self.templateLength + 1 + validStart

        self._gpuState = {
            "signal": signalGpu,
            "sampleIdx": sampleIdxGpu,
            "corrNfft": corrNfft,
            "templateFfts": templateFftsGpu,
            "validStart": validStart,
            "validEnd": validEnd,
            "signalLen": signalLen,
        }
        logger.info(
            f"[GPU:{self._gpu}] signalLen={signalLen}, templateLen={self.templateLength}, "
            f"corrNfft={corrNfft}"
        )

    def _ensureGpuState(self, signal: np.ndarray):
        if self._gpuState is None or self._gpuState["signalLen"] != len(signal):
            self._gpuState = None  # 释放旧 GPU 内存
            self._prepareGpuState(signal)

    # ── GPU 搜索 ────────────────────────────────────────────

    def _searchSingleFrequencyGpu(self, freqHz: float, nId2: int):
        """GPU 单频点搜索（在当前 CUDA stream 上执行）"""
        import torch
        gs = self._gpuState

        phase = torch.exp(
            -1j * 2.0 * torch.pi * freqHz * gs["sampleIdx"] / self.sampleRate
        ).type(torch.complex64)
        compensated = gs["signal"] * phase

        signalFft = torch.fft.fft(compensated, n=gs["corrNfft"]).type(torch.complex64)
        corrFull = torch.fft.ifft(signalFft * gs["templateFfts"][nId2])

        corr = torch.abs(corrFull[gs["validStart"]:gs["validEnd"]])
        peakIdx = torch.argmax(corr).item()
        peakVal = corr[peakIdx].item()
        return (float(freqHz), nId2, float(peakVal), peakIdx)

    @staticmethod
    def _gpuWorker(task, gpuState, sampleRate):
        """
        线程 worker：在独立 CUDA stream 上执行单个频点搜索

        每个线程使用自己的 CUDA stream，GPU 可以并发执行不同
        stream 上的 kernel（如 stream 0 做 FFT 时 stream 1 做相位旋转）
        """
        import torch
        gs = gpuState
        freqHz, nId2 = task

        # 每个线程获取独立的 CUDA stream
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            phase = torch.exp(
                -1j * 2.0 * torch.pi * freqHz * gs["sampleIdx"] / sampleRate
            ).type(torch.complex64)
            compensated = gs["signal"] * phase
            signalFft = torch.fft.fft(compensated, n=gs["corrNfft"]).type(torch.complex64)
            corrFull = torch.fft.ifft(signalFft * gs["templateFfts"][nId2])
            corr = torch.abs(corrFull[gs["validStart"]:gs["validEnd"]])
            peakIdx = torch.argmax(corr)
            peakVal = corr[peakIdx]

        # 同步该 stream 后读取结果
        stream.synchronize()
        return (float(freqHz), nId2, float(peakVal.item()), peakIdx.item())

    def _executeSearchGpu(
        self, signal: np.ndarray, freqGrid: np.ndarray, stageLabel: str = "",
    ) -> list:
        self._ensureGpuState(signal)
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks = self._buildTaskList(freqGrid)
        totalTasks = len(tasks)
        if not tasks:
            return []

        gpuWorkers = min(8, totalTasks)
        stepInfo = abs(freqGrid[1] - freqGrid[0]) if len(freqGrid) > 1 else 0
        label = f"GPU {stageLabel}"
        logger.info(
            f"[GPU] {stageLabel}: 范围=[{freqGrid[0]/1e3:.1f}, {freqGrid[-1]/1e3:.1f}] kHz, "
            f"步进={stepInfo:.1f} Hz, 频点={len(freqGrid)}, "
            f"任务={totalTasks}, 并发={gpuWorkers}"
        )

        startTime = time.time()
        results = [None] * totalTasks
        bestPeak = 0.0
        completed = 0
        logInterval = max(1, totalTasks // 10) if totalTasks >= 10 else 1

        with _ProgressReporter() as progress:
            progress.startStage(label, totalTasks)

            with ThreadPoolExecutor(max_workers=gpuWorkers) as executor:
                futureMap = {
                    executor.submit(
                        self._gpuWorker, task, self._gpuState, self.sampleRate,
                    ): idx
                    for idx, task in enumerate(tasks)
                }
                for future in as_completed(futureMap):
                    idx = futureMap[future]
                    freq, nid, peak, offset = future.result()
                    results[idx] = (freq, nid, peak, offset)
                    bestPeak = max(bestPeak, peak)
                    completed += 1
                    progress.advance(1, bestPeak)

                    if progress._rich is None and (completed % logInterval == 0 or completed == totalTasks):
                        elapsed = time.time() - startTime
                        rate = completed / max(elapsed, 1e-9)
                        eta = (totalTasks - completed) / max(rate, 1e-9)
                        logger.info(
                            f"[GPU:{stageLabel}] {completed}/{totalTasks} "
                            f"({100*completed/totalTasks:.0f}%), "
                            f"{rate:.1f} task/s, ETA {eta:.0f}s, peak={bestPeak:.2f}"
                        )

            progress.endStage()

        torch.cuda.synchronize()
        elapsed = time.time() - startTime
        logger.info(f"[GPU] {stageLabel} 完成: {totalTasks} 任务, 耗时 {elapsed:.1f}s, bestPeak={bestPeak:.2f}")
        return results

    # ── CPU 并行搜索 ────────────────────────────────────────

    def _executeSearchCpu(
        self, signal: np.ndarray, freqGrid: np.ndarray, stageLabel: str = "",
    ) -> list:
        tasks = self._buildTaskList(freqGrid)
        totalTasks = len(tasks)
        if not tasks:
            return []

        # Windows spawn 开销大，worker 太多反而更慢；cap 到 6
        cpuWorkers = min(self.workers, 6)

        label = f"CPU {stageLabel}"
        stepInfo = abs(freqGrid[1] - freqGrid[0]) if len(freqGrid) > 1 else 0
        logger.info(
            f"[CPU] {stageLabel}: 范围=[{freqGrid[0]/1e3:.1f}, {freqGrid[-1]/1e3:.1f}] kHz, "
            f"步进={stepInfo:.1f} Hz, 频点={len(freqGrid)}, "
            f"任务={totalTasks}, workers={cpuWorkers}"
        )

        signalC64 = signal.astype(np.complex64)
        convLength = len(signalC64) + self.templateLength - 1
        corrNfft = self._nextPow2(convLength)
        templateFfts = self._computeTemplateFftsNp(corrNfft)
        logger.info(f"[CPU] {stageLabel}: signalLen={len(signalC64)}, corrNfft={corrNfft}")
        logger.info(f"[CPU] {stageLabel}: spawning {cpuWorkers} workers, may take a moment on Windows...")

        startTime = time.time()
        results = []

        with _ProgressReporter() as progress:
            progress.startStage(label, totalTasks)

            with multiprocessing.Pool(
                processes=cpuWorkers,
                initializer=_initWorker,
                initargs=(signalC64, templateFfts, corrNfft,
                          self.sampleRate, self.templateLength),
            ) as pool:
                bestPeak = 0.0
                completed = 0
                logInterval = max(1, totalTasks // 10) if totalTasks >= 10 else 1
                # chunksize=1: 每完成一个任务立即返回，进度条实时更新
                for result in pool.imap_unordered(_searchSingleFrequency, tasks, chunksize=1):
                    results.append(result)
                    bestPeak = max(bestPeak, result[2])
                    completed += 1
                    progress.advance(1, bestPeak)

                    if progress._rich is None and (completed % logInterval == 0 or completed == totalTasks):
                        elapsed = time.time() - startTime
                        rate = completed / max(elapsed, 1e-9)
                        eta = (totalTasks - completed) / max(rate, 1e-9)
                        logger.info(
                            f"[CPU:{stageLabel}] {completed}/{totalTasks} "
                            f"({100*completed/totalTasks:.0f}%), "
                            f"{rate:.1f} task/s, ETA {eta:.0f}s, peak={bestPeak:.2f}"
                        )

            progress.endStage()

        elapsed = time.time() - startTime
        logger.info(f"[CPU] {stageLabel} 完成: {totalTasks} 任务, 耗时 {elapsed:.1f}s")
        return results

    def _executeSearch(
        self, signal: np.ndarray, freqGrid: np.ndarray, stageLabel: str = "",
    ) -> list:
        if self._gpu is not None:
            return self._executeSearchGpu(signal, freqGrid, stageLabel)
        return self._executeSearchCpu(signal, freqGrid, stageLabel)

    @staticmethod
    def _bestFromResults(results: list) -> tuple:
        return max(results, key=lambda x: x[2])

    # ── 自适应多层搜索 ──────────────────────────────────────

    def searchAdaptive(
        self,
        signal: np.ndarray,
        coarseStepHz: float = 1000.0,
        mediumStepHz: float = 100.0,
        fineStepHz: float = 15.0,
        subfineIterations: int = 5,
    ) -> dict:
        allPassPeaks = []

        grid1 = self._buildFreqGrid(self.freqMinHz, self.freqMaxHz, coarseStepHz)
        res1 = self._executeSearch(signal, grid1, "粗搜")
        allPassPeaks.extend((r[0], r[1], r[2], r[3]) for r in res1)
        f1, n1, p1, _ = self._bestFromResults(res1)
        logger.info(f"粗搜最佳: freq={f1/1e3:.3f} kHz, N_ID_2={n1}, peak={p1:.2f}")

        span2 = coarseStepHz * 2.0
        grid2 = self._buildFreqGrid(f1 - span2, f1 + span2, mediumStepHz)
        res2 = self._executeSearch(signal, grid2, "中搜")
        allPassPeaks.extend((r[0], r[1], r[2], r[3]) for r in res2)
        f2, n2, p2, _ = self._bestFromResults(res2)
        logger.info(f"中搜最佳: freq={f2/1e3:.3f} kHz, N_ID_2={n2}, peak={p2:.2f}")

        span3 = mediumStepHz * 2.0
        grid3 = self._buildFreqGrid(f2 - span3, f2 + span3, fineStepHz)
        res3 = self._executeSearch(signal, grid3, "细搜")
        allPassPeaks.extend((r[0], r[1], r[2], r[3]) for r in res3)
        f3, n3, p3, _ = self._bestFromResults(res3)
        logger.info(f"细搜最佳: freq={f3/1e3:.3f} kHz, N_ID_2={n3}, peak={p3:.2f}")

        finalNId2 = int(n3)
        center = float(f3)
        step = float(fineStepHz)
        history = []
        validIters = 0
        stopStatus = "max_iterations"

        for i in range(1, subfineIterations + 1):
            subStep = step / 2.0
            offsets = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0]) * subStep
            freqs = center + offsets

            subPeaks = []
            for f in freqs:
                corr = self.getCorrelationAtFreq(signal, float(f), finalNId2)
                idx = int(np.argmax(corr))
                val = float(corr[idx])
                subPeaks.append((float(f), finalNId2, val, idx))
            allPassPeaks.extend(subPeaks)

            peakArr = np.asarray([p[2] for p in subPeaks], dtype=np.float64)
            bestIdx = int(np.argmax(peakArr))
            interp = self._parabolicInterpolate(freqs, peakArr, bestIdx)

            history.append({
                "iteration": i,
                "centerFreqHz": float(center),
                "stepHz": float(subStep),
                "gridBestFreqHz": float(freqs[bestIdx]),
                "gridBestPeak": float(peakArr[bestIdx]),
                "interpValid": interp["valid"],
                "interpStatus": interp["status"],
                "interpFreqHz": interp["interpFreqHz"],
                "interpPeak": interp["interpPeak"],
            })

            if not interp["valid"]:
                stopStatus = interp["status"]
                break
            center = interp["interpFreqHz"]
            step = subStep
            validIters += 1

        if subfineIterations == 0:
            stopStatus = "disabled"

        finalFreqHz = float(center)
        finalCorr = self.getCorrelationAtFreq(signal, finalFreqHz, finalNId2)
        finalOffset = int(np.argmax(finalCorr))
        finalPeak = float(finalCorr[finalOffset])

        logger.info(
            f"子精细: nId2={finalNId2}, iters={validIters}/{subfineIterations}, "
            f"status={stopStatus}, freq={finalFreqHz:.3f} Hz, "
            f"offset={finalOffset}, peak={finalPeak:.6f}"
        )

        nId2BestResults = []
        for nid in range(3):
            corr = self.getCorrelationAtFreq(signal, finalFreqHz, nid)
            idx = int(np.argmax(corr))
            val = float(corr[idx])
            nId2BestResults.append({
                "nId2": nid, "gscn": 0, "freqOffset": finalFreqHz,
                "timingOffset": idx, "peakValue": val, "corrArray": corr,
            })

        nId2IndividualBest = []
        for nid in range(3):
            nidPeaks = [p for p in allPassPeaks if p[1] == nid]
            if nidPeaks:
                best = max(nidPeaks, key=lambda x: x[2])
                bFreq = float(best[0])
                corr = self.getCorrelationAtFreq(signal, bFreq, nid)
                nId2IndividualBest.append({
                    "nId2": nid, "freqHz": bFreq,
                    "timingOffset": int(np.argmax(corr)),
                    "peakValue": float(corr[int(np.argmax(corr))]),
                    "corrArray": corr,
                })
            else:
                corr = self.getCorrelationAtFreq(signal, finalFreqHz, nid)
                nId2IndividualBest.append({
                    "nId2": nid, "freqHz": finalFreqHz,
                    "timingOffset": int(np.argmax(corr)),
                    "peakValue": float(corr[int(np.argmax(corr))]),
                    "corrArray": corr,
                })

        return {
            "timingOffset": finalOffset,
            "nId2": finalNId2,
            "freqOffset": float(f3),
            "freqOffsetParabolic": finalFreqHz,
            "gscn": 0,
            "peakValue": finalPeak,
            "corrArray": nId2BestResults[finalNId2]["corrArray"],
            "scoreMode": "raw",
            "useBasebandMode": True,
            "nId2BestResults": nId2BestResults,
            "nId2IndividualBest": nId2IndividualBest,
            "adaptiveRefinement": {
                "method": "multi_pass_adaptive",
                "maxIterations": subfineIterations,
                "finalNId2": finalNId2,
                "finalRefinedFreqHz": finalFreqHz,
                "finalStatus": stopStatus,
                "finalValidIterations": validIters,
                "perNId2": [{
                    "nId2": finalNId2,
                    "startFreqHz": float(f3),
                    "startStepHz": float(fineStepHz),
                    "maxIterations": subfineIterations,
                    "validIterations": validIters,
                    "finalStatus": stopStatus,
                    "finalFreqHz": finalFreqHz,
                    "finalPeak": finalPeak,
                    "finalTimingOffset": finalOffset,
                    "history": history,
                }],
            },
            "freqSearch": {
                "method": "multi_pass_adaptive",
                "passes": {
                    "coarse": {"stepHz": coarseStepHz, "bestFreqHz": f1, "bestNId2": n1, "bestPeak": p1},
                    "medium": {"stepHz": mediumStepHz, "bestFreqHz": f2, "bestNId2": n2, "bestPeak": p2},
                    "fine":   {"stepHz": fineStepHz,   "bestFreqHz": f3, "bestNId2": n3, "bestPeak": p3},
                    "subfine": {"iterations": subfineIterations, "validIters": validIters, "finalFreqHz": finalFreqHz},
                },
            },
            "allPassPeaks": allPassPeaks,
        }

    # ── 抛物线插值 ──────────────────────────────────────────

    @staticmethod
    def _parabolicInterpolate(freqArray, peakArray, bestIndex):
        result = {
            "valid": False, "status": "unknown",
            "interpFreqHz": float(freqArray[bestIndex]),
            "interpPeak": float(peakArray[bestIndex]),
        }
        if len(freqArray) < 3:
            result["status"] = "insufficient_points"; return result
        if bestIndex <= 0 or bestIndex >= len(freqArray) - 1:
            result["status"] = "edge_point"; return result

        x = np.asarray([freqArray[bestIndex - 1], freqArray[bestIndex], freqArray[bestIndex + 1]], dtype=np.float64)
        y = np.asarray([peakArray[bestIndex - 1], peakArray[bestIndex], peakArray[bestIndex + 1]], dtype=np.float64)

        if not (y[1] >= y[0] and y[1] >= y[2]):
            result["status"] = "not_local_max"; return result

        x0, cLocal = float(x[1]), float(y[1])
        hL, hR = float(x0 - x[0]), float(x[2] - x0)
        if hL <= 0 or hR <= 0:
            result["status"] = "invalid_spacing"; return result

        mat = np.asarray([[hL * hL, -hL], [hR * hR, hR]], dtype=np.float64)
        rhs = np.asarray([float(y[0] - cLocal), float(y[2] - cLocal)], dtype=np.float64)
        try:
            a, b = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            result["status"] = "singular_system"; return result

        aLocal, bLocal = float(a), float(b)
        if abs(aLocal) < 1e-18:
            result["status"] = "flat_quadratic"; return result
        if aLocal >= 0:
            result["status"] = "non_concave"; return result

        tVertex = -bLocal / (2.0 * aLocal)
        xVertex = x0 + tVertex
        stepHz = abs(x[1] - x[0]) if len(freqArray) > 1 else 0
        if not (x[0] - 0.5 * stepHz <= xVertex <= x[2] + 0.5 * stepHz):
            result["status"] = "vertex_outside_neighbor"; return result

        result["interpFreqHz"] = float(xVertex)
        result["interpPeak"] = float(aLocal * tVertex * tVertex + bLocal * tVertex + cLocal)
        result["valid"] = True
        result["status"] = "ok"
        return result

    # ── 单点互相关 ──────────────────────────────────────────

    def getCorrelationAtFreq(
        self, signal: np.ndarray, freqHz: float, nId2: int,
    ) -> np.ndarray:
        """获取指定频率和 N_ID_2 的互相关序列（自动选择 GPU/CPU）"""
        if self._gpuState is not None and self._gpuState["signalLen"] == len(signal):
            return self._getCorrelationAtFreqGpu(freqHz, nId2)
        return self._getCorrelationAtFreqCpu(signal, freqHz, nId2)

    def _getCorrelationAtFreqGpu(self, freqHz: float, nId2: int) -> np.ndarray:
        import torch
        gs = self._gpuState
        phase = torch.exp(
            -1j * 2.0 * torch.pi * freqHz * gs["sampleIdx"] / self.sampleRate
        ).type(torch.complex64)
        compensated = gs["signal"] * phase

        signalFft = torch.fft.fft(compensated, n=gs["corrNfft"]).type(torch.complex64)
        corrFull = torch.fft.ifft(signalFft * gs["templateFfts"][nId2])
        corr = torch.abs(corrFull[gs["validStart"]:gs["validEnd"]])
        return corr.cpu().numpy().astype(np.float32)

    def _getCorrelationAtFreqCpu(
        self, signal: np.ndarray, freqHz: float, nId2: int,
    ) -> np.ndarray:
        signalLen = len(signal)
        convLength = signalLen + self.templateLength - 1
        corrNfft = self._nextPow2(convLength)
        templateFft = np.fft.fft(self._pssKernels[nId2], n=corrNfft).astype(np.complex64)

        sampleIndex = np.arange(signalLen, dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * freqHz * sampleIndex / self.sampleRate).astype(np.complex64)
        compensated = (np.asarray(signal, dtype=np.complex64) * phase).astype(np.complex64)

        signalFft = np.fft.fft(compensated, n=corrNfft).astype(np.complex64)
        corrFull = np.fft.ifft(signalFft * templateFft, n=corrNfft)

        validStart = self.templateLength - 1
        validEnd = signalLen - self.templateLength + 1 + validStart
        return np.abs(corrFull[validStart:validEnd]).astype(np.float32)
