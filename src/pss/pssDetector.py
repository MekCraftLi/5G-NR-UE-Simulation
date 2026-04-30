import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from common.config import SsbConfig
from pss.pssTemplateFactory import buildPssTimeDomainTemplates

logger = logging.getLogger(__name__)


_WORKER_RX_SIGNAL: np.ndarray | None = None
_WORKER_PSS_TEMPLATES: dict[int, np.ndarray] | None = None
_WORKER_SAMPLE_RATE: float | None = None
_WORKER_SAMPLE_INDEX: np.ndarray | None = None
_WORKER_CORR_NFFT: int | None = None
_WORKER_CORR_VALID_START: int | None = None
_WORKER_CORR_VALID_END: int | None = None
_WORKER_TEMPLATE_FFTS: dict[int, np.ndarray] | None = None
_WORKER_TEMPLATE_NORMS: dict[int, float] | None = None
_WORKER_SCORE_MODE: str = "raw"
_WORKER_WINDOW_ENERGY_SQRT: np.ndarray | None = None


def _nextPow2(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _buildCorrelationPlan(
    signalLength: int,
    pssTemplates: dict[int, np.ndarray],
) -> tuple[int, int, int, dict[int, np.ndarray]]:
    if len(pssTemplates) == 0:
        raise ValueError("pssTemplates is empty")

    templateLengths = {len(v) for v in pssTemplates.values()}
    if len(templateLengths) != 1:
        raise ValueError("All PSS templates must have same length")
    templateLength = int(next(iter(templateLengths)))
    if signalLength < templateLength:
        raise ValueError(
            f"Signal too short for correlation plan: signal={signalLength}, template={templateLength}"
        )

    convLength = signalLength + templateLength - 1
    corrNfft = _nextPow2(convLength)
    validStart = templateLength - 1
    validLength = signalLength - templateLength + 1
    validEnd = validStart + validLength

    templateFfts: dict[int, np.ndarray] = {}
    for nId2, template in pssTemplates.items():
        kernel = np.conjugate(template[::-1]).astype(np.complex64)
        templateFfts[int(nId2)] = np.fft.fft(kernel, n=corrNfft).astype(np.complex64)

    return corrNfft, validStart, validEnd, templateFfts


def _fftValidCorrelationMagnitude(
    signalFft: np.ndarray,
    templateFft: np.ndarray,
    corrNfft: int,
    validStart: int,
    validEnd: int,
) -> np.ndarray:
    corrFull = np.fft.ifft(signalFft * templateFft, n=corrNfft)
    return np.abs(corrFull[validStart:validEnd]).astype(np.float32)


def _initWorker(
    rxSignal: np.ndarray,
    pssTemplates: dict[int, np.ndarray],
    sampleRate: float,
    scoreMode: str,
    windowEnergySqrt: np.ndarray | None,
):
    global _WORKER_RX_SIGNAL, _WORKER_PSS_TEMPLATES, _WORKER_SAMPLE_RATE, _WORKER_SAMPLE_INDEX
    global _WORKER_CORR_NFFT, _WORKER_CORR_VALID_START, _WORKER_CORR_VALID_END, _WORKER_TEMPLATE_FFTS
    global _WORKER_TEMPLATE_NORMS, _WORKER_SCORE_MODE, _WORKER_WINDOW_ENERGY_SQRT
    _WORKER_RX_SIGNAL = rxSignal
    _WORKER_PSS_TEMPLATES = pssTemplates
    _WORKER_SAMPLE_RATE = sampleRate
    _WORKER_SAMPLE_INDEX = np.arange(len(rxSignal), dtype=np.float64)
    corrNfft, validStart, validEnd, templateFfts = _buildCorrelationPlan(len(rxSignal), pssTemplates)
    _WORKER_CORR_NFFT = int(corrNfft)
    _WORKER_CORR_VALID_START = int(validStart)
    _WORKER_CORR_VALID_END = int(validEnd)
    _WORKER_TEMPLATE_FFTS = templateFfts
    _WORKER_TEMPLATE_NORMS = {int(k): float(np.linalg.norm(v)) for k, v in pssTemplates.items()}
    _WORKER_SCORE_MODE = str(scoreMode).lower()
    _WORKER_WINDOW_ENERGY_SQRT = None if windowEnergySqrt is None else np.asarray(windowEnergySqrt, dtype=np.float32)


def _searchSingleFrequency(task: tuple[int, int, float]) -> dict:
    taskIndex, gscn, freqHz = task
    if (
        _WORKER_RX_SIGNAL is None
        or _WORKER_PSS_TEMPLATES is None
        or _WORKER_SAMPLE_RATE is None
        or _WORKER_SAMPLE_INDEX is None
        or _WORKER_CORR_NFFT is None
        or _WORKER_CORR_VALID_START is None
        or _WORKER_CORR_VALID_END is None
        or _WORKER_TEMPLATE_FFTS is None
        or _WORKER_TEMPLATE_NORMS is None
    ):
        raise RuntimeError("Worker is not initialized")

    phase = np.exp(-1j * 2.0 * np.pi * freqHz * _WORKER_SAMPLE_INDEX / _WORKER_SAMPLE_RATE).astype(np.complex64)
    compensated = (_WORKER_RX_SIGNAL * phase).astype(np.complex64)
    signalFft = np.fft.fft(compensated, n=_WORKER_CORR_NFFT).astype(np.complex64)

    peaks = {}
    for nId2, templateFft in _WORKER_TEMPLATE_FFTS.items():
        corr = _fftValidCorrelationMagnitude(
            signalFft=signalFft,
            templateFft=templateFft,
            corrNfft=_WORKER_CORR_NFFT,
            validStart=_WORKER_CORR_VALID_START,
            validEnd=_WORKER_CORR_VALID_END,
        )
        if _WORKER_SCORE_MODE == "ncc":
            if _WORKER_WINDOW_ENERGY_SQRT is None:
                raise RuntimeError("NCC mode requires window energy array")
            templateNorm = float(_WORKER_TEMPLATE_NORMS[int(nId2)])
            denom = _WORKER_WINDOW_ENERGY_SQRT * max(templateNorm, 1e-12)
            score = np.zeros_like(corr, dtype=np.float32)
            mask = denom > 1e-12
            score[mask] = corr[mask] / denom[mask]
        else:
            score = corr

        peakIndex = int(np.argmax(score))
        peakValue = float(score[peakIndex])
        peaks[int(nId2)] = (peakValue, peakIndex)

    return {
        "taskIndex": int(taskIndex),
        "gscn": int(gscn),
        "freqHz": float(freqHz),
        "peaks": peaks,
    }


@dataclass
class _StageCounter:
    stageName: str
    total: int
    done: int
    startTime: float


class _ProgressReporter:
    def __init__(self, mode: str = "auto", minLogIntervalSec: float = 0.7):
        self._mode = mode
        self._minLogIntervalSec = max(0.1, float(minLogIntervalSec))
        self._lastLogTime = 0.0
        self._stage: _StageCounter | None = None
        self._richModeEnabled = False
        self._progress = None
        self._taskId = None

        richRequested = mode.lower() in ("auto", "rich")
        if richRequested:
            try:
                from rich.progress import (
                    BarColumn,
                    Progress,
                    SpinnerColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )

                self._progress = Progress(
                    SpinnerColumn(),
                    TextColumn("[bold cyan]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TextColumn("•"),
                    TimeElapsedColumn(),
                    TextColumn("•"),
                    TimeRemainingColumn(),
                    transient=False,
                    refresh_per_second=10,
                )
                self._richModeEnabled = True
                self._richColumns = (
                    SpinnerColumn,
                    BarColumn,
                    TaskProgressColumn,
                    TextColumn,
                    TimeElapsedColumn,
                    TimeRemainingColumn,
                )
            except Exception:
                self._richModeEnabled = False
                if mode.lower() == "rich":
                    logger.info("rich unavailable, fallback to log progress mode")

    def __enter__(self):
        if self._richModeEnabled and self._progress is not None:
            self._progress.start()
        return self

    def __exit__(self, excType, excVal, excTb):
        if self._richModeEnabled and self._progress is not None:
            self._progress.stop()

    @staticmethod
    def _formatSeconds(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins:02d}:{secs:02d}"

    def startStage(self, stageName: str, total: int):
        self._stage = _StageCounter(
            stageName=stageName,
            total=max(1, int(total)),
            done=0,
            startTime=time.time(),
        )
        self._lastLogTime = 0.0

        if self._richModeEnabled and self._progress is not None:
            if self._taskId is not None:
                self._progress.remove_task(self._taskId)
            self._taskId = self._progress.add_task(stageName, total=max(1, int(total)))
        else:
            logger.info(f"{stageName}: 0/{total}")

    def advance(self, increment: int = 1, bestPeak: float | None = None):
        if self._stage is None:
            return
        self._stage.done += int(increment)
        done = min(self._stage.done, self._stage.total)
        elapsed = max(1e-9, time.time() - self._stage.startTime)
        speed = done / elapsed
        eta = (self._stage.total - done) / speed if speed > 0 else float("inf")
        peakText = f" | peak={bestPeak:.4f}" if bestPeak is not None else ""

        if self._richModeEnabled and self._progress is not None and self._taskId is not None:
            desc = (
                f"{self._stage.stageName}  "
                f"{done}/{self._stage.total}  "
                f"{speed:.2f} task/s  ETA {self._formatSeconds(eta)}"
                f"{peakText}"
            )
            self._progress.update(self._taskId, completed=done, description=desc)
            return

        now = time.time()
        if done >= self._stage.total or (now - self._lastLogTime) >= self._minLogIntervalSec:
            self._lastLogTime = now
            logger.info(
                f"{self._stage.stageName}: {done}/{self._stage.total} "
                f"({done / self._stage.total * 100.0:.1f}%) "
                f"speed={speed:.2f} task/s ETA={self._formatSeconds(eta)}{peakText}"
            )

    def endStage(self):
        if self._stage is None:
            return
        elapsed = time.time() - self._stage.startTime
        logger.info(f"{self._stage.stageName}: done in {elapsed:.2f}s")
        self._stage = None
        self._taskId = None


class PssDetector:
    def __init__(
        self,
        config: SsbConfig,
        fineStepHz: float = 100.0,
        fineSearchRangeHz: float = 10000.0,
        adaptiveRefineIterations: int = 5,
        maxWorkers: int | None = None,
        parallelMode: str = "process",
        progressMode: str = "auto",
        progressRefreshSec: float = 0.7,
        scoreMode: str = "raw",
    ):
        self.config = config
        self.sampleRate = float(config.SampleRate)
        self.fftSize = int(config.FftSize)
        self.fineStepHz = float(fineStepHz)
        self.fineSearchRangeHz = float(fineSearchRangeHz)
        self.adaptiveRefineIterations = max(0, int(adaptiveRefineIterations))
        self.parallelMode = str(parallelMode).lower()
        self.progressMode = str(progressMode)
        self.progressRefreshSec = float(progressRefreshSec)
        self.scoreMode = str(scoreMode).lower()

        if maxWorkers is None:
            envValue = os.getenv("PSS_MAX_WORKERS") or os.getenv("NR_PSS_MAX_WORKERS")
            if envValue:
                try:
                    maxWorkers = int(envValue)
                except ValueError:
                    maxWorkers = None
        self.maxWorkers = int(maxWorkers) if maxWorkers is not None else int(os.cpu_count() or 1)
        self.maxWorkers = max(1, self.maxWorkers)
        if self.parallelMode not in ("process", "thread"):
            raise ValueError("parallelMode must be one of: process, thread")
        if self.scoreMode not in ("raw", "ncc"):
            raise ValueError("scoreMode must be one of: raw, ncc")

        self._pssTemplates = buildPssTimeDomainTemplates(config)
        self._windowEnergySqrt: np.ndarray | None = None

    def _buildFineSearchFreqs(self, centerFreqHz: float) -> np.ndarray:
        start = centerFreqHz - self.fineSearchRangeHz
        stop = centerFreqHz + self.fineSearchRangeHz
        count = int(round((stop - start) / self.fineStepHz)) + 1
        return np.linspace(start, stop, num=count, dtype=np.float64)

    @staticmethod
    def _computeCorrArray(rxSignal: np.ndarray, template: np.ndarray, freqHz: float, sampleRate: float) -> np.ndarray:
        sampleIndex = np.arange(len(rxSignal), dtype=np.float64)
        phase = np.exp(-1j * 2.0 * np.pi * freqHz * sampleIndex / sampleRate).astype(np.complex64)
        compensated = (rxSignal * phase).astype(np.complex64)
        signalLength = int(len(compensated))
        templateLength = int(len(template))
        if signalLength < templateLength:
            raise ValueError(
                f"Signal too short for correlation: signal={signalLength}, template={templateLength}"
            )

        convLength = signalLength + templateLength - 1
        corrNfft = _nextPow2(convLength)
        validStart = templateLength - 1
        validEnd = validStart + (signalLength - templateLength + 1)

        signalFft = np.fft.fft(compensated, n=corrNfft).astype(np.complex64)
        templateKernel = np.conjugate(template[::-1]).astype(np.complex64)
        templateFft = np.fft.fft(templateKernel, n=corrNfft).astype(np.complex64)
        return _fftValidCorrelationMagnitude(
            signalFft=signalFft,
            templateFft=templateFft,
            corrNfft=corrNfft,
            validStart=validStart,
            validEnd=validEnd,
        )

    @staticmethod
    def _computeWindowEnergySqrt(rxSignal: np.ndarray, windowLength: int) -> np.ndarray:
        if len(rxSignal) < windowLength:
            raise ValueError("Signal too short for window energy")
        power = np.abs(rxSignal).astype(np.float64) ** 2
        prefix = np.concatenate([np.asarray([0.0], dtype=np.float64), np.cumsum(power, dtype=np.float64)])
        energy = prefix[windowLength:] - prefix[:-windowLength]
        return np.sqrt(np.maximum(energy, 0.0)).astype(np.float32)

    def _computeScoreArray(self, rxSignal: np.ndarray, template: np.ndarray, freqHz: float) -> np.ndarray:
        corr = self._computeCorrArray(
            rxSignal=rxSignal,
            template=template,
            freqHz=freqHz,
            sampleRate=self.sampleRate,
        )
        if self.scoreMode != "ncc":
            return corr
        if self._windowEnergySqrt is None:
            raise RuntimeError("NCC mode requires precomputed window energy")
        templateNorm = float(np.linalg.norm(template))
        denom = self._windowEnergySqrt * max(templateNorm, 1e-12)
        score = np.zeros_like(corr, dtype=np.float32)
        mask = denom > 1e-12
        score[mask] = corr[mask] / denom[mask]
        return score

    @staticmethod
    def _parabolicInterpolateFinePeak(freqArray: np.ndarray, peakArray: np.ndarray, bestIndex: int) -> dict:
        result = {
            "valid": False,
            "status": "unknown",
            "method": "none",
            "gridBestIndex": int(bestIndex),
            "gridBestFreqHz": float(freqArray[bestIndex]),
            "gridBestPeak": float(peakArray[bestIndex]),
            "interpFreqHz": float(freqArray[bestIndex]),
            "interpPeak": float(peakArray[bestIndex]),
            "coefA": 0.0,
            "coefB": 0.0,
            "coefC": 0.0,
        }

        if len(freqArray) < 3:
            result["status"] = "insufficient_points"
            return result
        if bestIndex <= 0 or bestIndex >= (len(freqArray) - 1):
            result["status"] = "edge_point"
            return result

        x = np.asarray(
            [freqArray[bestIndex - 1], freqArray[bestIndex], freqArray[bestIndex + 1]],
            dtype=np.float64,
        )
        y = np.asarray(
            [peakArray[bestIndex - 1], peakArray[bestIndex], peakArray[bestIndex + 1]],
            dtype=np.float64,
        )
        localIsMax = bool(y[1] >= y[0] and y[1] >= y[2])

        # Solve quadratic in local coordinates t around x0 to improve stability:
        # y(t) = a*t^2 + b*t + c, with t in {-hL, 0, hR}.
        x0 = float(x[1])
        hL = float(x0 - x[0])
        hR = float(x[2] - x0)
        if hL <= 0.0 or hR <= 0.0:
            result["status"] = "invalid_spacing"
            return result

        cLocal = float(y[1])
        mat = np.asarray(
            [
                [hL * hL, -hL],
                [hR * hR, hR],
            ],
            dtype=np.float64,
        )
        rhs = np.asarray(
            [
                float(y[0] - cLocal),
                float(y[2] - cLocal),
            ],
            dtype=np.float64,
        )
        try:
            sol = np.linalg.solve(mat, rhs)
        except np.linalg.LinAlgError:
            result["status"] = "singular_system"
            return result

        aLocal = float(sol[0])
        bLocal = float(sol[1])

        # Store global polynomial coefficients for traceability.
        a = aLocal
        b = -2.0 * aLocal * x0 + bLocal
        c = aLocal * x0 * x0 - bLocal * x0 + cLocal
        result["coefA"] = float(a)
        result["coefB"] = float(b)
        result["coefC"] = float(c)

        # First try strict 3-point interpolation.
        if localIsMax and abs(aLocal) >= 1e-18 and aLocal < 0.0:
            tVertex = -bLocal / (2.0 * aLocal)
            xVertex = x0 + tVertex
            yVertex = aLocal * tVertex * tVertex + bLocal * tVertex + cLocal
            stepHz = abs(float(freqArray[1] - freqArray[0])) if len(freqArray) > 1 else 0.0
            inNeighbor = (x[0] - 0.5 * stepHz) <= xVertex <= (x[2] + 0.5 * stepHz)

            if inNeighbor:
                result["interpFreqHz"] = float(xVertex)
                result["interpPeak"] = float(yVertex)
                result["valid"] = True
                result["status"] = "ok"
                result["method"] = "three_point"
                return result

        # Fallback: weighted quadratic fit in a wider local window.
        windowRadius = 2
        left = max(0, bestIndex - windowRadius)
        right = min(len(freqArray), bestIndex + windowRadius + 1)
        if (right - left) < 5:
            if not localIsMax:
                result["status"] = "not_local_max"
            elif abs(aLocal) < 1e-18:
                result["status"] = "flat_quadratic"
            elif aLocal >= 0.0:
                result["status"] = "non_concave"
            else:
                result["status"] = "vertex_outside_neighbor"
            return result

        xw = np.asarray(freqArray[left:right], dtype=np.float64)
        yw = np.asarray(peakArray[left:right], dtype=np.float64)
        tw = xw - x0
        base = float(np.min(yw))
        weights = np.maximum(yw - base, 1e-6)

        design = np.column_stack((tw * tw, tw, np.ones_like(tw)))
        sqrtW = np.sqrt(weights)
        designW = design * sqrtW[:, None]
        targetW = yw * sqrtW
        try:
            coefLocal, *_ = np.linalg.lstsq(designW, targetW, rcond=None)
        except np.linalg.LinAlgError:
            result["status"] = "window_fit_failed"
            return result

        aFit = float(coefLocal[0])
        bFit = float(coefLocal[1])
        cFit = float(coefLocal[2])
        result["coefA"] = float(aFit)
        result["coefB"] = float(-2.0 * aFit * x0 + bFit)
        result["coefC"] = float(aFit * x0 * x0 - bFit * x0 + cFit)

        if abs(aFit) < 1e-18:
            result["status"] = "flat_window_fit"
            return result
        if aFit >= 0.0:
            result["status"] = "non_concave_window_fit"
            return result

        tVertexFit = -bFit / (2.0 * aFit)
        xVertexFit = x0 + tVertexFit
        yVertexFit = aFit * tVertexFit * tVertexFit + bFit * tVertexFit + cFit
        stepHz = abs(float(freqArray[1] - freqArray[0])) if len(freqArray) > 1 else 0.0
        inWindow = (xw[0] - 0.5 * stepHz) <= xVertexFit <= (xw[-1] + 0.5 * stepHz)
        if not inWindow:
            result["status"] = "vertex_outside_window"
            return result

        result["interpFreqHz"] = float(xVertexFit)
        result["interpPeak"] = float(yVertexFit)
        result["valid"] = True
        result["status"] = "ok_window_fit"
        result["method"] = "weighted_window_fit"
        return result

    def _runAdaptiveSubfineForNid2(
        self,
        nId2: int,
        startFreqHz: float,
        startStepHz: float,
    ) -> dict:
        maxIterations = int(self.adaptiveRefineIterations)
        currentCenter = float(startFreqHz)
        currentStep = max(1e-9, abs(float(startStepHz)))
        history: list[dict] = []
        validIterations = 0
        stopStatus = "max_iterations"
        stopAt = 0

        for iterIdx in range(1, maxIterations + 1):
            subStep = currentStep / 2.0
            freqArray = currentCenter + np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64) * subStep
            peakArray = np.zeros(5, dtype=np.float64)
            timingArray = np.zeros(5, dtype=np.int32)

            for i, freqHz in enumerate(freqArray):
                scoreArray = self._computeScoreArray(
                    rxSignal=self._rxSignal,
                    template=self._pssTemplates[nId2],
                    freqHz=float(freqHz),
                )
                timingIndex = int(np.argmax(scoreArray))
                peakArray[i] = float(scoreArray[timingIndex])
                timingArray[i] = int(timingIndex)

            bestGridIndex = int(np.argmax(peakArray))
            interp = self._parabolicInterpolateFinePeak(
                freqArray=freqArray,
                peakArray=peakArray,
                bestIndex=bestGridIndex,
            )

            item = {
                "iteration": int(iterIdx),
                "centerFreqHz": float(currentCenter),
                "stepHz": float(subStep),
                "gridBestFreqHz": float(freqArray[bestGridIndex]),
                "gridBestPeak": float(peakArray[bestGridIndex]),
                "gridBestTimingOffset": int(timingArray[bestGridIndex]),
                "interpValid": bool(interp["valid"]),
                "interpStatus": str(interp["status"]),
                "interpMethod": str(interp.get("method", "none")),
                "interpFreqHz": float(interp["interpFreqHz"]),
                "interpPeak": float(interp["interpPeak"]),
            }
            history.append(item)

            if not bool(interp["valid"]):
                stopStatus = str(interp["status"])
                stopAt = int(iterIdx)
                break

            currentCenter = float(interp["interpFreqHz"])
            currentStep = float(subStep)
            validIterations += 1

        if maxIterations == 0:
            stopStatus = "disabled"
        elif stopAt == 0:
            stopAt = int(maxIterations)

        finalScoreArray = self._computeScoreArray(
            rxSignal=self._rxSignal,
            template=self._pssTemplates[nId2],
            freqHz=float(currentCenter),
        )
        finalTimingOffset = int(np.argmax(finalScoreArray))
        finalPeakValue = float(finalScoreArray[finalTimingOffset])

        return {
            "nId2": int(nId2),
            "startFreqHz": float(startFreqHz),
            "startStepHz": float(startStepHz),
            "maxIterations": int(maxIterations),
            "validIterations": int(validIterations),
            "finalStatus": str(stopStatus),
            "stopAtIteration": int(stopAt),
            "finalFreqHz": float(currentCenter),
            "finalPeak": float(finalPeakValue),
            "finalTimingOffset": int(finalTimingOffset),
            "history": history,
        }

    def _runParallelSearch(
        self,
        tasks: list[tuple[int, int, float]],
        progress: _ProgressReporter,
        stageName: str,
    ) -> list[dict]:
        if not tasks:
            return []

        rxSignal = np.asarray(self._rxSignal, dtype=np.complex64)
        templates = self._pssTemplates
        results: list[dict] = []
        bestPeak = -1.0

        progress.startStage(stageName, len(tasks))

        if self.maxWorkers == 1:
            _initWorker(rxSignal, templates, self.sampleRate, self.scoreMode, self._windowEnergySqrt)
            for task in tasks:
                result = _searchSingleFrequency(task)
                localBest = max(v[0] for v in result["peaks"].values())
                bestPeak = max(bestPeak, localBest)
                progress.advance(1, bestPeak=bestPeak)
                results.append(result)
            progress.endStage()
            return results

        if self.parallelMode == "process":
            try:
                executor = ProcessPoolExecutor(
                    max_workers=self.maxWorkers,
                    initializer=_initWorker,
                    initargs=(rxSignal, templates, self.sampleRate, self.scoreMode, self._windowEnergySqrt),
                )
            except PermissionError:
                logger.warning("ProcessPool unavailable in this environment, fallback to thread mode")
                _initWorker(rxSignal, templates, self.sampleRate, self.scoreMode, self._windowEnergySqrt)
                executor = ThreadPoolExecutor(max_workers=self.maxWorkers)
        else:
            _initWorker(rxSignal, templates, self.sampleRate, self.scoreMode, self._windowEnergySqrt)
            executor = ThreadPoolExecutor(max_workers=self.maxWorkers)

        with executor:
            futureMap = {executor.submit(_searchSingleFrequency, task): task for task in tasks}
            for future in as_completed(futureMap):
                result = future.result()
                localBest = max(v[0] for v in result["peaks"].values())
                bestPeak = max(bestPeak, localBest)
                progress.advance(1, bestPeak=bestPeak)
                results.append(result)

        progress.endStage()
        return results

    def detectPss(self, rxSignal: np.ndarray, validGscnList: list[tuple[int, float]]) -> dict:
        if len(validGscnList) == 0:
            raise ValueError("validGscnList is empty")

        self._rxSignal = np.asarray(rxSignal, dtype=np.complex64)
        templateLength = len(next(iter(self._pssTemplates.values())))
        if len(self._rxSignal) < templateLength:
            raise ValueError(f"Signal too short for correlation: len(rxSignal)={len(self._rxSignal)}, templateLen={templateLength}")

        if self.scoreMode == "ncc":
            self._windowEnergySqrt = self._computeWindowEnergySqrt(self._rxSignal, templateLength)
        else:
            self._windowEnergySqrt = None
        startAll = time.time()
        logger.info(
            f"PSS score mode: {self.scoreMode}, fineRange=+/-{self.fineSearchRangeHz:.0f}Hz, fineStep={self.fineStepHz:.0f}Hz"
        )
        logger.info(
            f"PSS Stage4: coarse={len(validGscnList)} candidates, "
            f"fineRange=±{self.fineSearchRangeHz:.0f}Hz, fineStep={self.fineStepHz:.0f}Hz, "
            f"workers={self.maxWorkers}, parallel={self.parallelMode}, progress={self.progressMode}"
        )

        with _ProgressReporter(mode=self.progressMode, minLogIntervalSec=self.progressRefreshSec) as progress:
            coarseTasks = [(idx, int(gscn), float(freqHz)) for idx, (gscn, freqHz) in enumerate(validGscnList)]
            coarseResults = self._runParallelSearch(coarseTasks, progress, "Stage4-Coarse")

            searchMatrix = np.zeros((len(validGscnList), 3), dtype=np.float32)
            coarseBest = {
                "gscn": int(validGscnList[0][0]),
                "freqOffset": float(validGscnList[0][1]),
                "nId2": 0,
                "timingOffset": 0,
                "peakValue": -1.0,
            }
            for result in coarseResults:
                taskIndex = int(result["taskIndex"])
                gscn = int(result["gscn"])
                freqHz = float(result["freqHz"])
                for nId2 in (0, 1, 2):
                    peakValue, peakIndex = result["peaks"][nId2]
                    searchMatrix[taskIndex, nId2] = float(peakValue)
                    if peakValue > coarseBest["peakValue"]:
                        coarseBest = {
                            "gscn": gscn,
                            "freqOffset": freqHz,
                            "nId2": int(nId2),
                            "timingOffset": int(peakIndex),
                            "peakValue": float(peakValue),
                        }

            fineFreqList = self._buildFineSearchFreqs(float(coarseBest["freqOffset"]))
            fineTasks = [(idx, int(coarseBest["gscn"]), float(freq)) for idx, freq in enumerate(fineFreqList)]
            fineResults = self._runParallelSearch(fineTasks, progress, "Stage4-Fine")

        fineFreqArray = np.asarray(fineFreqList, dtype=np.float64)
        finePeakMatrix = np.full((len(fineFreqArray), 3), -np.inf, dtype=np.float64)
        for result in fineResults:
            taskIndex = int(result["taskIndex"])
            for nId2 in (0, 1, 2):
                finePeakMatrix[taskIndex, nId2] = float(result["peaks"][nId2][0])

        nId2BestMap: dict[int, dict] = {
            0: {"nId2": 0, "gscn": int(coarseBest["gscn"]), "freqOffset": float(coarseBest["freqOffset"]), "timingOffset": 0, "peakValue": -1.0},
            1: {"nId2": 1, "gscn": int(coarseBest["gscn"]), "freqOffset": float(coarseBest["freqOffset"]), "timingOffset": 0, "peakValue": -1.0},
            2: {"nId2": 2, "gscn": int(coarseBest["gscn"]), "freqOffset": float(coarseBest["freqOffset"]), "timingOffset": 0, "peakValue": -1.0},
        }

        for result in fineResults:
            freqHz = float(result["freqHz"])
            for nId2 in (0, 1, 2):
                peakValue, peakIndex = result["peaks"][nId2]
                if float(peakValue) > float(nId2BestMap[nId2]["peakValue"]):
                    nId2BestMap[nId2] = {
                        "nId2": int(nId2),
                        "gscn": int(coarseBest["gscn"]),
                        "freqOffset": float(freqHz),
                        "timingOffset": int(peakIndex),
                        "peakValue": float(peakValue),
                    }

        parabolicPerNId2 = []
        parabolicMap: dict[int, dict] = {}
        for nId2 in (0, 1, 2):
            bestIndex = int(np.argmax(finePeakMatrix[:, nId2]))
            interp = self._parabolicInterpolateFinePeak(
                freqArray=fineFreqArray,
                peakArray=finePeakMatrix[:, nId2],
                bestIndex=bestIndex,
            )
            interpItem = {
                "nId2": int(nId2),
                "status": str(interp["status"]),
                "valid": bool(interp["valid"]),
                "method": str(interp.get("method", "none")),
                "gridBestFreqHz": float(interp["gridBestFreqHz"]),
                "gridBestPeak": float(interp["gridBestPeak"]),
                "interpFreqHz": float(interp["interpFreqHz"]),
                "interpPeak": float(interp["interpPeak"]),
                "coefA": float(interp["coefA"]),
                "coefB": float(interp["coefB"]),
                "coefC": float(interp["coefC"]),
            }
            parabolicPerNId2.append(interpItem)
            parabolicMap[int(nId2)] = interpItem

        adaptivePerNId2 = []
        adaptiveMap: dict[int, dict] = {}
        for nId2 in (0, 1, 2):
            item = nId2BestMap[nId2]
            item["freqOffsetGrid"] = float(item["freqOffset"])
            interpItem = parabolicMap[int(nId2)]
            if bool(interpItem["valid"]):
                startFreqHz = float(interpItem["interpFreqHz"])
                adaptive = self._runAdaptiveSubfineForNid2(
                    nId2=int(nId2),
                    startFreqHz=startFreqHz,
                    startStepHz=float(self.fineStepHz),
                )
            else:
                startFreqHz = float(item["freqOffset"])
                scoreArray = self._computeScoreArray(
                    rxSignal=self._rxSignal,
                    template=self._pssTemplates[nId2],
                    freqHz=startFreqHz,
                )
                timingOffset = int(np.argmax(scoreArray))
                peakValue = float(scoreArray[timingOffset])
                adaptive = {
                    "nId2": int(nId2),
                    "startFreqHz": float(startFreqHz),
                    "startStepHz": float(self.fineStepHz),
                    "maxIterations": int(self.adaptiveRefineIterations),
                    "validIterations": 0,
                    "finalStatus": f"initial_{interpItem['status']}",
                    "stopAtIteration": 0,
                    "finalFreqHz": float(startFreqHz),
                    "finalPeak": float(peakValue),
                    "finalTimingOffset": int(timingOffset),
                    "history": [],
                }

            adaptivePerNId2.append(adaptive)
            adaptiveMap[int(nId2)] = adaptive
            item["freqOffset"] = float(adaptive["finalFreqHz"])
            item["timingOffset"] = int(adaptive["finalTimingOffset"])
            item["peakValue"] = float(adaptive["finalPeak"])

        for nId2 in (0, 1, 2):
            item = nId2BestMap[nId2]
            corrArray = self._computeScoreArray(
                rxSignal=self._rxSignal,
                template=self._pssTemplates[nId2],
                freqHz=float(item["freqOffset"]),
            )
            item["corrArray"] = corrArray

        bestItem = max((nId2BestMap[0], nId2BestMap[1], nId2BestMap[2]), key=lambda x: x["peakValue"])
        finalParabolic = parabolicMap[int(bestItem["nId2"])]
        finalAdaptive = adaptiveMap[int(bestItem["nId2"])]
        finalParabolicFreqHz = (
            float(finalParabolic["interpFreqHz"])
            if bool(finalParabolic["valid"])
            else float(bestItem.get("freqOffsetGrid", bestItem["freqOffset"]))
        )
        finalRefinedFreqHz = float(finalAdaptive["finalFreqHz"])
        totalCorrCount = (len(validGscnList) + len(fineFreqList)) * 3
        elapsedAll = time.time() - startAll
        logger.info(
            f"PSS Stage4 done: totalCorr={totalCorrCount}, elapsed={elapsedAll:.2f}s, "
            f"throughput={totalCorrCount / max(1e-9, elapsedAll):.2f} corr/s"
        )
        logger.info(
            "Parabolic CFO refinement: "
            f"nId2={bestItem['nId2']}, status={finalParabolic['status']}, "
            f"grid={float(bestItem.get('freqOffsetGrid', bestItem['freqOffset'])):.3f} Hz -> interp={finalParabolicFreqHz:.3f} Hz"
        )
        logger.info(
            "Adaptive sub-fine refinement: "
            f"nId2={bestItem['nId2']}, validIters={finalAdaptive['validIterations']}/{finalAdaptive['maxIterations']}, "
            f"status={finalAdaptive['finalStatus']}, refinedFreq={finalRefinedFreqHz:.3f} Hz, "
            f"timing={finalAdaptive['finalTimingOffset']}, peak={finalAdaptive['finalPeak']:.6f}"
        )

        return {
            "timingOffset": int(bestItem["timingOffset"]),
            "nId2": int(bestItem["nId2"]),
            "freqOffset": float(bestItem["freqOffset"]),
            "freqOffsetParabolic": float(finalRefinedFreqHz),
            "gscn": int(bestItem["gscn"]),
            "peakValue": float(bestItem["peakValue"]),
            "corrArray": np.asarray(bestItem["corrArray"]),
            "scoreMode": str(self.scoreMode),
            "searchMatrix": searchMatrix,
            "coarseBest": coarseBest,
            "fineSearch": {
                "centerFreqHz": float(coarseBest["freqOffset"]),
                "rangeHz": float(self.fineSearchRangeHz),
                "stepHz": float(self.fineStepHz),
                "points": int(len(fineFreqList)),
            },
            "nId2BestResults": [
                nId2BestMap[0],
                nId2BestMap[1],
                nId2BestMap[2],
            ],
            "parabolicRefinement": {
                "method": "quadratic_three_point_with_weighted_window_fallback",
                "stepHz": float(self.fineStepHz),
                "finalNId2": int(bestItem["nId2"]),
                "finalGridFreqHz": float(bestItem.get("freqOffsetGrid", bestItem["freqOffset"])),
                "finalInterpFreqHz": float(finalParabolicFreqHz),
                "finalInterpValid": bool(finalParabolic["valid"]),
                "finalStatus": str(finalParabolic["status"]),
                "finalMethod": str(finalParabolic.get("method", "none")),
                "perNId2": parabolicPerNId2,
            },
            "adaptiveRefinement": {
                "method": "iterative_subfine_research",
                "maxIterations": int(self.adaptiveRefineIterations),
                "finalNId2": int(bestItem["nId2"]),
                "finalRefinedFreqHz": float(finalRefinedFreqHz),
                "finalStatus": str(finalAdaptive["finalStatus"]),
                "finalValidIterations": int(finalAdaptive["validIterations"]),
                "perNId2": adaptivePerNId2,
            },
            "peakAnalysis": None,
        }
