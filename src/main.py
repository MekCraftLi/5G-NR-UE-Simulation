"""SS/PBCH 端到端接收流程与任务交接说明。

本文件是主接收机入口，串起 `PSS -> SSS -> PBCH -> BCH/MIB`。这里的注释
故意比普通业务代码更详细：它是任务交接入口，后续接手者应先沿着本文件的
变量流向阅读，再进入各阶段模块。

规范锚点：
- TS 38.101-1 Clause 5.4.3: 同步栅格与 SS block 捕获。
- TS 38.211 Clause 7.4.2.1: N_ID_cell = 3 * N_ID_1 + N_ID_2。
- TS 38.211 Clause 7.4.2.2: PSS 序列，长度 127，由 N_ID_2 决定。
- TS 38.211 Clause 7.4.2.3: SSS 序列，长度 127，由 N_ID_1 和 N_ID_2 决定。
- TS 38.211 Clause 7.4.3.1: SS/PBCH block 为 4 个 OFDM 符号、240 个子载波；
  PSS 位于符号 0 的 k=56..182，SSS 位于符号 2 的 k=56..182，PBCH/DM-RS
  位于符号 1、2、3 的指定 RE。
- TS 38.211 Clause 7.4.1.4: PBCH DM-RS 序列和 RE 映射，用于信道估计。
- TS 38.213 Clause 4.1: SS/PBCH block 时域候选位置由 SS block pattern 约束。

主数据流：
1. `_loadRxSignal()` 读取 `data/rxSignal.npy` 或 `data/txs0.npy`，输出
   `rxSignal: np.ndarray`，这是后续所有阶段共享的复基带 IQ 样本。
2. `createSsbConfig()` 和 `CpManager` 根据 `cfg.sampleRate/cfg.scs` 推导
   `ssbConfig`、FFT size、normal CP、long CP；这些 OFDM 几何参数必须同时
   供 PSS 模板、SSS 滑窗和 PBCH FFT 窗口使用。
3. `detectBandwidth()` 输出 `bwEstimate`，主流程只用它确定 PSS 频偏搜索范围
   `freqMinHz/freqMaxHz` 并做 SCS 配置一致性提醒。
4. `PssBasebandSearcher.searchAdaptive(rxSignal)` 输出 `pssResult`：
   `nId2` 来自 PSS，`timingOffset` 是 PSS 符号 CP 起点，`freqOffsetParabolic`
   是基带频偏初估，`corrArray` 交给可视化。随后 PSS 后选会把
   `ssbSubcarrierOffset` 和更稳定的 `timingOffset` 写回同一个 `pssResult`。
5. PSS 后的半符号 FFO、PSS-LS、CP 相位估计只补充频偏证据；主线把
   `totalCfoHz` 作为默认 SSS/PBCH 频偏补偿基准。
6. `SssDetector.detectSss(rxSignal, pssResult, freqCompHz)` 对每个
   `sssOffsetCandidates` 构造独立 SSB 频域偏移配置，输出 `candidateSss`。
   主流程选择 `sssResult` 并写入 `pssResult["sssResult"]`，其中
   `nId1/nIdCell/verifiedSymbolStart/verifiedFreqCompHz` 是 PBCH 的输入锚点。
7. `PbchDecoder.decodePbch(rxSignal, sssResult, pssResult, freqCompHz)` 使用
   SSS 验证出的物理层小区 ID、SSB 定时、SSB 子载波偏移和频偏补偿提取
   SS/PBCH block，生成 PBCH DM-RS，做信道估计与 QPSK 硬判决，输出
   `pbchResult`；其中 `pbchEq/hardBits/noiseVarEstimate/iSsbBar` 交给 BCH。
8. `PbchBchDecoder.decode(pbchEq, nIdCell, iSsbBar, noiseVar)` 尝试官方
   MATLAB 5G Toolbox 链路 `nrPBCHDecode -> nrBCHDecode`，不可用时返回
   Python 诊断后备链路。结果写入 `pssResult["pbchBchResult"]`。

交接约定：
- 频偏变量必须带 `Hz` 后缀；`cfg.scs` 是子载波间隔，不是频偏。
- `ssbSubcarrierOffset` 是频域子载波平移，不能和频偏补偿混用。
- `timingOffset` / `bestSymbolStart` / `ssbStart` 都是样本索引，但含义不同：
  PSS CP 起点、SSS 符号 CP 起点、SS/PBCH block 符号 0 CP 起点。
- 本文件保存的 `pssResult` 是总汇交接对象：PSS、SSS、PBCH、BCH 的摘要都
  会逐步挂到这个 dict 上，最终由可视化和 JSON 输出消费。
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.bandwidthEstimator import detectBandwidth, inferScs
from common.config import createSsbConfig, createSsbConfigWithOverrides
from common.cpManager import CpManager
from pss.pssBasebandSearcher import PssBasebandSearcher
from pss.cpFreqOffsetEstimator import CpFreqOffsetEstimator
from pss.pssFreqOffsetEstimator import PssFreqOffsetEstimator
from pss.pssTemplateFactory import generatePssSequence
from pss.pssVisualizer import PssVisualizer
from pss.scsDetectVisualizer import plotScsDetection
from pbch.pbchBchDecoder import PbchBchDecoder
from pbch.pbchDecoder import PbchDecoder
from sss.sssDetector import SssDetector
from sss.sssVisualizer import SssVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeConfig:
    sampleRate: float
    scs: int
    prbCount: int
    inputPath: str | None = None
    outputPrefix: str = ""
    pssWorkers: int | None = None
    pssParallelMode: str = "process"
    pssScoreMode: str = "raw"
    pssAdaptiveIterations: int = 5
    pssLsqMinR2: float = 0.8
    pssCoarseStepHz: float = 1000.0
    pssMediumStepHz: float = 100.0
    pssFineStepHz: float = 15.0
    pssCenterFreqHz: float | None = None
    pssSearchHalfRangeHz: float | None = None
    pssTimingCenterSample: int | None = None
    pssTimingHalfRangeSamples: int | None = None
    pssAllowLongCpPostSelect: bool = False
    pssUseGpu: bool = True
    sssEnable: bool = True
    pssOnly: bool = False
    sssFreqBaseMode: str = "pss"
    sssOffsetCandidates: tuple[int, ...] = (-48, -7, 0, 16, 36)
    sssSearchStartSymbol: float = 0.5
    sssSearchEndSymbol: float = 2.5
    sssSearchStepSamples: int = 1
    pbchEnable: bool = True
    pbchResidualFreqSearchHz: float = 4000.0
    pbchResidualFreqStepHz: float = 100.0
    pbchIssbCandidates: tuple[int, ...] = tuple(range(8))
    pbchCpProfiles: tuple[str, ...] = ("all_normal", "slot_head")
    pbchSssCandidateLimit: int = 10
    progressMode: str = "auto"
    progressRefreshSec: float = 0.7


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="5G NR PSS blind search with dynamic config loading."
    )
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config file")
    parser.add_argument("--sample-rate", type=float, default=None, help="Sample rate in Hz")
    parser.add_argument("--scs", type=int, default=None, help="Subcarrier spacing in Hz")
    parser.add_argument("--prb-count", type=int, default=None, help="Number of PRBs")
    parser.add_argument("--input-path", type=str, default=None, help="Input .npy signal path")
    parser.add_argument("--output-prefix", type=str, default=None, help="Output file prefix")
    parser.add_argument("--pss-workers", type=int, default=None, help="PSS detector worker count")
    parser.add_argument(
        "--pss-parallel-mode",
        type=str,
        default=None,
        choices=["process", "thread"],
        help="PSS detector parallel mode",
    )
    parser.add_argument(
        "--pss-score-mode",
        type=str,
        default=None,
        choices=["raw", "ncc"],
        help="PSS detector score mode",
    )
    parser.add_argument(
        "--pss-adaptive-iterations",
        type=int,
        default=None,
        help="Adaptive sub-fine refinement iterations",
    )
    parser.add_argument(
        "--pss-lsq-min-r2",
        type=float,
        default=None,
        help="Minimum R^2 to accept PSS-assisted CFO LS fit",
    )
    parser.add_argument(
        "--pss-coarse-step",
        type=float,
        default=None,
        help="PSS adaptive search coarse step (Hz), default 1000",
    )
    parser.add_argument(
        "--pss-medium-step",
        type=float,
        default=None,
        help="PSS adaptive search medium step (Hz), default 100",
    )
    parser.add_argument(
        "--pss-fine-step",
        type=float,
        default=None,
        help="PSS adaptive search fine step (Hz), default 15",
    )
    parser.add_argument(
        "--pss-center-freq",
        type=float,
        default=None,
        help="Optional baseband center frequency (Hz) to narrow PSS search around",
    )
    parser.add_argument(
        "--pss-search-half-range",
        type=float,
        default=None,
        help="Optional half-range (Hz) around --pss-center-freq for PSS search",
    )
    parser.add_argument(
        "--pss-allow-long-cp-post-select",
        action="store_true",
        help="Allow long-CP candidates in PSS post-selection (debug use)",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=None,
        help="Use GPU for PSS search (default: auto-detect)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", default=None,
        help="Force CPU for PSS search",
    )
    parser.add_argument(
        "--disable-sss",
        action="store_true",
        help="Disable SSS sliding search stage",
    )
    parser.add_argument(
        "--pss-only",
        action="store_true",
        help="Run PSS stage only and skip SSS stage",
    )
    parser.add_argument(
        "--sss-freq-base-mode",
        type=str,
        default=None,
        choices=["pss", "scs", "zero"],
        help="Base frequency used for SSS/PBCH search",
    )
    parser.add_argument(
        "--sss-offset-candidates",
        type=str,
        default=None,
        help="Comma-separated SSB subcarrier offset candidates for blind SSS/PBCH scan",
    )
    parser.add_argument(
        "--sss-start-symbol",
        type=float,
        default=None,
        help="SSS sliding-search start (in OFDM-symbol units from PSS timing)",
    )
    parser.add_argument(
        "--sss-end-symbol",
        type=float,
        default=None,
        help="SSS sliding-search end (in OFDM-symbol units from PSS timing)",
    )
    parser.add_argument(
        "--sss-step-samples",
        type=int,
        default=None,
        help="SSS sliding-search step size in samples",
    )
    parser.add_argument(
        "--disable-pbch",
        action="store_true",
        help="Disable PBCH DM-RS equalization and constellation output",
    )
    parser.add_argument(
        "--pbch-residual-freq-search",
        type=float,
        default=None,
        help="PBCH residual CFO search half-range in Hz",
    )
    parser.add_argument(
        "--pbch-residual-freq-step",
        type=float,
        default=None,
        help="PBCH residual CFO search step in Hz",
    )
    parser.add_argument(
        "--pbch-issb-candidates",
        type=str,
        default=None,
        help="Comma-separated iSSB candidate list for PBCH search",
    )
    parser.add_argument(
        "--pbch-cp-profiles",
        type=str,
        default=None,
        help="Comma-separated PBCH CP profiles to keep, e.g. all_normal or slot_head",
    )
    parser.add_argument(
        "--pbch-sss-candidate-limit",
        type=int,
        default=None,
        help="How many top SSS candidates to forward to PBCH search",
    )
    parser.add_argument(
        "--progress-mode",
        type=str,
        default=None,
        choices=["auto", "rich", "log"],
        help="Progress display mode",
    )
    parser.add_argument(
        "--progress-refresh",
        type=float,
        default=None,
        help="Progress refresh interval in seconds (log mode)",
    )
    return parser.parse_args(argv)


def _loadConfigFile(configPath: str) -> dict[str, Any]:
    path = Path(configPath)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("YAML config requires PyYAML installed") from e
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError("Unsupported config format. Use .json/.yaml/.yml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Config file root must be an object/dict")
    return data


def _pick(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return default


def _parseCsvInts(text: str) -> tuple[int, ...]:
    values = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return tuple(values)


def _pssScoreAt(
    rxSignal: np.ndarray,
    sampleRate: float,
    nfft: int,
    cpLength: int,
    nId2: int,
    timingOffset: int,
    ssbSubcarrierOffset: int,
    freqHz: float,
) -> dict[str, Any]:
    pssSeq = generatePssSequence(int(nId2))
    pssFreq = np.zeros(int(nfft), dtype=np.complex64)
    pssFreq[int(nfft) // 2 - 63 + int(ssbSubcarrierOffset):int(nfft) // 2 + 64 + int(ssbSubcarrierOffset)] = pssSeq
    pssTimeTemplate = np.fft.ifft(np.fft.ifftshift(pssFreq)).astype(np.complex64)
    pssTimeTemplate /= max(float(np.linalg.norm(pssTimeTemplate)), 1e-12)

    usefulStart = int(timingOffset) + int(cpLength)
    usefulEnd = usefulStart + int(nfft)
    if usefulStart < 0 or usefulEnd > len(rxSignal):
        return {"valid": False, "score": 0.0, "timingOffset": int(timingOffset)}

    sampleIndex = np.arange(usefulStart, usefulEnd, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * float(freqHz) * sampleIndex / float(sampleRate)).astype(np.complex64)
    rx = np.asarray(rxSignal[usefulStart:usefulEnd], dtype=np.complex64) * phase
    corr = np.abs(np.vdot(pssTimeTemplate, rx))
    score = float(corr / max(float(np.linalg.norm(rx)), 1e-12))
    return {
        "valid": True,
        "score": score,
        "timingOffset": int(timingOffset),
        "cpLength": int(cpLength),
        "ssbSubcarrierOffset": int(ssbSubcarrierOffset),
    }


def _selectStablePssPostCandidate(
    candidates: list[dict[str, Any]],
    anchorTimingOffset: int | None = None,
    anchorOffset: int | None = None,
    scoreRelTolerance: float = 1e-4,
    scoreAbsTolerance: float = 1e-9,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates is empty")

    bestScore = max(float(item.get("score", -np.inf)) for item in candidates)
    tol = max(float(scoreAbsTolerance), abs(float(bestScore)) * float(scoreRelTolerance))
    pool = [item for item in candidates if float(item.get("score", -np.inf)) >= float(bestScore) - tol]
    if not pool:
        pool = [max(candidates, key=lambda x: float(x.get("score", -np.inf)))]

    anchorTiming = None if anchorTimingOffset is None else int(anchorTimingOffset)
    anchorK = None if anchorOffset is None else int(anchorOffset)

    def stableKey(item: dict[str, Any]) -> tuple:
        timing = int(item.get("timingOffset", 0))
        k = int(item.get("ssbSubcarrierOffset", 0))
        cpLen = int(item.get("cpLength", 0))
        cpKind = str(item.get("cpKind", ""))
        score = float(item.get("score", -np.inf))
        timingDist = abs(timing - anchorTiming) if anchorTiming is not None else abs(timing)
        offsetDist = abs(k - anchorK) if anchorK is not None else abs(k)
        return (
            int(timingDist),
            int(offsetDist),
            0 if cpKind == "long" else 1,
            int(cpLen),
            -float(score),
            int(k),
            int(timing),
        )

    return min(pool, key=stableKey)


def _selectStableSssCandidate(
    candidates: list[dict[str, Any]],
    anchorOffset: int | None = None,
    anchorSymbolStart: int | None = None,
    anchorFreqHz: float | None = None,
    scoreRelTolerance: float = 1e-4,
    scoreAbsTolerance: float = 1e-9,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("candidates is empty")

    def primaryScore(item: dict[str, Any]) -> float:
        return float(item.get("selectionScore", 0.0))

    bestScore = max(primaryScore(item) for item in candidates)
    tol = max(float(scoreAbsTolerance), abs(float(bestScore)) * float(scoreRelTolerance))
    pool = [item for item in candidates if primaryScore(item) >= float(bestScore) - tol]
    if not pool:
        pool = [max(candidates, key=primaryScore)]

    anchorK = None if anchorOffset is None else int(anchorOffset)
    anchorStart = None if anchorSymbolStart is None else int(anchorSymbolStart)

    def stableKey(item: dict[str, Any]) -> tuple:
        k = int(item.get("ssbSubcarrierOffset", 0))
        start = int(item.get("verifiedSymbolStart", item.get("bestSymbolStart", 0)))
        freq = float(item.get("verifiedFreqCompHz", item.get("freqCompHz", 0.0)))
        score = primaryScore(item)
        fdScore = float(item.get("verifiedFdScore", 0.0))
        tdScore = float(item.get("bestScore", 0.0))
        offsetDist = abs(k - anchorK) if anchorK is not None else abs(k)
        startDist = abs(start - anchorStart) if anchorStart is not None else abs(start)
        freqDist = abs(freq - float(anchorFreqHz)) if anchorFreqHz is not None else abs(freq)
        return (
            int(offsetDist),
            int(startDist),
            float(freqDist),
            -float(score),
            -float(fdScore),
            -float(tdScore),
            int(k),
            int(start),
        )

    return min(pool, key=stableKey)


def _buildRuntimeConfig(args: argparse.Namespace) -> RuntimeConfig:
    defaults = {
        "sampleRate": 30.72e6,
        "scs": 30000,
        "prbCount": 51,
        "inputPath": None,
        "outputPrefix": "",
        "pssWorkers": None,
        "pssParallelMode": "process",
        "pssScoreMode": "raw",
        "pssAdaptiveIterations": 5,
        "pssLsqMinR2": 0.8,
        "pssCoarseStepHz": 1000.0,
        "pssMediumStepHz": 100.0,
        "pssFineStepHz": 15.0,
        "pssCenterFreqHz": None,
        "pssSearchHalfRangeHz": None,
        "pssAllowLongCpPostSelect": False,
        "pssUseGpu": True,
        "sssEnable": True,
        "pssOnly": False,
        "sssFreqBaseMode": "pss",
        "sssOffsetCandidates": (-48, -7, 0, 16, 36),
        "sssSearchStartSymbol": 0.5,
        "sssSearchEndSymbol": 2.5,
        "sssSearchStepSamples": 1,
        "pbchEnable": True,
        "pbchResidualFreqSearchHz": 4000.0,
        "pbchResidualFreqStepHz": 100.0,
        "pbchIssbCandidates": tuple(range(8)),
        "pbchCpProfiles": ("all_normal", "slot_head"),
        "pbchSssCandidateLimit": 10,
        "progressMode": "auto",
        "progressRefreshSec": 0.7,
    }

    fileCfg: dict[str, Any] = {}
    if args.config:
        fileCfg = _loadConfigFile(args.config)

    sampleRate = float(
        args.sample_rate
        if args.sample_rate is not None
        else _pick(fileCfg, "sample_rate", "sampleRate", default=defaults["sampleRate"])
    )
    scs = int(
        args.scs
        if args.scs is not None
        else _pick(fileCfg, "scs", "subcarrier_spacing", "subcarrierSpacing", default=defaults["scs"])
    )
    prbCount = int(
        args.prb_count
        if args.prb_count is not None
        else _pick(fileCfg, "prb_count", "prbCount", "rb_count", default=defaults["prbCount"])
    )
    inputPath = (
        args.input_path
        if args.input_path is not None
        else _pick(fileCfg, "input_path", "inputPath", "signal_path", default=defaults["inputPath"])
    )
    outputPrefix = (
        args.output_prefix
        if args.output_prefix is not None
        else _pick(fileCfg, "output_prefix", "outputPrefix", default=defaults["outputPrefix"])
    )
    pssWorkersRaw = (
        args.pss_workers
        if args.pss_workers is not None
        else _pick(fileCfg, "pss_workers", "pssWorkers", default=defaults["pssWorkers"])
    )
    pssWorkers = int(pssWorkersRaw) if pssWorkersRaw is not None else None
    pssParallelMode = str(
        args.pss_parallel_mode
        if args.pss_parallel_mode is not None
        else _pick(fileCfg, "pss_parallel_mode", "pssParallelMode", default=defaults["pssParallelMode"])
    ).lower()
    pssScoreMode = str(
        args.pss_score_mode
        if args.pss_score_mode is not None
        else _pick(fileCfg, "pss_score_mode", "pssScoreMode", default=defaults["pssScoreMode"])
    ).lower()
    pssAdaptiveIterations = int(
        args.pss_adaptive_iterations
        if args.pss_adaptive_iterations is not None
        else _pick(
            fileCfg,
            "pss_adaptive_iterations",
            "pssAdaptiveIterations",
            default=defaults["pssAdaptiveIterations"],
        )
    )
    pssLsqMinR2 = float(
        args.pss_lsq_min_r2
        if args.pss_lsq_min_r2 is not None
        else _pick(fileCfg, "pss_lsq_min_r2", "pssLsqMinR2", default=defaults["pssLsqMinR2"])
    )
    pssCoarseStepHz = float(
        args.pss_coarse_step
        if args.pss_coarse_step is not None
        else _pick(fileCfg, "pss_coarse_step_hz", "pssCoarseStepHz", default=defaults["pssCoarseStepHz"])
    )
    pssMediumStepHz = float(
        args.pss_medium_step
        if args.pss_medium_step is not None
        else _pick(fileCfg, "pss_medium_step_hz", "pssMediumStepHz", default=defaults["pssMediumStepHz"])
    )
    pssFineStepHz = float(
        args.pss_fine_step
        if args.pss_fine_step is not None
        else _pick(fileCfg, "pss_fine_step_hz", "pssFineStepHz", default=defaults["pssFineStepHz"])
    )
    pssCenterFreqHzRaw = (
        args.pss_center_freq
        if args.pss_center_freq is not None
        else _pick(fileCfg, "pss_center_freq_hz", "pssCenterFreqHz", default=defaults["pssCenterFreqHz"])
    )
    pssCenterFreqHz = float(pssCenterFreqHzRaw) if pssCenterFreqHzRaw is not None else None
    pssSearchHalfRangeRaw = (
        args.pss_search_half_range
        if args.pss_search_half_range is not None
        else _pick(
            fileCfg,
            "pss_search_half_range_hz",
            "pssSearchHalfRangeHz",
            default=defaults["pssSearchHalfRangeHz"],
        )
    )
    pssSearchHalfRangeHz = float(pssSearchHalfRangeRaw) if pssSearchHalfRangeRaw is not None else None
    pssAllowLongCpPostSelect = bool(
        _pick(
            fileCfg,
            "pss_allow_long_cp_post_select",
            "pssAllowLongCpPostSelect",
            default=defaults["pssAllowLongCpPostSelect"],
        )
    )
    if args.pss_allow_long_cp_post_select:
        pssAllowLongCpPostSelect = True
    pssUseGpu = bool(
        _pick(fileCfg, "pss_use_gpu", "pssUseGpu", default=defaults["pssUseGpu"])
    )
    if args.use_gpu:
        pssUseGpu = True
    if args.no_gpu:
        pssUseGpu = False
    sssEnableCfg = _pick(fileCfg, "sss_enable", "sssEnable", default=defaults["sssEnable"])
    sssEnable = bool(sssEnableCfg)
    if args.disable_sss:
        sssEnable = False
    pssOnlyCfg = _pick(fileCfg, "pss_only", "pssOnly", default=defaults["pssOnly"])
    pssOnly = bool(pssOnlyCfg) or bool(args.pss_only)
    if pssOnly:
        sssEnable = False
    sssFreqBaseMode = str(
        args.sss_freq_base_mode
        if args.sss_freq_base_mode is not None
        else _pick(fileCfg, "sss_freq_base_mode", "sssFreqBaseMode", default=defaults["sssFreqBaseMode"])
    ).lower()
    sssOffsetCandidatesRaw = (
        args.sss_offset_candidates
        if args.sss_offset_candidates is not None
        else _pick(
            fileCfg,
            "sss_offset_candidates",
            "sssOffsetCandidates",
            default=defaults["sssOffsetCandidates"],
        )
    )
    if isinstance(sssOffsetCandidatesRaw, str):
        sssOffsetCandidates = _parseCsvInts(sssOffsetCandidatesRaw)
    elif isinstance(sssOffsetCandidatesRaw, (list, tuple)):
        sssOffsetCandidates = tuple(int(x) for x in sssOffsetCandidatesRaw)
    else:
        sssOffsetCandidates = tuple(int(x) for x in defaults["sssOffsetCandidates"])
    sssSearchStartSymbol = float(
        args.sss_start_symbol
        if args.sss_start_symbol is not None
        else _pick(
            fileCfg,
            "sss_start_symbol",
            "sssSearchStartSymbol",
            default=defaults["sssSearchStartSymbol"],
        )
    )
    sssSearchEndSymbol = float(
        args.sss_end_symbol
        if args.sss_end_symbol is not None
        else _pick(
            fileCfg,
            "sss_end_symbol",
            "sssSearchEndSymbol",
            default=defaults["sssSearchEndSymbol"],
        )
    )
    sssSearchStepSamples = int(
        args.sss_step_samples
        if args.sss_step_samples is not None
        else _pick(
            fileCfg,
            "sss_step_samples",
            "sssSearchStepSamples",
            default=defaults["sssSearchStepSamples"],
        )
    )
    pbchEnableCfg = _pick(fileCfg, "pbch_enable", "pbchEnable", default=defaults["pbchEnable"])
    pbchEnable = bool(pbchEnableCfg)
    if args.disable_pbch:
        pbchEnable = False
    pbchResidualFreqSearchHz = float(
        args.pbch_residual_freq_search
        if args.pbch_residual_freq_search is not None
        else _pick(
            fileCfg,
            "pbch_residual_freq_search_hz",
            "pbchResidualFreqSearchHz",
            default=defaults["pbchResidualFreqSearchHz"],
        )
    )
    pbchResidualFreqStepHz = float(
        args.pbch_residual_freq_step
        if args.pbch_residual_freq_step is not None
        else _pick(
            fileCfg,
            "pbch_residual_freq_step_hz",
            "pbchResidualFreqStepHz",
            default=defaults["pbchResidualFreqStepHz"],
        )
    )
    pbchIssbCandidatesRaw = (
        args.pbch_issb_candidates
        if args.pbch_issb_candidates is not None
        else _pick(fileCfg, "pbch_issb_candidates", "pbchIssbCandidates", default=defaults["pbchIssbCandidates"])
    )
    if isinstance(pbchIssbCandidatesRaw, str):
        pbchIssbCandidates = _parseCsvInts(pbchIssbCandidatesRaw)
    elif isinstance(pbchIssbCandidatesRaw, (list, tuple)):
        pbchIssbCandidates = tuple(int(x) for x in pbchIssbCandidatesRaw)
    else:
        pbchIssbCandidates = tuple(int(x) for x in defaults["pbchIssbCandidates"])
    pbchCpProfilesRaw = (
        args.pbch_cp_profiles
        if args.pbch_cp_profiles is not None
        else _pick(fileCfg, "pbch_cp_profiles", "pbchCpProfiles", default=defaults["pbchCpProfiles"])
    )
    if isinstance(pbchCpProfilesRaw, str):
        pbchCpProfiles = tuple(part.strip() for part in pbchCpProfilesRaw.split(",") if part.strip())
    elif isinstance(pbchCpProfilesRaw, (list, tuple)):
        pbchCpProfiles = tuple(str(x).strip() for x in pbchCpProfilesRaw if str(x).strip())
    else:
        pbchCpProfiles = tuple(str(x) for x in defaults["pbchCpProfiles"])
    pbchSssCandidateLimit = int(
        args.pbch_sss_candidate_limit
        if args.pbch_sss_candidate_limit is not None
        else _pick(
            fileCfg,
            "pbch_sss_candidate_limit",
            "pbchSssCandidateLimit",
            default=defaults["pbchSssCandidateLimit"],
        )
    )
    progressMode = str(
        args.progress_mode
        if args.progress_mode is not None
        else _pick(fileCfg, "progress_mode", "progressMode", default=defaults["progressMode"])
    ).lower()
    progressRefreshSec = float(
        args.progress_refresh
        if args.progress_refresh is not None
        else _pick(fileCfg, "progress_refresh", "progressRefreshSec", default=defaults["progressRefreshSec"])
    )

    if sampleRate <= 0:
        raise ValueError("sampleRate must be > 0")
    if scs <= 0:
        raise ValueError("scs must be > 0")
    if prbCount <= 0:
        raise ValueError("prbCount must be > 0")
    if pssWorkers is not None and pssWorkers <= 0:
        raise ValueError("pssWorkers must be > 0 when provided")
    if pssParallelMode not in ("process", "thread"):
        raise ValueError("pssParallelMode must be one of: process, thread")
    if pssScoreMode not in ("raw", "ncc"):
        raise ValueError("pssScoreMode must be one of: raw, ncc")
    if pssAdaptiveIterations < 0:
        raise ValueError("pssAdaptiveIterations must be >= 0")
    if pssLsqMinR2 < 0.0 or pssLsqMinR2 > 1.0:
        raise ValueError("pssLsqMinR2 must be in [0, 1]")
    if pssCoarseStepHz <= 0:
        raise ValueError("pssCoarseStepHz must be > 0")
    if pssMediumStepHz <= 0:
        raise ValueError("pssMediumStepHz must be > 0")
    if pssFineStepHz <= 0:
        raise ValueError("pssFineStepHz must be > 0")
    if not (pssFineStepHz < pssMediumStepHz < pssCoarseStepHz):
        raise ValueError("step sizes must satisfy: fine < medium < coarse")
    if (pssCenterFreqHz is None) != (pssSearchHalfRangeHz is None):
        raise ValueError("pssCenterFreqHz and pssSearchHalfRangeHz must be provided together")
    if pssSearchHalfRangeHz is not None and pssSearchHalfRangeHz <= 0:
        raise ValueError("pssSearchHalfRangeHz must be > 0")
    if sssFreqBaseMode not in ("pss", "scs", "zero"):
        raise ValueError("sssFreqBaseMode must be one of: pss, scs, zero")
    if sssSearchEndSymbol <= sssSearchStartSymbol:
        raise ValueError("sssSearchEndSymbol must be > sssSearchStartSymbol")
    if sssSearchStepSamples <= 0:
        raise ValueError("sssSearchStepSamples must be > 0")
    if pbchResidualFreqSearchHz < 0:
        raise ValueError("pbchResidualFreqSearchHz must be >= 0")
    if pbchResidualFreqStepHz <= 0:
        raise ValueError("pbchResidualFreqStepHz must be > 0")
    if not pbchIssbCandidates:
        raise ValueError("pbchIssbCandidates must not be empty")
    if not pbchCpProfiles:
        raise ValueError("pbchCpProfiles must not be empty")
    validPbchCpProfiles = {"all_normal", "slot_head"}
    if any(profile not in validPbchCpProfiles for profile in pbchCpProfiles):
        raise ValueError("pbchCpProfiles must be a subset of: all_normal, slot_head")
    if pbchSssCandidateLimit <= 0:
        raise ValueError("pbchSssCandidateLimit must be > 0")
    if progressMode not in ("auto", "rich", "log"):
        raise ValueError("progressMode must be one of: auto, rich, log")
    if progressRefreshSec <= 0:
        raise ValueError("progressRefreshSec must be > 0")

    return RuntimeConfig(
        sampleRate=sampleRate,
        scs=scs,
        prbCount=prbCount,
        inputPath=inputPath,
        outputPrefix=outputPrefix or "",
        pssWorkers=pssWorkers,
        pssParallelMode=pssParallelMode,
        pssScoreMode=pssScoreMode,
        pssAdaptiveIterations=pssAdaptiveIterations,
        pssLsqMinR2=pssLsqMinR2,
        pssCoarseStepHz=pssCoarseStepHz,
        pssMediumStepHz=pssMediumStepHz,
        pssFineStepHz=pssFineStepHz,
        pssCenterFreqHz=pssCenterFreqHz,
        pssSearchHalfRangeHz=pssSearchHalfRangeHz,
        pssAllowLongCpPostSelect=pssAllowLongCpPostSelect,
        pssUseGpu=pssUseGpu,
        sssEnable=sssEnable,
        pssOnly=pssOnly,
        sssFreqBaseMode=sssFreqBaseMode,
        sssOffsetCandidates=sssOffsetCandidates,
        sssSearchStartSymbol=sssSearchStartSymbol,
        sssSearchEndSymbol=sssSearchEndSymbol,
        sssSearchStepSamples=sssSearchStepSamples,
        pbchEnable=pbchEnable,
        pbchResidualFreqSearchHz=pbchResidualFreqSearchHz,
        pbchResidualFreqStepHz=pbchResidualFreqStepHz,
        pbchIssbCandidates=pbchIssbCandidates,
        pbchCpProfiles=pbchCpProfiles,
        pbchSssCandidateLimit=pbchSssCandidateLimit,
        progressMode=progressMode,
        progressRefreshSec=progressRefreshSec,
    )


def _loadRxSignal(inputPath: str | None) -> np.ndarray:
    """读取所有接收阶段共用的 IQ 采样序列。

    数据交接：
    - 输入：命令行或配置文件中的可选 `inputPath`。
    - 输出：`rxSignal`，按采样顺序排列的一维复数 NumPy 数组。
    - 下游：CP/SCS 检测、PSS 搜索、SSS 检测、PBCH 解调。

    如果没有显式指定路径，盲接收流程优先读取 `data/rxSignal.npy`；
    `data/txs0.npy` 只作为 TX 参考烟测后备。该函数只读数据，不推断任何
    同步结果。
    """
    if inputPath:
        path = Path(inputPath)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input signal file not found: {path}")
        logger.info(f"Loading input signal: {path}")
        return np.load(str(path))

    dataDir = Path(__file__).resolve().parent.parent / "data"
    preferredPath = dataDir / "rxSignal.npy"
    fallbackPath = dataDir / "txs0.npy"

    if preferredPath.exists():
        logger.info(f"Loading input signal: {preferredPath}")
        return np.load(str(preferredPath))
    if fallbackPath.exists():
        logger.info(f"rxSignal.npy not found, fallback to: {fallbackPath}")
        return np.load(str(fallbackPath))

    raise FileNotFoundError("No input signal found: data/rxSignal.npy or data/txs0.npy")


def main(argv: list[str] | None = None):
    args = _parseArgs(argv)
    try:
        cfg = _buildRuntimeConfig(args)
    except Exception as e:
        logger.error(f"Config error: {e}")
        return

    logger.info(
        "Runtime config: "
        f"sampleRate={cfg.sampleRate}, scs={cfg.scs}, prbCount={cfg.prbCount}, "
        f"inputPath={cfg.inputPath}, outputPrefix={cfg.outputPrefix}, "
        f"pssWorkers={cfg.pssWorkers}, pssParallelMode={cfg.pssParallelMode}, "
        f"pssScoreMode={cfg.pssScoreMode}, pssAdaptiveIterations={cfg.pssAdaptiveIterations}, "
        f"pssLsqMinR2={cfg.pssLsqMinR2}, "
        f"pssCoarseStepHz={cfg.pssCoarseStepHz}, pssMediumStepHz={cfg.pssMediumStepHz}, "
        f"pssFineStepHz={cfg.pssFineStepHz}, "
        f"pssCenterFreqHz={cfg.pssCenterFreqHz}, pssSearchHalfRangeHz={cfg.pssSearchHalfRangeHz}, "
        f"pssAllowLongCpPostSelect={cfg.pssAllowLongCpPostSelect}, "
        f"pssUseGpu={cfg.pssUseGpu}, "
        f"sssEnable={cfg.sssEnable}, pssOnly={cfg.pssOnly}, sssFreqBaseMode={cfg.sssFreqBaseMode}, "
        f"sssOffsetCandidates={cfg.sssOffsetCandidates}, sssSearchStartSymbol={cfg.sssSearchStartSymbol}, "
        f"sssSearchEndSymbol={cfg.sssSearchEndSymbol}, sssSearchStepSamples={cfg.sssSearchStepSamples}, "
        f"pbchEnable={cfg.pbchEnable}, pbchResidualFreqSearchHz={cfg.pbchResidualFreqSearchHz}, "
        f"pbchResidualFreqStepHz={cfg.pbchResidualFreqStepHz}, "
        f"pbchIssbCandidates={cfg.pbchIssbCandidates}, pbchCpProfiles={cfg.pbchCpProfiles}, "
        f"pbchSssCandidateLimit={cfg.pbchSssCandidateLimit}, "
        f"progressMode={cfg.progressMode}, progressRefreshSec={cfg.progressRefreshSec}"
    )

    logger.info("1) Load signal")
    try:
        rxSignal = _loadRxSignal(cfg.inputPath)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    logger.info(f"Signal length: {len(rxSignal)} samples")

    logger.info("2) Derive OFDM parameters")
    # 数据流：`cfg.sampleRate/cfg.scs` -> `ssbConfig` 与 `cpManager`。
    # 这些参数定义后续所有阶段共用的 OFDM 采样几何：PSS 模板长度、
    # SSS 相对 PSS 的符号偏移、PBCH FFT 窗口。
    # 规范依据：TS 38.211 Clause 5.3.1（OFDM 基带信号生成）和
    # Clause 7.4.3.1（SS/PBCH block 包含 4 个 OFDM 符号）。
    ssbConfig = createSsbConfig(sampleRate=cfg.sampleRate, subcarrierSpacing=cfg.scs)
    cpManager = CpManager(ssbConfig)
    logger.info(f"FFT: {ssbConfig.FftSize}, NormalCP: {ssbConfig.NormalCpLength}, LongCP: {cpManager.longCpLength}")

    logger.info("3) Detect SCS and signal bandwidth via CP autocorrelation")
    # 数据流：`rxSignal` -> `bwEstimate` -> PSS 基带频偏搜索范围。
    # 这里的检测值只作为提醒和搜索范围依据；除非调用者改配置，否则实际
    # OFDM 模板仍按 `cfg.scs` 构造。
    bwEstimate = detectBandwidth(rxSignal, cfg.sampleRate)
    scsCandidates = inferScs(bwEstimate)

    detectedScs = bwEstimate.scsHz
    scsMatch = (int(cfg.scs) == int(detectedScs))
    # 搜索范围: 默认用检测到的 SSB 带宽；若提供目标附近范围，则只在局部搜索。
    if cfg.pssCenterFreqHz is not None and cfg.pssSearchHalfRangeHz is not None:
        freqMinHz = float(cfg.pssCenterFreqHz) - float(cfg.pssSearchHalfRangeHz)
        freqMaxHz = float(cfg.pssCenterFreqHz) + float(cfg.pssSearchHalfRangeHz)
        logger.info(
            "Override PSS search range by target vicinity: center=%.2f Hz, halfRange=%.2f Hz",
            float(cfg.pssCenterFreqHz),
            float(cfg.pssSearchHalfRangeHz),
        )
    else:
        freqMinHz = -bwEstimate.bandwidthHz / 2
        freqMaxHz = +bwEstimate.bandwidthHz / 2

    logger.info(
        f"Detected: SCS={detectedScs/1000:.0f}kHz, FFT={bwEstimate.fftSize}, "
        f"CP={bwEstimate.cpLength}, SSB BW={bwEstimate.bandwidthHz/1e6:.3f}MHz, "
        f"confidence={bwEstimate.confidence:.2f}"
    )
    logger.info(f"Search range: [{freqMinHz/1e6:.3f}, {freqMaxHz/1e6:.3f}] MHz")

    def _richWarn(title, body, style="yellow"):
        try:
            from rich.console import Console
            from rich.panel import Panel
            Console(stderr=True).print(Panel.fit(body, title=title, border_style=style))
        except ImportError:
            logger.warning(f"{title}: {body}")

    if bwEstimate.confidence < 0.5:
        _richWarn(
            "SCS Detection",
            f"[dim]CP 自相关置信度仅 {bwEstimate.confidence:.2f}，"
            f"SCS 推断可能不准确[/dim]\n"
            f"检测值: SCS={detectedScs/1000:.0f}kHz, "
            f"SSB BW={bwEstimate.bandwidthHz/1e6:.2f}MHz"
        )

    if not scsMatch:
        _richWarn(
            "SCS Validation",
            f"检测 SCS: [bold green]{detectedScs/1000:.0f} kHz[/bold green] "
            f"(SSB BW={bwEstimate.bandwidthHz/1e6:.2f} MHz)\n"
            f"指定 SCS: [bold red]{cfg.scs/1000:.0f} kHz[/bold red] "
            f"— [bold yellow]不匹配![/bold yellow]",
        )
    else:
        logger.info(f"SCS {cfg.scs/1000:.0f}kHz: matches detected SCS")

    plotScsDetection(rxSignal, cfg.sampleRate, outputPrefix=cfg.outputPrefix)

    logger.info("4) Run PSS adaptive baseband frequency search")
    # PSS 输入：
    #   `_loadRxSignal()` 读出的 `rxSignal`，步骤 2 得到的 `ssbConfig`，
    #   以及由 CP/带宽估计或命令行覆盖得到的 [freqMinHz, freqMaxHz]。
    # PSS 处理：
    #   PssBasebandSearcher 构造 3 个本地 PSS 时域模板（N_ID_2=0,1,2），
    #   对每个候选频偏先补偿整段信号，再用 FFT 卷积做相关，最后返回最强
    #   的（频偏、N_ID_2、定时）假设。
    # PSS 输出写入 `pssResult`：
    #   `timingOffset`        -> PSS 符号 CP 起点采样；
    #   `nId2`                -> 物理层小区 ID 的 N_ID_2 部分；
    #   `freqOffsetParabolic` -> 细化后的频偏估计，单位 Hz；
    #   `corrArray`           -> 用于画图的 PSS 定时相关曲线。
    # 规范依据：TS 38.211 Clause 7.4.2.2 和 Clause 7.4.3.1.1。
    pssSearcher = PssBasebandSearcher(
        config=ssbConfig,
        freqMinHz=freqMinHz,
        freqMaxHz=freqMaxHz,
        workers=cfg.pssWorkers,
        parallelMode=cfg.pssParallelMode,
        useGpu=cfg.pssUseGpu,
    )
    pssResult = pssSearcher.searchAdaptive(
        signal=rxSignal,
        coarseStepHz=cfg.pssCoarseStepHz,
        mediumStepHz=cfg.pssMediumStepHz,
        fineStepHz=cfg.pssFineStepHz,
        subfineIterations=cfg.pssAdaptiveIterations,
    )
    # `baseFreqForEstHz` 是 PSS 给下游的第一个频偏交接值。后续估计器
    # 可以细化或交叉验证它，但下游补偿量必须始终是 Hz 频率，不能把它
    # 和子载波间隔或子载波偏移混用。
    baseFreqForEstHz = float(pssResult.get("freqOffsetParabolic", pssResult["freqOffset"]))

    nId2 = int(pssResult["nId2"])
    timingInitial = int(pssResult["timingOffset"])
    normalCp = int(ssbConfig.NormalCpLength)
    longCp = int(cpManager.longCpLength)
    cpDelta = abs(longCp - normalCp)
    timingSeeds = {timingInitial}
    for item in pssResult.get("nId2BestResults", []):
        if int(item.get("nId2", -1)) == nId2:
            timingSeeds.add(int(item.get("timingOffset", timingInitial)))
    if cpDelta > 0:
        timingSeeds.add(timingInitial + cpDelta)
        if timingInitial - cpDelta >= 0:
            timingSeeds.add(timingInitial - cpDelta)

    # PSS 后选交接：
    # 原始 PSS 搜索只决定 N_ID_2、粗频偏和定时；如果接收 SSB 在 FFT
    # 内有子载波平移，原始搜索还不知道最佳 `ssbSubcarrierOffset`。
    # 这里对每个配置的 `sssOffsetCandidates`、邻近 CP/定时假设重放 PSS
    # 得分，再把稳定的 `ssbSubcarrierOffset` 和修正后的 `timingOffset`
    # 写回 `pssResult`。SSS 只接收这个增强后的 PSS 字典。
    pssPostCandidates: list[dict[str, Any]] = []
    cpCandidates: list[tuple[int, str]] = [(normalCp, "normal")]
    if bool(cfg.pssAllowLongCpPostSelect):
        cpCandidates.append((longCp, "long"))
    for cpLenCand, cpKind in cpCandidates:
        for offsetCand in cfg.sssOffsetCandidates:
            for seed in sorted(timingSeeds):
                for dt in (-2, -1, 0, 1, 2):
                    timingCand = int(seed + dt)
                    if timingCand < 0:
                        continue
                    scoreItem = _pssScoreAt(
                        rxSignal=rxSignal,
                        sampleRate=cfg.sampleRate,
                        nfft=int(ssbConfig.FftSize),
                        cpLength=int(cpLenCand),
                        nId2=int(nId2),
                        timingOffset=int(timingCand),
                        ssbSubcarrierOffset=int(offsetCand),
                        freqHz=float(baseFreqForEstHz),
                    )
                    if scoreItem.get("valid", False):
                        scoreItem["cpKind"] = cpKind
                        pssPostCandidates.append(scoreItem)

    if pssPostCandidates:
        pssPostCandidates = sorted(pssPostCandidates, key=lambda x: float(x["score"]), reverse=True)
        bestPost = _selectStablePssPostCandidate(
            candidates=pssPostCandidates,
            anchorTimingOffset=int(timingInitial),
            anchorOffset=int(getattr(ssbConfig, "SsbSubcarrierOffset", 0)),
        )
        pssResult["timingOffsetOriginal"] = int(pssResult["timingOffset"])
        pssResult["timingOffset"] = int(bestPost["timingOffset"])
        pssResult["ssbSubcarrierOffset"] = int(bestPost["ssbSubcarrierOffset"])
        pssResult["pssPostSelection"] = {
            "cpLength": int(bestPost["cpLength"]),
            "cpKind": str(bestPost["cpKind"]),
            "ssbSubcarrierOffset": int(bestPost["ssbSubcarrierOffset"]),
            "timingOffset": int(bestPost["timingOffset"]),
            "score": float(bestPost["score"]),
            "topCandidates": pssPostCandidates[:10],
        }

    # PSS 半符号 FFO 交接：
    # 先用 `baseFreqForEstHz` 补偿 PSS useful 部分，再把它和本地 PSS 模板
    # 分成前后两半比较。两半之间的相位旋转得到 `ffoHz`；
    # `totalCfoHz = baseFreqForEstHz + ffoHz` 是默认交给 SSS/PBCH 的频偏。
    timingOffset = int(pssResult["timingOffset"])
    fftSize = int(ssbConfig.FftSize)
    cpLen = int(pssResult.get("pssPostSelection", {}).get("cpLength", ssbConfig.NormalCpLength))
    halfLen = fftSize // 2
    sampleRate = float(ssbConfig.SampleRate)

    selectedPssOffset = int(
        pssResult.get("ssbSubcarrierOffset", getattr(ssbConfig, "SsbSubcarrierOffset", 0))
    )
    pssSeq = generatePssSequence(nId2)
    pssFreq = np.zeros(fftSize, dtype=np.complex64)
    pssFreq[
        fftSize // 2 - 63 + selectedPssOffset:fftSize // 2 + 64 + selectedPssOffset
    ] = pssSeq
    pssTimeTemplate = np.fft.ifft(np.fft.ifftshift(pssFreq)).astype(np.complex64)
    pssTimeTemplate /= np.linalg.norm(pssTimeTemplate)

    usefulStart = timingOffset + cpLen
    globalIdx = np.arange(usefulStart, usefulStart + fftSize, dtype=np.float64)
    phase = np.exp(-1j * 2.0 * np.pi * baseFreqForEstHz * globalIdx / sampleRate).astype(np.complex64)
    pssRx = (np.asarray(rxSignal[usefulStart:usefulStart + fftSize], dtype=np.complex64) * phase).astype(np.complex64)

    C0 = np.vdot(pssRx[:halfLen], pssTimeTemplate[:halfLen])
    C1 = np.vdot(pssRx[halfLen:], pssTimeTemplate[halfLen:])
    deltaTheta = float(np.angle(C1 * np.conjugate(C0)))
    ffoHz = deltaTheta * sampleRate / (np.pi * fftSize)
    totalCfoHz = baseFreqForEstHz + ffoHz

    pssFfoResult = {
        "method": "pss_half_symbol_ffo",
        "C0": complex(C0),
        "C1": complex(C1),
        "deltaThetaRad": deltaTheta,
        "ffoHz": ffoHz,
        "totalCfoHz": totalCfoHz,
        "baseFreqHz": baseFreqForEstHz,
    }
    pssResult["pssFfoEstimation"] = pssFfoResult

    # PSS LS 相位拟合：
    # 当样本能拟合出线性相位斜率时，写入
    # `pssResult["pssFreqOffsetEstimation"]`。R^2 是质量诊断值；SSS 实际
    # 使用下面显式选择的 `sssCompFreqHz`。
    pssRefConfig = createSsbConfigWithOverrides(
        sampleRate=cfg.sampleRate,
        subcarrierSpacing=cfg.scs,
        fftSize=ssbConfig.FftSize,
        normalCpLength=ssbConfig.NormalCpLength,
        ssbSubcarrierOffset=selectedPssOffset,
    )
    lsEstimator = PssFreqOffsetEstimator(pssRefConfig)
    lsResult = lsEstimator.estimate(
        rxSignal=rxSignal,
        nId2=nId2,
        timingOffset=timingOffset,
        baseFreqHz=baseFreqForEstHz,
    )
    if lsResult is not None:
        pssResult["pssFreqOffsetEstimation"] = lsResult

    # CP 相位频偏估计：
    # 比较 CP 样本与对应 useful symbol 尾部，写入
    # `pssResult["cpFreqOffsetEstimation"]`。这是另一条共享 PSS 定时锚点
    # 的频偏诊断路径。
    cpEstimator = CpFreqOffsetEstimator(ssbConfig)
    cpResult = cpEstimator.estimate(
        rxSignal=rxSignal,
        timingOffset=timingOffset,
        baseFreqHz=baseFreqForEstHz,
    )
    if cpResult is not None:
        pssResult["cpFreqOffsetEstimation"] = cpResult
    else:
        logger.warning("CP frequency offset estimation returned None (no valid CP windows)")

    logger.info("5) Search result")
    logger.info(f"  Mode: adaptive baseband (no GSCN)")
    logger.info(f"  Best baseband frequency: {pssResult['freqOffsetParabolic'] / 1e3:.3f} kHz")
    freqSearch = pssResult.get("freqSearch", {})
    if isinstance(freqSearch, dict):
        for name in ["coarse", "medium", "fine"]:
            p = freqSearch.get("passes", {}).get(name, {})
            if p:
                logger.info(
                    f"  {name}: step={p.get('stepHz', 0):.0f} Hz, "
                    f"bestFreq={p.get('bestFreqHz', 0) / 1e3:.3f} kHz, "
                    f"bestNId2={p.get('bestNId2', '?')}"
                )
    ar = pssResult.get("adaptiveRefinement", {})
    if ar:
        logger.info(
            f"  Sub-fine: iters={ar.get('finalValidIterations', 0)}/{ar.get('maxIterations', 0)}, "
            f"status={ar.get('finalStatus', '?')}"
        )
    logger.info(f"  Best N_ID_2: {pssResult['nId2']}")
    logger.info(f"  Best timing offset: {pssResult['timingOffset']}")
    logger.info(f"  Peak value: {pssResult['peakValue']:.6f}")
    if "pssPostSelection" in pssResult:
        post = pssResult["pssPostSelection"]
        logger.info(
            "  PSS post-select: "
            f"cp={post['cpKind']}({post['cpLength']}), offset={post['ssbSubcarrierOffset']}, "
            f"timing={post['timingOffset']}, score={post['score']:.6f}"
        )
    logger.info(
        f"  Half-symbol FFO: C0={np.abs(C0):.0f}, C1={np.abs(C1):.0f}, "
        f"Δθ={np.degrees(deltaTheta):.2f}°, "
        f"FFO={ffoHz:.2f} Hz, totalCFO={totalCfoHz:.2f} Hz"
    )
    if lsResult is not None:
        logger.info(
            f"  LS phase fit: R²={lsResult['rSquared']:.6f}, "
            f"residual={lsResult['residualFreqHz']:.2f} Hz, "
            f"refined={lsResult['refinedFreqHz']:.2f} Hz, "
            f"validPts={lsResult['validSampleCount']}/{lsResult['sampleCount']}"
        )
    if cpResult is not None:
        logger.info(
            f"  CP phase: residual={cpResult['residualFreqHz']:.2f} Hz, "
            f"std={cpResult['residualStdHz']:.2f} Hz, "
            f"windows={cpResult['symbolCountUsed']}, "
            f"coherence={np.mean(cpResult['coherence']):.3f}, "
            f"confidence={cpResult['confidence']:.3f} ({cpResult['confidenceLevel']})"
        )

    sssResult = None
    sssSearchCandidates: list[dict[str, Any]] = []
    # SSS 频偏交接：
    # `sssCompFreqHz` 是 SssDetector.detectSss() 内部使用的频偏补偿值。
    # 默认使用 PSS 得到的 `totalCfoHz`。它必须和 `ssbSubcarrierOffset`
    # 分开理解，后者是子载波索引平移。`cfg.scs` 仍然只是子载波间隔；
    # 兼容遗留配置的 "scs" 模式会映射回 `totalCfoHz`，避免把 SCS 当 CFO。
    if cfg.sssFreqBaseMode == "zero":
        sssCompFreqHz = 0.0
    elif cfg.sssFreqBaseMode == "scs":
        logger.warning(
            f"sssFreqBaseMode='scs' 已废弃，SCS={cfg.scs}Hz 不是频偏值。"
            f"请使用 'pss' (推荐) 或 'zero'。本次将使用 totalCfoHz={totalCfoHz:.2f}Hz 代替。"
        )
        sssCompFreqHz = totalCfoHz
    else:  # "pss" 或默认
        sssCompFreqHz = totalCfoHz
    if cfg.sssEnable:
        logger.info("6) Run SSS sliding search")
        # SSS 输入：
        #   `rxSignal`、增强后的 `pssResult`、一个频偏补偿值 `sssCompFreqHz`，
        #   以及逐个测试的候选 `ssbSubcarrierOffset`。
        # SSS 处理：
        #   针对 PSS 检出的 `nId2` 构造 336 个模板；围绕理论符号 2 位置
        #   做滑窗；再用小范围残余频偏网格对 Top 候选做频域复核。
        # SSS 每个候选的输出：
        #   `nId1`、`nIdCell`、`bestSymbolStart`、`verifiedSymbolStart`、
        #   `verifiedFreqCompHz` 以及相关分数曲线。
        # 规范依据：TS 38.211 Clause 7.4.2.1、7.4.2.3 和 Table 7.4.3.1-1。
        offsetCandidates = list(cfg.sssOffsetCandidates) or [int(getattr(ssbConfig, "SsbSubcarrierOffset", 0))]
        selectedOffset = int(pssResult.get("ssbSubcarrierOffset", 0))
        if selectedOffset in offsetCandidates:
            offsetCandidates = [selectedOffset] + [x for x in offsetCandidates if x != selectedOffset]
        else:
            offsetCandidates = [selectedOffset] + offsetCandidates
        for offset in offsetCandidates:
            candidateConfig = createSsbConfigWithOverrides(
                sampleRate=cfg.sampleRate,
                subcarrierSpacing=cfg.scs,
                fftSize=ssbConfig.FftSize,
                normalCpLength=ssbConfig.NormalCpLength,
                ssbSubcarrierOffset=int(offset),
            )
            sssDetector = SssDetector(candidateConfig)
            try:
                candidateSss = sssDetector.detectSss(
                    rxSignal=rxSignal,
                    pssResult=pssResult,
                    freqCompHz=sssCompFreqHz,
                    startSymbol=cfg.sssSearchStartSymbol,
                    endSymbol=cfg.sssSearchEndSymbol,
                    stepSamples=cfg.sssSearchStepSamples,
                )
                candidateSss["ssbSubcarrierOffset"] = int(offset)
                candidateSss["selectionScore"] = float(
                    candidateSss.get("verifiedFdScore", 0.0) * max(candidateSss.get("bestScore", 0.0), 1e-12)
                )
                sssSearchCandidates.append(candidateSss)
                logger.info(
                    "  SSS candidate offset=%s -> N_ID_1=%s, N_ID_cell=%s, bestOffset=%s, score=%.6f, fdScore=%.6f",
                    offset,
                    candidateSss["nId1"],
                    candidateSss["nIdCell"],
                    candidateSss["bestOffsetSamples"],
                    candidateSss["bestScore"],
                    candidateSss.get("verifiedFdScore", 0.0),
                )
            except Exception as e:
                logger.warning(f"  SSS candidate offset={offset} failed: {e}")

        if sssSearchCandidates:
            sssSearchCandidates = sorted(
                sssSearchCandidates,
                key=lambda item: (
                    item.get("selectionScore", 0.0),
                    item.get("verifiedFdScore", 0.0),
                    item.get("bestScore", 0.0),
                ),
                reverse=True,
            )
            pssAnchorOffset = int(pssResult.get("ssbSubcarrierOffset", getattr(ssbConfig, "SsbSubcarrierOffset", 0)))
            pssAnchorSymbolStart = int(pssResult["timingOffset"]) + int(cpManager.getSymbolLength(0)) + int(cpManager.getSymbolLength(1))
            # `sssResult` 是选中的 SSS 交接对象。PBCH 以后以它作为完整
            # 小区 ID、SSS 定时、频域复核频偏和 SSB 子载波偏移的来源。
            # 同时保留 `pssResult["sssSearchCandidates"]`，让 PBCH 能重试
            # 接近并列的 SSS 候选。
            sssResult = _selectStableSssCandidate(
                candidates=sssSearchCandidates,
                anchorOffset=pssAnchorOffset,
                anchorSymbolStart=pssAnchorSymbolStart,
                anchorFreqHz=float(sssCompFreqHz),
            )
            pssResult["sssResult"] = sssResult
            pssResult["sssSearchCandidates"] = sssSearchCandidates[:10]
            logger.info(
                "  SSS result: "
                f"offset={sssResult['ssbSubcarrierOffset']}, "
                f"N_ID_1={sssResult['nId1']}, N_ID_cell={sssResult['nIdCell']}, "
                f"bestOffset={sssResult['bestOffsetSamples']} samples, score={sssResult['bestScore']:.6f}, "
                f"fdScore={sssResult.get('verifiedFdScore', 0.0):.6f}, "
                f"residualCFO={sssResult.get('verifiedResidualFreqHz', 0.0):.2f} Hz"
            )
        else:
            logger.warning("  SSS sliding search failed for all offset candidates")

    pbchResult = None
    bchResult = None
    if cfg.pbchEnable and sssResult is not None:
        logger.info("7) Run PBCH DM-RS equalization and constellation demod")

        # PBCH 输入：
        #   `rxSignal`、选中的 `sssResult`，以及受数量限制的近似并列 SSS
        #   候选列表。PBCH 必须优先使用每个 SSS 候选的
        #   `verifiedFreqCompHz`；这样保留 SSS 频域复核结果，而不是退回到
        #   更早的 PSS-only 频偏。
        # PBCH 处理：
        #   对每个 SSS 交接对象扫描 SSB 起点、残余频偏、CP profile 和
        #   `iSsbBar`；提取 240x4 的 SS/PBCH 网格；由 `nIdCell/iSsbBar`
        #   生成 PBCH DM-RS；用 DM-RS 估计信道；均衡 PBCH RE；做 QPSK
        #   硬判决。
        # PBCH 输出：
        #   `pbchResult["pbchEq"]` -> BCH 软/硬解码；
        #   `pbchResult["hardBits"]` -> 诊断硬比特；
        #   `pbchResult["noiseVarEstimate"]` -> BCH LLR 缩放；
        #   摘要指标 -> JSON 和候选排序。
        # 规范依据：TS 38.211 Clause 7.4.3.1 和 7.4.1.4。
        pbchSssCandidates: list[dict[str, Any]] = []
        seenPbchSssKeys: set[tuple[int, int, int]] = set()
        preferredSssCandidates = [sssResult] + sssSearchCandidates
        for candidateSss in preferredSssCandidates:
            key = (
                int(candidateSss.get("ssbSubcarrierOffset", 0)),
                int(candidateSss.get("verifiedSymbolStart", candidateSss.get("bestSymbolStart", 0))),
                int(round(float(candidateSss.get("verifiedFreqCompHz", candidateSss.get("freqCompHz", 0.0))))),
            )
            if key in seenPbchSssKeys:
                continue
            seenPbchSssKeys.add(key)
            pbchSssCandidates.append(candidateSss)
            if len(pbchSssCandidates) >= int(cfg.pbchSssCandidateLimit):
                break

        pbchSearchResults: list[dict[str, Any]] = []
        for idx, candidateSss in enumerate(pbchSssCandidates, start=1):
            # SSS 到 PBCH 的数据交接：
            # 优先用 `candidateSss["verifiedFreqCompHz"]`，因为它经过 SSS
            # 频域复核；只有缺失时才退回到 PSS 派生的 `sssCompFreqHz`。
            pbchFreqCompHz = float(candidateSss.get("verifiedFreqCompHz", sssCompFreqHz))
            pbchConfig = createSsbConfigWithOverrides(
                sampleRate=cfg.sampleRate,
                subcarrierSpacing=cfg.scs,
                fftSize=ssbConfig.FftSize,
                normalCpLength=ssbConfig.NormalCpLength,
                ssbSubcarrierOffset=int(
                    candidateSss.get(
                        "ssbSubcarrierOffset",
                        getattr(ssbConfig, "SsbSubcarrierOffset", 0),
                    )
                ),
            )
            pbchDecoder = PbchDecoder(
                pbchConfig,
                ssbIndexCandidates=list(cfg.pbchIssbCandidates),
                cpProfileNames=list(cfg.pbchCpProfiles),
            )
            try:
                candidatePbch = pbchDecoder.decodePbch(
                    rxSignal=rxSignal,
                    sssResult=candidateSss,
                    pssResult=pssResult,
                    freqCompHz=pbchFreqCompHz,
                    residualFreqSearchHz=cfg.pbchResidualFreqSearchHz,
                    residualFreqStepHz=cfg.pbchResidualFreqStepHz,
                )
            except Exception as e:
                logger.warning(
                    "  PBCH candidate from SSS[%s] failed: offset=%s, start=%s, freq=%.2f Hz, err=%s",
                    idx,
                    candidateSss.get("ssbSubcarrierOffset"),
                    candidateSss.get("verifiedSymbolStart", candidateSss.get("bestSymbolStart")),
                    pbchFreqCompHz,
                    e,
                )
                continue

            candidatePbch["sourceSssCandidate"] = {
                "ssbSubcarrierOffset": int(candidateSss.get("ssbSubcarrierOffset", 0)),
                "nId1": int(candidateSss.get("nId1", -1)),
                "nIdCell": int(candidateSss.get("nIdCell", -1)),
                "bestSymbolStart": int(candidateSss.get("bestSymbolStart", 0)),
                "verifiedSymbolStart": int(
                    candidateSss.get("verifiedSymbolStart", candidateSss.get("bestSymbolStart", 0))
                ),
                "bestScore": float(candidateSss.get("bestScore", 0.0)),
                "verifiedFdScore": float(candidateSss.get("verifiedFdScore", 0.0)),
                "selectionScore": float(candidateSss.get("selectionScore", 0.0)),
                "verifiedFreqCompHz": float(candidateSss.get("verifiedFreqCompHz", pbchFreqCompHz)),
            }
            pbchSearchResults.append(candidatePbch)
            logger.info(
                "  PBCH from SSS[%s]: offset=%s, sssStart=%s, sssFreq=%.2f Hz -> "
                "i_SSB_bar=%s, ssbStart=%s, pbchFreq=%.2f Hz, cp=%s, EVM=%.2f%%, DMRS=%.4f",
                idx,
                candidatePbch["sourceSssCandidate"]["ssbSubcarrierOffset"],
                candidatePbch["sourceSssCandidate"]["verifiedSymbolStart"],
                candidatePbch["sourceSssCandidate"]["verifiedFreqCompHz"],
                candidatePbch["iSsbBar"],
                candidatePbch["ssbStart"],
                candidatePbch["freqCompHz"],
                candidatePbch["cpProfile"],
                candidatePbch["evmPercent"],
                candidatePbch.get("dmrsCorrNorm", 0.0),
            )

        if not pbchSearchResults:
            logger.warning("  PBCH demod failed for all SSS candidates")
        else:
            pbchSearchResults.sort(
                key=lambda item: (
                    PbchDecoder._candidateRankKey(item, mode="evm_guarded"),
                    float(item.get("sourceSssCandidate", {}).get("selectionScore", 0.0)),
                ),
                reverse=True,
            )
            pbchResult = pbchSearchResults[0]
            pssResult["pbchSearchResults"] = [
                {
                    "iSsbBar": int(item["iSsbBar"]),
                    "ssbStart": int(item["ssbStart"]),
                    "freqCompHz": float(item["freqCompHz"]),
                    "cpProfile": str(item["cpProfile"]),
                    "evmPercent": float(item["evmPercent"]),
                    "dmrsCorrNorm": float(item.get("dmrsCorrNorm", 0.0)),
                    "dmrsEvmPercent": float(item.get("dmrsEvmPercent", 0.0)),
                    "sourceSssCandidate": dict(item.get("sourceSssCandidate", {})),
                }
                for item in pbchSearchResults[:10]
            ]
            pssResult["pbchSelectedSssCandidate"] = dict(pbchResult.get("sourceSssCandidate", {}))
            pssResult["pbchResult"] = {
                k: v for k, v in pbchResult.items()
                if k not in ("pbchEq", "pbchHardRef", "hardBits", "dataRe", "dmrsRe")
            }
            logger.info(
                "  PBCH result: "
                f"i_SSB_bar={pbchResult['iSsbBar']}, ssbStart={pbchResult['ssbStart']}, "
                f"freq={pbchResult['freqCompHz']:.2f} Hz, EVM={pbchResult['evmPercent']:.2f}% "
                f"({'PASS' if pbchResult['evmPass10Percent'] else 'FAIL'} <10%), "
                f"from SSS offset={pbchResult.get('sourceSssCandidate', {}).get('ssbSubcarrierOffset')} "
                f"start={pbchResult.get('sourceSssCandidate', {}).get('verifiedSymbolStart')}"
            )

            logger.info("8) Run PBCH/BCH decode pipeline")
            # BCH/MIB 输入：
            #   `pbchResult["pbchEq"]` 中的等化 PBCH QPSK 符号、SSS 给出的
            #   `nIdCell`、PBCH DM-RS 选择出的 `iSsbBar` 和 PBCH 噪声估计。
            #   这一阶段不再接触原始 IQ 采样。
            # 输出 `bchResult` 会挂到 `pssResult` 进入总 JSON，也会挂到
            # `pbchResult` 便于 PBCH 专属工件保存。
            bchDecoder = PbchBchDecoder()
            bchResult = bchDecoder.decode(
                pbchEq=pbchResult["pbchEq"],
                nIdCell=int(pbchResult["nIdCell"]),
                iSsbBar=int(pbchResult["iSsbBar"]),
                noiseVar=float(pbchResult.get("noiseVarEstimate", 1.0)),
                outputPrefix=str(cfg.outputPrefix),
            )
            pssResult["pbchBchResult"] = bchResult
            if bchResult.get("crcOk", False):
                mib = bchResult.get("mib", {})
                logger.info(
                    "  BCH result: CRC PASS, SFN=%s, HRF=%s",
                    mib.get("systemFrameNumber10"),
                    mib.get("halfFrameBit"),
                )
            else:
                logger.info(
                    "  BCH result: CRC FAIL, failureStage=%s, crcRemainder=%s",
                    bchResult.get("failureStage", ""),
                    bchResult.get("crcRemainder"),
                )

    logger.info("9) Save plots and result json")
    PssVisualizer.plotPssSearchValidation(
        pssResult,
        [],  # validGscnList (unused, kept for API compatibility)
        len(rxSignal),
        outputPrefix=cfg.outputPrefix,
    )
    if sssResult is not None:
        SssVisualizer.plotAndSave(sssResult, outputPrefix=cfg.outputPrefix)
    if pbchResult is not None:
        if bchResult is not None:
            pbchResult["bchDecode"] = bchResult
        PbchDecoder.saveArtifacts(pbchResult, outputPrefix=cfg.outputPrefix)
    logger.info("Done")


if __name__ == "__main__":
    main()
