import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.config import createSsbConfigWithOverrides
from pbch.pbchDecoder import PbchDecoder, PbchRe
from pss.pssTemplateFactory import generatePssSequence
from sss.sssDetector import SssDetector


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-parameter RX diagnostics for the PBCH/MIB candidate."
    )
    parser.add_argument("--signal-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--output-prefix", type=str, default="rx_fixed_diagnostics")
    parser.add_argument("--sample-rate", type=float, default=122880000.0)
    parser.add_argument("--scs", type=int, default=30000)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--ssb-subcarrier-offset", type=int, default=-7)
    parser.add_argument("--n-id-cell", type=int, default=0)
    parser.add_argument("--i-ssb-bar", type=int, default=0)
    parser.add_argument("--ssb-start", type=int, default=6)
    parser.add_argument("--freq-comp-hz", type=float, default=30288.0)
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    return value


def _normCorr(rx: np.ndarray, ref: np.ndarray) -> float:
    rx = np.asarray(rx, dtype=np.complex128).reshape(-1)
    ref = np.asarray(ref, dtype=np.complex128).reshape(-1)
    return float(abs(np.vdot(ref, rx)) / max(float(np.linalg.norm(ref) * np.linalg.norm(rx)), 1e-12))


def _nearestQpsk(symbols: np.ndarray) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    realBits = (np.real(symbols) < 0).astype(np.float64)
    imagBits = (np.imag(symbols) < 0).astype(np.float64)
    return ((1.0 - 2.0 * realBits) + 1j * (1.0 - 2.0 * imagBits)) / np.sqrt(2.0)


def _evmPercent(symbols: np.ndarray, reference: np.ndarray) -> float:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    return float(100.0 * np.sqrt(np.mean(np.abs(symbols - reference) ** 2) / max(float(np.mean(np.abs(reference) ** 2)), 1e-12)))


def _normalizeAndEvm(symbols: np.ndarray) -> dict[str, Any]:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    ref0 = _nearestQpsk(symbols)
    gain = np.vdot(ref0, symbols) / max(float(np.vdot(ref0, ref0).real), 1e-12)
    eq = symbols / (gain if abs(gain) > 1e-12 else 1.0 + 0j)
    ref = _nearestQpsk(eq)
    return {
        "evmPercent": _evmPercent(eq, ref),
        "gainReal": float(np.real(gain)),
        "gainImag": float(np.imag(gain)),
        "symbolPower": float(np.mean(np.abs(eq) ** 2)),
    }


def _rawPhaseMagnitudeInterpolation(
    dmrsItems: list[PbchRe],
    hDmrs: np.ndarray,
    dataItems: list[PbchRe],
) -> np.ndarray:
    bySymbol: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sym in (1, 2, 3):
        idx = [i for i, item in enumerate(dmrsItems) if item.l == sym]
        k = np.asarray([dmrsItems[i].k for i in idx], dtype=np.float64)
        h = np.asarray([hDmrs[i] for i in idx], dtype=np.complex128)
        order = np.argsort(k)
        k = k[order]
        h = h[order]
        bySymbol[sym] = (k, np.abs(h), np.unwrap(np.angle(h)))

    out = np.zeros(len(dataItems), dtype=np.complex128)
    for i, item in enumerate(dataItems):
        kRef, magRef, phaseRef = bySymbol[item.l]
        mag = float(np.interp(float(item.k), kRef, magRef))
        phase = float(np.interp(float(item.k), kRef, phaseRef))
        out[i] = mag * np.exp(1j * phase)
    return out


def _symbolMeanChannel(dmrsItems: list[PbchRe], hDmrs: np.ndarray, dataItems: list[PbchRe]) -> np.ndarray:
    bySymbol: dict[int, complex] = {}
    for sym in (1, 2, 3):
        vals = [hDmrs[i] for i, item in enumerate(dmrsItems) if item.l == sym]
        bySymbol[sym] = complex(np.mean(vals))
    return np.asarray([bySymbol[item.l] for item in dataItems], dtype=np.complex128)


def _angleLoopRefine(dataEq: np.ndarray, dataItems: list[PbchRe], iterations: int = 3) -> np.ndarray:
    out = np.asarray(dataEq, dtype=np.complex128).copy()
    for _ in range(int(iterations)):
        for sym in (1, 2, 3):
            idx = np.asarray([i for i, item in enumerate(dataItems) if item.l == sym], dtype=np.int32)
            if len(idx) == 0:
                continue
            symVals = out[idx]
            ref = _nearestQpsk(symVals)
            metric = np.vdot(ref, symVals)
            cpe = float(np.angle(metric)) if abs(metric) > 1e-12 else 0.0
            out[idx] = symVals * np.exp(-1j * cpe)
    return out


def _compensate(rxSignal: np.ndarray, sampleRate: float, freqHz: float) -> np.ndarray:
    n = np.arange(len(rxSignal), dtype=np.float64)
    return np.asarray(rxSignal, dtype=np.complex128) * np.exp(-1j * 2.0 * np.pi * float(freqHz) * n / float(sampleRate))


def _cpResidualCfoHz(rxComp: np.ndarray, ssbStart: int, nfft: int, cpLengths: list[int], sampleRate: float) -> dict[str, Any]:
    starts: list[int] = []
    current = int(ssbStart)
    perSymbol: list[dict[str, float]] = []
    for sym, cpLen in enumerate(cpLengths):
        cpLen = int(cpLen)
        if current >= 0 and current + cpLen + nfft <= len(rxComp) and cpLen > 0:
            cp = rxComp[current:current + cpLen]
            tail = rxComp[current + nfft:current + nfft + cpLen]
            corr = np.vdot(tail, cp)
            residualHz = float(np.angle(corr) * float(sampleRate) / (2.0 * np.pi * float(nfft)))
            perSymbol.append({
                "symbol": int(sym),
                "residualHz": residualHz,
                "corrNorm": float(abs(corr) / max(float(np.linalg.norm(cp) * np.linalg.norm(tail)), 1e-12)),
            })
        starts.append(current)
        current += int(cpLen) + int(nfft)
    vals = [x["residualHz"] for x in perSymbol if np.isfinite(x["residualHz"])]
    return {
        "method": "cp_phase_after_fixed_compensation",
        "meanResidualHz": float(np.mean(vals)) if vals else None,
        "stdResidualHz": float(np.std(vals)) if vals else None,
        "perSymbol": perSymbol,
        "symbolStarts": starts,
    }


def _phaseSlopeResidualHz(phases: list[float], sampleTimes: list[float], sampleRate: float) -> dict[str, Any]:
    if len(phases) < 2:
        return {"residualHz": None, "phaseSlopeRadPerSample": None}
    pha = np.unwrap(np.asarray(phases, dtype=np.float64))
    t = np.asarray(sampleTimes, dtype=np.float64)
    coef = np.polyfit(t, pha, deg=1)
    slope = float(coef[0])
    fit = np.polyval(coef, t)
    return {
        "residualHz": float(slope * float(sampleRate) / (2.0 * np.pi)),
        "phaseSlopeRadPerSample": slope,
        "phaseResidualStdRad": float(np.std(pha - fit)),
        "phasesRad": [float(x) for x in pha],
        "sampleTimes": [float(x) for x in t],
    }


def _symbolUsefulStarts(ssbStart: int, nfft: int, cpLengths: list[int]) -> list[int]:
    starts: list[int] = []
    current = int(ssbStart)
    for cpLen in cpLengths:
        starts.append(current + int(cpLen))
        current += int(cpLen) + int(nfft)
    return starts


def _sssScan(grid: np.ndarray, sssDetector: SssDetector, nId2: int, ssbSubcarrierOffset: int) -> dict[str, Any]:
    sssStart = 56
    sssRx = grid[sssStart:sssStart + 127, 2]
    scores = []
    for nId1 in range(336):
        ref = sssDetector._buildSssSequence(nId1=nId1, nId2=nId2)  # noqa: SLF001
        scores.append(_normCorr(sssRx, ref))
    order = np.argsort(scores)[::-1]
    return {
        "bestNId1": int(order[0]),
        "bestNIdCell": int(3 * int(order[0]) + int(nId2)),
        "bestScore": float(scores[order[0]]),
        "top5": [
            {
                "nId1": int(i),
                "nIdCell": int(3 * int(i) + int(nId2)),
                "score": float(scores[i]),
            }
            for i in order[:5]
        ],
        "ssbSubcarrierOffset": int(ssbSubcarrierOffset),
    }


def _localTimingScores(
    decoder: PbchDecoder,
    rxSignal: np.ndarray,
    nId2: int,
    nIdCell: int,
    iSsbBar: int,
    ssbStart: int,
    freqCompHz: float,
    cpLengths: list[int],
    sssDetector: SssDetector,
) -> list[dict[str, Any]]:
    pssRef = generatePssSequence(nId2)
    sssRef = sssDetector._buildSssSequence(nId1=nIdCell // 3, nId2=nId2)  # noqa: SLF001
    out = []
    for delta in range(-8, 9):
        candStart = int(ssbStart + delta)
        if candStart < 0:
            continue
        grid = decoder._extractSsbGrid(rxSignal, candStart, freqCompHz, cpLengths=cpLengths)  # noqa: SLF001
        dmrsItems = decoder._dmrsReList(nIdCell)  # noqa: SLF001
        dmrsRx = decoder._extract(grid, dmrsItems)  # noqa: SLF001
        dmrsRef = decoder.generatePbchDmrs(nIdCell, iSsbBar)
        out.append({
            "ssbStart": candStart,
            "delta": int(delta),
            "pssCorr": _normCorr(grid[56:183, 0], pssRef),
            "sssCorr": _normCorr(grid[56:183, 2], sssRef),
            "dmrsCorrNorm": _normCorr(dmrsRx, dmrsRef),
        })
    return sorted(out, key=lambda x: (x["pssCorr"] + x["sssCorr"] + x["dmrsCorrNorm"]), reverse=True)


def _channelStats(dmrsItems: list[PbchRe], hDmrs: np.ndarray, usefulStarts: list[int], sampleRate: float) -> dict[str, Any]:
    perSymbol = []
    cpePhases = []
    cpeTimes = []
    for sym in (1, 2, 3):
        idx = np.asarray([i for i, item in enumerate(dmrsItems) if item.l == sym], dtype=np.int32)
        h = np.asarray(hDmrs[idx], dtype=np.complex128)
        k = np.asarray([dmrsItems[i].k for i in idx], dtype=np.float64)
        phase = np.unwrap(np.angle(h))
        mag = np.abs(h)
        phaseCoef = np.polyfit(k, phase, deg=1) if len(k) >= 2 else np.asarray([0.0, float(phase[0])])
        phaseFit = np.polyval(phaseCoef, k)
        cpe = complex(np.mean(h))
        cpePhases.append(float(np.angle(cpe)))
        cpeTimes.append(float(usefulStarts[sym]))
        perSymbol.append({
            "symbol": int(sym),
            "dmrsCount": int(len(h)),
            "magMean": float(np.mean(mag)),
            "magStd": float(np.std(mag)),
            "magCv": float(np.std(mag) / max(float(np.mean(mag)), 1e-12)),
            "phaseLinearSlopeRadPerSubcarrier": float(phaseCoef[0]),
            "phaseResidualStdRad": float(np.std(phase - phaseFit)),
            "cpePhaseRad": float(np.angle(cpe)),
        })
    return {
        "perSymbol": perSymbol,
        "dmrsCpeResidualCfo": _phaseSlopeResidualHz(cpePhases, cpeTimes, sampleRate),
    }


def _dmrsEvmStats(dmrsItems: list[PbchRe], dmrsRx: np.ndarray, dmrsRef: np.ndarray) -> dict[str, Any]:
    hDmrs = dmrsRx / dmrsRef
    overallMean = np.mean(hDmrs)
    overallEq = dmrsRx / (overallMean if abs(overallMean) > 1e-12 else 1.0 + 0j)
    perSymbolEvm = []
    perSymbolEq = np.zeros_like(dmrsRx, dtype=np.complex128)
    for sym in (1, 2, 3):
        idx = np.asarray([i for i, item in enumerate(dmrsItems) if item.l == sym], dtype=np.int32)
        hMean = np.mean(hDmrs[idx])
        perSymbolEq[idx] = dmrsRx[idx] / (hMean if abs(hMean) > 1e-12 else 1.0 + 0j)
        perSymbolEvm.append({
            "symbol": int(sym),
            "evmPercent": _evmPercent(perSymbolEq[idx], dmrsRef[idx]),
        })
    return {
        "overallMeanChannelEvmPercent": _evmPercent(overallEq, dmrsRef),
        "perSymbolMeanChannelEvmPercent": _evmPercent(perSymbolEq, dmrsRef),
        "perSymbol": perSymbolEvm,
    }


def _loadJson(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _renderMarkdown(result: dict[str, Any]) -> str:
    pbch = result["pbch"]
    sync = result["sync"]
    freq = result["frequency"]
    ch = result["channel"]
    coreset = result["coreset0"]
    lines = [
        "# RX Fixed-Parameter Diagnostic Report",
        "",
        "## Verdict",
        f"- PBCH/BCH: `{pbch['bchCrcOk']}`; MIB decode is locally proven only for the fixed SSB candidate.",
        f"- PBCH current EVM: `{pbch['evmVariants']['currentDecoder']['evmPercent']:.2f}%`; QPSK quality is not compliant/healthy.",
        f"- PDCCH/SIB1: `{coreset['pdcchCrcOk']}` / `{coreset['sib1CrcOk']}`; full downlink parsing is not proven.",
        "",
        "## Fixed Candidate",
        f"- `nIdCell={result['parameters']['nIdCell']}`, `iSsbBar={result['parameters']['iSsbBar']}`, "
        f"`ssbStart={result['parameters']['ssbStart']}`, `freqCompHz={result['parameters']['freqCompHz']}` Hz, "
        f"`ssbSubcarrierOffset={result['parameters']['ssbSubcarrierOffset']}`.",
        "",
        "## Synchronization",
        f"- PSS corr at fixed point: `{sync['pssCorr']:.4f}`.",
        f"- SSS best: NID1=`{sync['sssScan']['bestNId1']}`, PCI=`{sync['sssScan']['bestNIdCell']}`, score=`{sync['sssScan']['bestScore']:.4f}`.",
        f"- Case C frame-start estimate from standard SSB timing: `{sync['caseCFrameStartEstimateSamples']}` samples.",
        f"- CORESET0 slot0 symbol1 relative start: `{coreset['slot0FirstSymbolStartInCaptureSamples']}` samples.",
        f"- CORESET0 slot1 symbol1 relative start: `{coreset['slot1FirstSymbolStartInCaptureSamples']}` samples.",
        "",
        "## Frequency",
        f"- CP residual CFO after fixed compensation: `{freq['cpResidualCfo']['meanResidualHz']}` Hz mean.",
        f"- PSS/SSS CPE residual CFO: `{freq['pssSssResidualCfo']['residualHz']}` Hz.",
        f"- PBCH DMRS CPE residual CFO: `{ch['dmrsCpeResidualCfo']['residualHz']}` Hz.",
        "",
        "## Channel / EVM",
        f"- DMRS corr norm: `{ch['dmrsCorrNorm']:.4f}`.",
        f"- DMRS EVM, overall channel mean: `{ch['dmrsEvm']['overallMeanChannelEvmPercent']:.2f}%`.",
        f"- DMRS EVM, per-symbol channel mean: `{ch['dmrsEvm']['perSymbolMeanChannelEvmPercent']:.2f}%`.",
        f"- Raw phase/magnitude interp PBCH EVM: `{pbch['evmVariants']['rawPhaseMagnitudeInterp']['evmPercent']:.2f}%`.",
        f"- Angle-loop after raw interp PBCH EVM: `{pbch['evmVariants']['rawInterpAngleLoop']['evmPercent']:.2f}%`.",
        f"- Symbol-mean + angle-loop PBCH EVM: `{pbch['evmVariants']['symbolMeanAngleLoop']['evmPercent']:.2f}%`.",
        "",
        "## Conclusion",
        "- Evidence supports PBCH/MIB decoding at one local candidate.",
        "- Evidence does not support claiming timing/frequency/channel/Point-A are fully correct.",
        "- The strongest current blockers are poor DMRS/PBCH EVM and failed SI-RNTI PDCCH CRC.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = _parseArgs()
    rxPath = Path(args.signal_path)
    if not rxPath.is_absolute():
        rxPath = (Path.cwd() / rxPath).resolve()
    rxSignal = np.asarray(np.load(rxPath), dtype=np.complex128).reshape(-1)

    cfg = createSsbConfigWithOverrides(
        sampleRate=float(args.sample_rate),
        subcarrierSpacing=int(args.scs),
        fftSize=int(args.nfft),
        ssbSubcarrierOffset=int(args.ssb_subcarrier_offset),
    )
    decoder = PbchDecoder(cfg, ssbIndexCandidates=[int(args.i_ssb_bar)])
    cpMap = {name: vals for name, vals in decoder._cpLengthProfiles()}  # noqa: SLF001
    cpLengths = [int(v) for v in cpMap["slot_head"]]
    sssDetector = SssDetector(cfg)

    best = decoder._evaluateCandidate(  # noqa: SLF001
        rxSignal=rxSignal,
        nIdCell=int(args.n_id_cell),
        ssbStart=int(args.ssb_start),
        freqCompHz=float(args.freq_comp_hz),
        iSsbBar=int(args.i_ssb_bar),
        cpProfileName="slot_head",
        cpLengths=cpLengths,
    )

    grid = best["grid"]
    dmrsItems = best["dmrsItems"]
    dataItems = best["dataItems"]
    dmrsRx = np.asarray(best["dmrsRx"], dtype=np.complex128)
    dmrsRef = np.asarray(best["dmrsRef"], dtype=np.complex128)
    hDmrs = np.asarray(best["hDmrs"], dtype=np.complex128)
    dataRx = np.asarray(best["dataRx"], dtype=np.complex128)

    nId2 = int(args.n_id_cell) % 3
    pssRef = generatePssSequence(nId2)
    pssCorr = _normCorr(grid[56:183, 0], pssRef)
    sssScan = _sssScan(grid, sssDetector, nId2=nId2, ssbSubcarrierOffset=int(args.ssb_subcarrier_offset))
    localTiming = _localTimingScores(
        decoder=decoder,
        rxSignal=rxSignal,
        nId2=nId2,
        nIdCell=int(args.n_id_cell),
        iSsbBar=int(args.i_ssb_bar),
        ssbStart=int(args.ssb_start),
        freqCompHz=float(args.freq_comp_hz),
        cpLengths=cpLengths,
        sssDetector=sssDetector,
    )

    rxComp = _compensate(rxSignal, sampleRate=float(args.sample_rate), freqHz=float(args.freq_comp_hz))
    usefulStarts = _symbolUsefulStarts(int(args.ssb_start), int(args.nfft), cpLengths)
    cpResidual = _cpResidualCfoHz(rxComp, int(args.ssb_start), int(args.nfft), cpLengths, float(args.sample_rate))

    pssH = grid[56:183, 0] / pssRef
    sssRef = sssDetector._buildSssSequence(nId1=int(args.n_id_cell) // 3, nId2=nId2)  # noqa: SLF001
    sssH = grid[56:183, 2] / sssRef
    pssSssResidual = _phaseSlopeResidualHz(
        [float(np.angle(np.mean(pssH))), float(np.angle(np.mean(sssH)))],
        [float(usefulStarts[0]), float(usefulStarts[2])],
        float(args.sample_rate),
    )

    rawH = _rawPhaseMagnitudeInterpolation(dmrsItems, hDmrs, dataItems)
    rawEq = dataRx / np.where(np.abs(rawH) > 1e-12, rawH, 1.0 + 0j)
    rawAngleEq = _angleLoopRefine(rawEq, dataItems, iterations=3)
    symMeanH = _symbolMeanChannel(dmrsItems, hDmrs, dataItems)
    symMeanEq = dataRx / np.where(np.abs(symMeanH) > 1e-12, symMeanH, 1.0 + 0j)
    symMeanAngleEq = _angleLoopRefine(symMeanEq, dataItems, iterations=3)

    chStats = _channelStats(dmrsItems, hDmrs, usefulStarts, float(args.sample_rate))
    chStats.update(
        {
            "dmrsCorrNorm": float(best["dmrsCorrNorm"]),
            "dmrsEvmPercentCurrentMetric": float(best["dmrsEvmPercent"]),
            "dmrsSnrDbFromCorrNorm": float(best["dmrsSnrDb"]),
            "hDmrsMagMean": float(np.mean(np.abs(hDmrs))),
            "hDmrsMagStd": float(np.std(np.abs(hDmrs))),
            "hDmrsMagCv": float(np.std(np.abs(hDmrs)) / max(float(np.mean(np.abs(hDmrs))), 1e-12)),
            "dmrsEvm": _dmrsEvmStats(dmrsItems, dmrsRx, dmrsRef),
        }
    )

    bchJson = _loadJson(Path("output/pbch_narrow_official_bch_scan_bch_official_result.json"))
    downlinkJson = _loadJson(Path("output/downlink_parse_probe_v3_result.json"))
    mib = None
    bchCrc = None
    if bchJson:
        bchCrc = bool(bchJson.get("crcOk", False))
        mib = bchJson.get("bestAttempt", {}).get("mib")

    # Case C SSB #0 starts at symbol 2. With the current slot_head CP this
    # puts the frame boundary before the captured SSB.
    caseCStandardSsbOffset = int(cpLengths[0] + int(args.nfft) + cpLengths[1] + int(args.nfft))
    frameStartEstimate = int(args.ssb_start) - caseCStandardSsbOffset
    slotLength = int(round(float(args.sample_rate) * 0.5e-3))
    symbol1Offset = int(cpLengths[0] + int(args.nfft))
    coreset0 = {
        "fromMib": {
            "pdcchConfigSIB1": int(mib.get("pdcchConfigSIB1", 170)) if isinstance(mib, dict) else 170,
            "kSSB": int(mib.get("kSsb", 13)) if isinstance(mib, dict) else 13,
            "dmrsTypeAPosition": int(mib.get("dmrsTypeAPosition", 3)) if isinstance(mib, dict) else 3,
        },
        "resourceIndex": 10,
        "searchSpaceZero": 10,
        "minChanBw5or10MHz": {"coresetNRB": 48, "durationSymbols": 1, "offsetRB": 12, "muxPattern": 1},
        "caseCStandardSsbOffsetSamples": caseCStandardSsbOffset,
        "caseCFrameStartEstimateSamples": frameStartEstimate,
        "slot0FirstSymbolStartInCaptureSamples": frameStartEstimate + symbol1Offset,
        "slot1FirstSymbolStartInCaptureSamples": frameStartEstimate + slotLength + symbol1Offset,
        "pdcchCrcOk": bool(downlinkJson.get("pdcchCrcOk", False)) if downlinkJson else False,
        "sib1CrcOk": bool(downlinkJson.get("sib1CrcOk", False)) if downlinkJson else False,
        "downlinkProbeStage": downlinkJson.get("stage") if downlinkJson else "not_run",
    }

    result = {
        "method": "fixed_rx_diagnostics",
        "input": {"signalPath": str(rxPath), "sampleCount": int(len(rxSignal))},
        "parameters": {
            "sampleRate": float(args.sample_rate),
            "scs": int(args.scs),
            "nfft": int(args.nfft),
            "cpProfile": "slot_head",
            "cpLengths": cpLengths,
            "nIdCell": int(args.n_id_cell),
            "nId2": nId2,
            "iSsbBar": int(args.i_ssb_bar),
            "ssbStart": int(args.ssb_start),
            "freqCompHz": float(args.freq_comp_hz),
            "ssbSubcarrierOffset": int(args.ssb_subcarrier_offset),
        },
        "sync": {
            "pssCorr": pssCorr,
            "sssScan": sssScan,
            "localTimingTop": localTiming[:10],
            "caseCStandardSsbOffsetSamples": caseCStandardSsbOffset,
            "caseCFrameStartEstimateSamples": frameStartEstimate,
        },
        "frequency": {
            "cpResidualCfo": cpResidual,
            "pssSssResidualCfo": pssSssResidual,
        },
        "channel": chStats,
        "pbch": {
            "bchCrcOk": bchCrc,
            "mib": mib,
            "evmVariants": {
                "currentDecoder": {
                    "evmPercent": float(best["evmPercent"]),
                    "noiseVarEstimate": float(best["noiseVarEstimate"]),
                },
                "rawPhaseMagnitudeInterp": _normalizeAndEvm(rawEq),
                "rawInterpAngleLoop": _normalizeAndEvm(rawAngleEq),
                "symbolMeanAngleLoop": _normalizeAndEvm(symMeanAngleEq),
            },
        },
        "coreset0": coreset0,
        "assessment": {
            "pbchMibLocallyProven": bool(bchCrc),
            "evmHealthyForQpskReference17p5": bool(float(best["evmPercent"]) < 17.5),
            "pdcchSib1Proven": bool(coreset0["pdcchCrcOk"] and coreset0["sib1CrcOk"]),
            "mainSuspects": [
                "PBCH DM-RS/channel quality is poor despite BCH CRC pass",
                "CORESET0/Point-A/frequency mapping is not proven because SI-RNTI PDCCH CRC fails",
                "Capture appears to start near SSB, so the first CORESET0 monitoring slot is partly before the file",
            ],
        },
    }

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    jsonPath = outDir / f"{args.output_prefix}_report.json"
    mdPath = outDir / f"{args.output_prefix}_report.md"
    jsonPath.write_text(json.dumps(_jsonable(result), ensure_ascii=False, indent=2), encoding="utf-8")
    mdPath.write_text(_renderMarkdown(result), encoding="utf-8")
    print(json.dumps({"json": str(jsonPath.resolve()), "markdown": str(mdPath.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
