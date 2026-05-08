import argparse
import json
from pathlib import Path

import numpy as np

from common.config import createSsbConfigWithOverrides
from pbch.pbchBchDecoder import PbchBchDecoder
from pbch.pbchDecoder import PbchDecoder


def _parseIntList(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _parseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast fixed/narrow PBCH reproduction around the known 45% EVM candidate."
    )
    parser.add_argument("--signal-path", type=str, default="data/rxSignal.npy")
    parser.add_argument("--output-prefix", type=str, default="pbch_narrow_repro")
    parser.add_argument("--sample-rate", type=float, default=122880000.0)
    parser.add_argument("--scs", type=int, default=30000)
    parser.add_argument("--nfft", type=int, default=4096)
    parser.add_argument("--ssb-subcarrier-offset", type=int, default=-7)
    parser.add_argument("--n-id-cell", type=int, default=0)
    parser.add_argument("--i-ssb-bars", type=str, default="0")
    parser.add_argument("--ssb-start", type=int, default=4)
    parser.add_argument("--start-radius", type=int, default=0)
    parser.add_argument("--freq-comp-hz", type=float, default=30308.0)
    parser.add_argument("--freq-search-hz", type=float, default=0.0)
    parser.add_argument("--freq-step-hz", type=float, default=2.0)
    parser.add_argument("--cp-profiles", type=str, default="slot_head")
    parser.add_argument(
        "--ranking-mode",
        choices=["evm_guarded", "hard_evm", "dmrs_only"],
        default="hard_evm",
    )
    parser.add_argument("--decode-bch", action="store_true")
    return parser.parse_args()


def _freqGrid(center: float, searchHz: float, stepHz: float) -> np.ndarray:
    if searchHz <= 0.0:
        return np.asarray([float(center)], dtype=np.float64)
    step = max(float(stepHz), 1e-9)
    return float(center) + np.arange(-abs(searchHz), abs(searchHz) + step / 2.0, step, dtype=np.float64)


def _jsonableCandidate(item: dict) -> dict:
    return {
        "nIdCell": int(item["nIdCell"]),
        "iSsbBar": int(item["iSsbBar"]),
        "ssbStart": int(item["ssbStart"]),
        "freqCompHz": float(item["freqCompHz"]),
        "cpProfile": str(item["cpProfile"]),
        "evmPercent": float(item["evmPercent"]),
        "dmrsCorr": float(item["dmrsCorr"]),
        "dmrsCorrNorm": float(item["dmrsCorrNorm"]),
        "dmrsEvmPercent": float(item["dmrsEvmPercent"]),
        "noiseVarEstimate": float(item["noiseVarEstimate"]),
        "dmrsSnrDb": float(item["dmrsSnrDb"]),
        "pbchHardEvmPercent": float(item["evmPercent"]),
    }


def main() -> None:
    args = _parseArgs()
    rxPath = Path(args.signal_path)
    if not rxPath.is_absolute():
        rxPath = (Path.cwd() / rxPath).resolve()
    rxSignal = np.asarray(np.load(rxPath), dtype=np.complex64).reshape(-1)

    cfg = createSsbConfigWithOverrides(
        sampleRate=float(args.sample_rate),
        subcarrierSpacing=int(args.scs),
        fftSize=int(args.nfft),
        ssbSubcarrierOffset=int(args.ssb_subcarrier_offset),
    )
    decoder = PbchDecoder(cfg, ssbIndexCandidates=_parseIntList(args.i_ssb_bars))
    cpMap = {name: vals for name, vals in decoder._cpLengthProfiles()}
    requestedProfiles = [p.strip() for p in str(args.cp_profiles).split(",") if p.strip()]

    candidates: list[dict] = []
    for cpProfile in requestedProfiles:
        if cpProfile not in cpMap:
            raise ValueError(f"Unknown cpProfile={cpProfile}; valid={sorted(cpMap)}")
        cpLengths = [int(v) for v in cpMap[cpProfile]]
        for ssbStart in range(int(args.ssb_start) - int(args.start_radius), int(args.ssb_start) + int(args.start_radius) + 1):
            if ssbStart < 0:
                continue
            for freqHz in _freqGrid(float(args.freq_comp_hz), float(args.freq_search_hz), float(args.freq_step_hz)):
                for iSsbBar in _parseIntList(args.i_ssb_bars):
                    item = decoder._evaluateCandidate(
                        rxSignal=rxSignal,
                        nIdCell=int(args.n_id_cell),
                        ssbStart=int(ssbStart),
                        freqCompHz=float(freqHz),
                        iSsbBar=int(iSsbBar),
                        cpProfileName=str(cpProfile),
                        cpLengths=cpLengths,
                    )
                    candidates.append(item)

    if not candidates:
        raise RuntimeError("No PBCH candidates evaluated")

    ranked = sorted(candidates, key=lambda x: decoder._candidateRankKey(x, mode=str(args.ranking_mode)), reverse=True)
    best = ranked[0]
    result = _jsonableCandidate(best)
    result.update(
        {
            "method": "pbch_fixed_narrow_repro",
            "signalPath": str(rxPath),
            "sampleRate": float(args.sample_rate),
            "scs": int(args.scs),
            "nfft": int(args.nfft),
            "ssbSubcarrierOffset": int(args.ssb_subcarrier_offset),
            "candidateCount": int(len(candidates)),
            "rankingMode": str(args.ranking_mode),
            "topCandidates": [_jsonableCandidate(item) for item in ranked[:10]],
        }
    )

    if args.decode_bch:
        bch = PbchBchDecoder().decode(
            pbchEq=best["pbchEq"],
            nIdCell=int(best["nIdCell"]),
            iSsbBar=int(best["iSsbBar"]),
            noiseVar=float(best.get("noiseVarEstimate", 1.0)),
            outputPrefix=str(args.output_prefix),
        )
        result["bchDecode"] = bch

    pbchResult = {
        "method": "pbch_fixed_narrow_repro",
        "nIdCell": int(best["nIdCell"]),
        "iSsbBar": int(best["iSsbBar"]),
        "ssbStart": int(best["ssbStart"]),
        "freqCompHz": float(best["freqCompHz"]),
        "cpProfile": str(best["cpProfile"]),
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
        "rankingMode": str(args.ranking_mode),
        "dmrsCorr": float(best["dmrsCorr"]),
        "dmrsCorrNorm": float(best["dmrsCorrNorm"]),
        "dmrsEvmPercent": float(best["dmrsEvmPercent"]),
        "noiseVarEstimate": float(best["noiseVarEstimate"]),
        "dmrsSnrDb": float(best["dmrsSnrDb"]),
        "pbchHardEvmPercent": float(best["evmPercent"]),
        "topCandidates": result["topCandidates"],
        "pbchEq": best["pbchEq"],
        "pbchHardRef": best["pbchHardRef"],
        "hardBits": best["hardBits"],
        "dataRe": np.asarray([[item.k, item.l] for item in best["dataItems"]], dtype=np.int16),
        "dmrsRe": np.asarray([[item.k, item.l] for item in best["dmrsItems"]], dtype=np.int16),
    }
    if "bchDecode" in result:
        pbchResult["bchDecode"] = result["bchDecode"]

    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    jsonPath = outDir / f"{args.output_prefix}_result.json"
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    PbchDecoder.saveArtifacts(pbchResult, outputPrefix=str(args.output_prefix))

    print(json.dumps({"result": result, "jsonPath": str(jsonPath.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
