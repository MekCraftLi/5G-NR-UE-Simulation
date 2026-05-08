import argparse
import json
from pathlib import Path

import numpy as np

from common.config import createSsbConfigWithOverrides
from pbch.pbchDecoder import PbchDecoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PBCH DMRS filter over multiple PSS/SSS candidates")
    p.add_argument("--signal-path", type=str, default="data/rxSignal.npy")
    p.add_argument("--candidate-json", type=str, nargs="+", required=True, help="One or more SSS batch result JSON files")
    p.add_argument("--max-candidates-per-file", type=int, default=8)
    p.add_argument("--residual-freq-search-hz", type=float, default=2000.0)
    p.add_argument("--residual-freq-step-hz", type=float, default=100.0)
    p.add_argument("--output-prefix", type=str, default="rx_pbch_dmrs_filter")
    return p.parse_args()


def load_candidates(path: Path, limit: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("validRanked") or data.get("results") or []
    out = []
    if rows:
        rows = [r for r in rows if r.get("valid", True)]
        for row in rows[: int(limit)]:
            item = {
                "sourceFile": str(path.resolve()),
                "fs": float(row["fs"]),
                "scs": int(row["scs"]),
                "nfft": int(row["nfft"]),
                "timingOffset": int(row["timingOffset"]),
                "ssbSubcarrierOffset": int(row["ssbSubcarrierOffset"]),
                "nId2": int(row["nId2"]),
                "nIdCell": int(row["best"]["nIdCell"]),
                "sssSymbolStart": int(row["best"]["sssSymbolStart"]),
                "sssScore": float(row["best"]["score"]),
                "sssUniqueness": float(row["uniquenessRatio"]),
                "freqCompHz": float(row.get("freqCompHz", 0.0)),
                "expectedSssSymbolStart": int(row.get("expectedSssSymbolStart", row["best"]["sssSymbolStart"])),
                "deltaFromExpected": int(row["best"]["deltaFromExpected"]),
            }
            out.append(item)
        return out

    # Compatibility path A: scan_rx_pss_params output ("topCandidates")
    top = data.get("topCandidates") or []
    if not top:
        # Compatibility path B: converted simplified format ("top")
        top = data.get("top") or []
    fs = float(data.get("sampleRate", 0.0))
    for row in top[: int(limit)]:
        if "best" in row:
            # already in batch-like structure
            timing = row.get("timingOffset", row.get("bestRxTiming"))
            offset = row.get("ssbSubcarrierOffset", row.get("bestRxOffset"))
            nid2 = row.get("nId2", row.get("bestRxNid2"))
            score = row.get("sssScore", row.get("bestRxScore", row["best"]["score"]))
            item = {
                "sourceFile": str(path.resolve()),
                "fs": float(row["fs"]),
                "scs": int(row["scs"]),
                "nfft": int(row["nfft"]),
                "timingOffset": int(timing),
                "ssbSubcarrierOffset": int(offset),
                "nId2": int(nid2),
                "nIdCell": int(row["best"]["nIdCell"]),
                "sssSymbolStart": int(row["best"]["sssSymbolStart"]),
                "sssScore": float(score),
                "sssUniqueness": float(row.get("uniquenessRatio", 1.0)),
                "freqCompHz": float(row.get("freqCompHz", 0.0)),
                "expectedSssSymbolStart": int(row.get("expectedSssSymbolStart", row["best"]["sssSymbolStart"])),
                "deltaFromExpected": int(row["best"].get("deltaFromExpected", 0)),
            }
            out.append(item)
            continue

        val = row.get("validation") or {}
        nfft = int(row["nfft"])
        cp0 = int(row.get("cp", row.get("cp0", 0)))
        cp_kind = str(row.get("cpKind", "normal"))
        cp_other = int(round(144 * int(nfft) / 2048))
        if cp_kind == "long":
            sss_symbol_start = int(row["timingOffset"]) + (nfft + cp0) + (nfft + cp_other)
        else:
            sss_symbol_start = int(row["timingOffset"]) + 2 * (nfft + cp0)
        item = {
            "sourceFile": str(path.resolve()),
            "fs": float(row.get("fs", fs)),
            "scs": int(row["scs"]),
            "nfft": int(row["nfft"]),
            "timingOffset": int(row["timingOffset"]),
            "ssbSubcarrierOffset": int(row["ssbSubcarrierOffset"]),
            "nId2": int(row["nId2"]),
            "nIdCell": int(val.get("nIdCell", 3 * int(val.get("nId1", 0)) + int(row["nId2"]))),
            "sssSymbolStart": int(sss_symbol_start),
            "sssScore": float(val.get("sssFdScore", row.get("score", 0.0))),
            "sssUniqueness": float(1.0),
            "freqCompHz": float(val.get("freqHz", 0.0)),
            "expectedSssSymbolStart": int(sss_symbol_start),
            "deltaFromExpected": 0,
        }
        out.append(item)
    return out


def dedupe_candidates(rows: list[dict]) -> list[dict]:
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for r in rows:
        key = (
            r["fs"],
            r["scs"],
            r["nfft"],
            r["timingOffset"],
            r["ssbSubcarrierOffset"],
            r["nId2"],
            r["nIdCell"],
            r["sssSymbolStart"],
            round(r["freqCompHz"], 3),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def evaluate_candidate(rx: np.ndarray, c: dict, residual_search_hz: float, residual_step_hz: float) -> dict:
    cfg = createSsbConfigWithOverrides(
        sampleRate=float(c["fs"]),
        subcarrierSpacing=int(c["scs"]),
        fftSize=int(c["nfft"]),
        ssbSubcarrierOffset=int(c["ssbSubcarrierOffset"]),
    )
    dec = PbchDecoder(cfg)
    sss_result = {
        "nIdCell": int(c["nIdCell"]),
        "bestSymbolStart": int(c["sssSymbolStart"]),
        "freqCompHz": float(c["freqCompHz"]),
    }
    pss_result = {
        "timingOffset": int(c["timingOffset"]),
    }
    pbch = dec.decodePbch(
        rxSignal=rx,
        sssResult=sss_result,
        pssResult=pss_result,
        freqCompHz=float(c["freqCompHz"]),
        residualFreqSearchHz=float(residual_search_hz),
        residualFreqStepHz=float(residual_step_hz),
    )
    return {
        "candidate": c,
        "pbch": {
            "nIdCell": int(pbch["nIdCell"]),
            "iSsbBar": int(pbch["iSsbBar"]),
            "ssbStart": int(pbch["ssbStart"]),
            "freqCompHz": float(pbch["freqCompHz"]),
            "evmPercent": float(pbch["evmPercent"]),
            "evmPass10Percent": bool(pbch["evmPass10Percent"]),
            "dmrsPower": float(pbch["dmrsPower"]),
            "dataPower": float(pbch["dataPower"]),
            "channelStd": float(pbch["channelStd"]),
            "candidateCount": int(pbch["candidateCount"]),
            "rankingMode": str(pbch.get("rankingMode", "")),
            "dmrsCorr": float(pbch.get("dmrsCorr", 0.0)),
            "dmrsCorrNorm": float(pbch.get("dmrsCorrNorm", 0.0)),
            "dmrsEvmPercent": float(pbch.get("dmrsEvmPercent", 0.0)),
            "noiseVarEstimate": float(pbch.get("noiseVarEstimate", 0.0)),
            "dmrsSnrDb": float(pbch.get("dmrsSnrDb", 0.0)),
            "pbchHardEvmPercent": float(pbch.get("pbchHardEvmPercent", pbch["evmPercent"])),
            "topCandidates": pbch["topCandidates"],
        },
    }


def main() -> None:
    args = parse_args()
    rx = np.asarray(np.load(args.signal_path), dtype=np.complex64).reshape(-1)

    all_rows: list[dict] = []
    for fp in args.candidate_json:
        p = Path(fp)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        rows = load_candidates(p, int(args.max_candidates_per_file))
        all_rows.extend(rows)

    rows = dedupe_candidates(all_rows)
    print(f"Loaded {len(all_rows)} candidates, unique={len(rows)}")

    results: list[dict] = []
    for i, c in enumerate(rows, 1):
        try:
            r = evaluate_candidate(
                rx=rx,
                c=c,
                residual_search_hz=float(args.residual_freq_search_hz),
                residual_step_hz=float(args.residual_freq_step_hz),
            )
            results.append(r)
            print(
                f"[{i}/{len(rows)}] fs={c['fs']/1e6:.2f} scs={c['scs']/1e3:.0f}k nfft={c['nfft']} "
                f"nid2={c['nId2']} ncell={c['nIdCell']} sss={c['sssScore']:.4f}/{c['sssUniqueness']:.3f} "
                f"-> PBCH EVM={r['pbch']['evmPercent']:.2f}% pass={r['pbch']['evmPass10Percent']}"
            )
        except Exception as e:
            results.append({"candidate": c, "error": str(e)})
            print(
                f"[{i}/{len(rows)}] fs={c['fs']/1e6:.2f} scs={c['scs']/1e3:.0f}k nfft={c['nfft']} "
                f"nid2={c['nId2']} ncell={c['nIdCell']} -> ERROR: {e}"
            )

    valid = [r for r in results if "pbch" in r]
    ranked = sorted(
        valid,
        key=lambda r: (
            0 if r["pbch"]["evmPass10Percent"] else 1,
            r["pbch"]["evmPercent"],
            -r["candidate"]["sssUniqueness"],
            -r["candidate"]["sssScore"],
        ),
    )
    best = ranked[0] if ranked else None

    out = {
        "signalPath": str(Path(args.signal_path).resolve()),
        "candidateJson": [str((Path(p).resolve() if Path(p).is_absolute() else (Path.cwd() / p).resolve())) for p in args.candidate_json],
        "maxCandidatesPerFile": int(args.max_candidates_per_file),
        "residualFreqSearchHz": float(args.residual_freq_search_hz),
        "residualFreqStepHz": float(args.residual_freq_step_hz),
        "totalCandidates": int(len(rows)),
        "evaluatedCount": int(len(valid)),
        "passCountEvm10": int(sum(1 for r in valid if r["pbch"]["evmPass10Percent"])),
        "bestOverall": best,
        "ranked": ranked,
        "allResults": results,
    }

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.output_prefix}_result.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path.resolve()}")
    if best is not None:
        print(
            json.dumps(
                {
                    "bestCandidate": best["candidate"],
                    "bestPbch": best["pbch"],
                    "output": str(out_path.resolve()),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
