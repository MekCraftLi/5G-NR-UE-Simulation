import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert scan_rx_pss_params output to batch_sss/pbch candidate format")
    p.add_argument("--input-json", type=str, required=True)
    p.add_argument("--score-field", type=str, default="bestRxScore")
    p.add_argument("--timing-field", type=str, default="bestRxTiming")
    p.add_argument("--offset-field", type=str, default="bestRxOffset")
    p.add_argument("--nid2-field", type=str, default="bestRxNid2")
    p.add_argument("--output-json", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    rows = data.get("topCandidates", [])
    mode = "topCandidates"
    if not rows:
        rows = data.get("rankedByJointBlindScore") or data.get("results") or []
        mode = "jointResults"
    out_rows = []
    if mode == "topCandidates":
        for r in rows:
            val = r.get("validation") or {}
            out_rows.append(
                {
                    "fs": float(data["sampleRate"]),
                    "scs": int(r["scs"]),
                    "nfft": int(r["nfft"]),
                    "cp0": int(r["cp"]),
                    "cpOther": int(round(144 * int(r["nfft"]) / 2048)),
                    args.score_field: float(r["score"]),
                    args.timing_field: int(r["timingOffset"]),
                    args.offset_field: int(r["ssbSubcarrierOffset"]),
                    args.nid2_field: int(r["nId2"]),
                    "validation_nId1": int(val.get("nId1", -1)),
                    "validation_nIdCell": int(val.get("nIdCell", -1)),
                    "validation_pssFdScore": float(val.get("pssFdScore", 0.0)),
                    "validation_sssFdScore": float(val.get("sssFdScore", 0.0)),
                    "validation_freqHz": float(val.get("freqHz", 0.0)),
                }
            )
    else:
        for r in rows:
            if not r.get("valid", True):
                continue
            best = r.get("best") or {}
            expected = r.get("expectedTimingBest") or {}
            out_rows.append(
                {
                    "fs": float(r["fs"]),
                    "scs": int(r["scs"]),
                    "nfft": int(r["nfft"]),
                    "cp0": int(r.get("cp0", 0)),
                    "cpOther": int(r.get("cpOther", round(144 * int(r["nfft"]) / 2048))),
                    args.score_field: float(r.get("pssScore", r.get("jointBlindScore", 0.0))),
                    args.timing_field: int(r["timingOffset"]),
                    args.offset_field: int(r["ssbSubcarrierOffset"]),
                    args.nid2_field: int(r["nId2"]),
                    "best": {
                        "nId1": int(best.get("nId1", expected.get("nId1", -1))),
                        "nIdCell": int(best.get("nIdCell", expected.get("nIdCell", -1))),
                        "sssSymbolStart": int(best.get("sssSymbolStart", expected.get("sssSymbolStart", 0))),
                        "score": float(best.get("score", expected.get("score", 0.0))),
                        "deltaFromExpected": int(best.get("deltaFromExpected", 0)),
                    },
                    "uniquenessRatio": float(r.get("uniquenessRatio", 1.0)),
                    "freqCompHz": float(best.get("freqHz", expected.get("freqHz", 0.0))),
                    "expectedSssSymbolStart": int(r.get("expectedSssSymbolStart", expected.get("sssSymbolStart", 0))),
                }
            )

    out = {
        "source": str(Path(args.input_json).resolve()),
        "mode": mode,
        "top": out_rows,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_path.resolve()} rows={len(out_rows)}")


if __name__ == "__main__":
    main()
