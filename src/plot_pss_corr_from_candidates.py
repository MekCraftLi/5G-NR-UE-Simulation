import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pss.pssTemplateFactory import generatePssSequence


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot normalized PSS correlation curves for candidate parameter sets")
    p.add_argument("--result-json", type=str, required=True, help="Scan result JSON path")
    p.add_argument("--time-start", type=int, default=1, help="1-based plot start sample")
    p.add_argument("--time-end", type=int, default=15000, help="1-based plot end sample")
    p.add_argument("--max-candidates", type=int, default=20, help="Max number of candidates to plot")
    p.add_argument("--ratio-min", type=float, default=0.0, help="Filter by peakRatio >= this value")
    p.add_argument("--output-dir", type=str, default="output/pss_candidate_corr")
    return p.parse_args()


def _next_pow2(v: int) -> int:
    return 1 << (int(v) - 1).bit_length()


def _build_template(nfft: int, cp: int, n_id2: int, offset: int) -> np.ndarray:
    pss_start = int(nfft) // 2 - 63 + int(offset)
    if pss_start < 0 or pss_start + 127 > int(nfft):
        raise ValueError("PSS bins out of FFT range")
    grid = np.zeros(int(nfft), dtype=np.complex64)
    grid[pss_start:pss_start + 127] = generatePssSequence(int(n_id2))
    useful = np.fft.ifft(np.fft.ifftshift(grid)).astype(np.complex64)
    template = np.concatenate([useful[-int(cp):], useful]).astype(np.complex64)
    norm = float(np.linalg.norm(template))
    return (template / max(norm, 1e-12)).astype(np.complex64)


def _window_energy_sqrt(signal: np.ndarray, window_len: int) -> np.ndarray:
    power = np.abs(signal).astype(np.float64) ** 2
    prefix = np.concatenate([np.asarray([0.0]), np.cumsum(power, dtype=np.float64)])
    return np.sqrt(np.maximum(prefix[window_len:] - prefix[:-window_len], 0.0)).astype(np.float32)


def _score_template(signal_fft: np.ndarray, corr_nfft: int, template: np.ndarray, energy: np.ndarray) -> np.ndarray:
    kernel = np.conjugate(template[::-1]).astype(np.complex64)
    corr = np.fft.ifft(signal_fft * np.fft.fft(kernel, n=corr_nfft), n=corr_nfft)
    valid_start = len(template) - 1
    valid_end = valid_start + len(energy)
    mag = np.abs(corr[valid_start:valid_end]).astype(np.float32)
    score = np.zeros_like(mag)
    m = energy > 1e-12
    score[m] = mag[m] / energy[m]
    return score


def _top_peaks(score: np.ndarray, count: int, min_distance: int) -> list[dict]:
    work = score.copy()
    out = []
    for _ in range(int(count)):
        idx = int(np.argmax(work))
        val = float(work[idx])
        if val <= 0.0:
            break
        out.append({"timingOffset": int(idx), "score": val})
        left = max(0, idx - int(min_distance))
        right = min(len(work), idx + int(min_distance) + 1)
        work[left:right] = 0.0
    return out


def main() -> None:
    args = _parse_args()
    result_path = Path(args.result_json)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    input_path = Path(result["inputPath"])
    x = np.asarray(np.load(input_path), dtype=np.complex64).reshape(-1)
    time_start = max(1, int(args.time_start))
    time_end = max(time_start, int(args.time_end))
    plot_slice = slice(time_start - 1, time_end)
    x_axis = np.arange(time_start, time_end + 1, dtype=np.int32)

    candidates = result.get("topCandidates", [])
    candidates = [c for c in candidates if float(c.get("peakRatio", 0.0)) >= float(args.ratio_min)]
    candidates = sorted(candidates, key=lambda c: float(c.get("score", 0.0)), reverse=True)[: int(args.max_candidates)]

    out_dir = Path(args.output_dir) / result_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i, cand in enumerate(candidates, start=1):
        nfft = int(cand["nfft"])
        cp = int(cand["cp"])
        n_id2 = int(cand["nId2"])
        offset = int(cand["ssbSubcarrierOffset"])
        template = _build_template(nfft=nfft, cp=cp, n_id2=n_id2, offset=offset)
        template_len = len(template)

        needed = (time_end - 1) + template_len + 1
        if needed > len(x):
            continue
        signal = x[:needed].astype(np.complex64)
        corr_nfft = _next_pow2(len(signal) + template_len - 1)
        signal_fft = np.fft.fft(signal, n=corr_nfft).astype(np.complex64)
        energy = _window_energy_sqrt(signal, template_len)
        score = _score_template(signal_fft, corr_nfft, template, energy)
        score_plot = score[plot_slice]

        local_best_idx = int(np.argmax(score_plot))
        local_best_score = float(score_plot[local_best_idx])
        local_best_t1 = int(x_axis[local_best_idx])
        peaks = _top_peaks(score, count=3, min_distance=max(template_len // 2, 1))

        fig, ax = plt.subplots(1, 1, figsize=(14, 4.5), constrained_layout=True)
        ax.plot(x_axis, score_plot, linewidth=1.0, color="#0b5394")
        ax.axvline(local_best_t1, color="#cc0000", linestyle="--", linewidth=1.0)
        ax.set_title(
            f"PSS Corr | fs={result['sampleRate']/1e6:.2f}MHz scs={cand['scs']/1000:.0f}kHz "
            f"nfft={nfft} cp={cp}({cand['cpKind']}) off={offset} nId2={n_id2}"
        )
        ax.set_xlabel("Time index (1-based sample)")
        ax.set_ylabel("Normalized correlation")
        ax.grid(True, alpha=0.25)
        ax.text(
            0.01,
            0.98,
            f"best@window: t={local_best_t1} score={local_best_score:.6f}\n"
            f"scanScore={float(cand.get('score',0.0)):.6f} ratio={float(cand.get('peakRatio',0.0)):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

        name = (
            f"{i:02d}_scs{int(cand['scs'])}_nfft{nfft}_cp{cp}_{cand['cpKind']}"
            f"_off{offset}_nid2{n_id2}.png"
        )
        fig_path = out_dir / name
        fig.savefig(fig_path, dpi=170, bbox_inches="tight")
        plt.close(fig)

        written.append(
            {
                "index": i,
                "figure": str(fig_path.resolve()),
                "candidate": cand,
                "windowBest": {"time1Based": local_best_t1, "score": local_best_score},
                "globalTop3Peaks": peaks,
            }
        )

    summary = {
        "resultJson": str(result_path.resolve()),
        "inputPath": str(input_path.resolve()),
        "timeStart1Based": int(time_start),
        "timeEnd1Based": int(time_end),
        "ratioMin": float(args.ratio_min),
        "maxCandidates": int(args.max_candidates),
        "writtenCount": int(len(written)),
        "items": written,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=4), encoding="utf-8")
    print(str(out_dir.resolve()))
    print(str(summary_path.resolve()))
    print(f"written={len(written)}")


if __name__ == "__main__":
    main()

