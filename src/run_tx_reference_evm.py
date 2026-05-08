import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolvePath(pathText: str) -> Path:
    path = Path(pathText)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _loadComplexNpy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = np.load(str(path))
    if not np.iscomplexobj(data):
        data = data.astype(np.float32) + 0j
    return np.asarray(data.reshape(-1), dtype=np.complex64)


def _estimateLag(reference: np.ndarray, observed: np.ndarray, maxSamples: int | None = None) -> int:
    ref = reference
    obs = observed
    if maxSamples is not None and maxSamples > 0:
        ref = ref[: min(len(ref), int(maxSamples))]
        obs = obs[: min(len(obs), int(maxSamples))]
    nfft = 1 << (len(ref) + len(obs) - 2).bit_length()
    corr = np.fft.ifft(np.fft.fft(obs, nfft) * np.fft.fft(np.conjugate(ref[::-1]), nfft))
    return int(np.argmax(np.abs(corr)) - (len(ref) - 1))


def _alignedSlices(reference: np.ndarray, observed: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag >= 0:
        length = min(len(reference), len(observed) - lag)
        return reference[:length], observed[lag:lag + length]
    offset = -lag
    length = min(len(reference) - offset, len(observed))
    return reference[offset:offset + length], observed[:length]


def _estimateCommonGain(reference: np.ndarray, observed: np.ndarray) -> complex:
    denom = np.vdot(reference, reference)
    if abs(denom) <= 1e-12:
        return 0.0 + 0.0j
    return complex(np.vdot(reference, observed) / denom)


def _evmPercent(reference: np.ndarray, observed: np.ndarray, gain: complex) -> float:
    ideal = gain * reference
    error = observed - ideal
    return float(100.0 * np.sqrt(np.mean(np.abs(error) ** 2) / (np.mean(np.abs(ideal) ** 2) + 1e-12)))


def _saveArtifacts(result: dict, reference: np.ndarray, observed: np.ndarray, outputPrefix: str) -> None:
    outDir = Path(__file__).resolve().parent.parent / "output"
    outDir.mkdir(exist_ok=True)
    prefix = f"{outputPrefix}_" if outputPrefix else "tx_reference_"

    gain = complex(result["gainReal"], result["gainImag"])
    ideal = gain * reference
    error = observed - ideal
    sampleCount = min(5000, len(reference))

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(np.real(observed[:sampleCount]), label="observed I", linewidth=0.8)
    axes[0, 0].plot(np.real(ideal[:sampleCount]), label="aligned reference I", linewidth=0.8, alpha=0.75)
    axes[0, 0].set_title("Time-Domain I Overlay")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(20.0 * np.log10(np.abs(error[:sampleCount]) + 1e-12), linewidth=0.8)
    axes[0, 1].set_title("Error Magnitude")
    axes[0, 1].set_ylabel("dB")
    axes[0, 1].grid(True, alpha=0.3)

    decim = max(1, len(observed) // 20000)
    axes[1, 0].scatter(np.real(observed[::decim]), np.imag(observed[::decim]), s=2, alpha=0.35)
    axes[1, 0].set_title("Observed IQ Samples")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].scatter(np.real(error[::decim]), np.imag(error[::decim]), s=2, alpha=0.35)
    axes[1, 1].set_title(f"Error IQ | EVM={result['evmPercent']:.6f}%")
    axes[1, 1].set_aspect("equal", adjustable="box")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    figPath = outDir / f"{prefix}evm.png"
    fig.savefig(figPath, dpi=160, bbox_inches="tight")
    plt.close(fig)

    npzPath = outDir / f"{prefix}evm_aligned.npz"
    np.savez_compressed(
        npzPath,
        reference=reference.astype(np.complex64),
        observed=observed.astype(np.complex64),
        ideal=ideal.astype(np.complex64),
        error=error.astype(np.complex64),
    )

    result["savedFiles"] = {
        "figure": str(figPath.resolve()),
        "alignedData": str(npzPath.resolve()),
    }
    jsonPath = outDir / f"{prefix}evm_result.json"
    jsonPath.write_text(json.dumps(result, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info("EVM figure saved: %s", figPath.resolve())
    logger.info("EVM result saved: %s", jsonPath.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute waveform EVM against data/txs0.npy TX reference")
    parser.add_argument("--reference", default="data/txs0.npy", help="Reference .npy waveform")
    parser.add_argument("--observed", default="data/txsig0_frame0.npy", help="Observed .npy waveform to evaluate")
    parser.add_argument("--output-prefix", default="txs0_reference", help="Output artifact prefix")
    parser.add_argument("--max-lag-samples", type=int, default=200000, help="Limit samples used for lag estimation")
    args = parser.parse_args()

    referencePath = _resolvePath(args.reference)
    observedPath = _resolvePath(args.observed)
    reference = _loadComplexNpy(referencePath)
    observed = _loadComplexNpy(observedPath)

    lag = _estimateLag(reference, observed, maxSamples=args.max_lag_samples)
    refAligned, obsAligned = _alignedSlices(reference, observed, lag)
    if len(refAligned) <= 0:
        raise RuntimeError(f"No overlap after lag alignment: lag={lag}")

    gain = _estimateCommonGain(refAligned, obsAligned)
    evm = _evmPercent(refAligned, obsAligned, gain)
    result = {
        "method": "time_domain_reference_evm",
        "referencePath": str(referencePath),
        "observedPath": str(observedPath),
        "referenceSamples": int(len(reference)),
        "observedSamples": int(len(observed)),
        "alignedSamples": int(len(refAligned)),
        "lagSamples": int(lag),
        "gainReal": float(np.real(gain)),
        "gainImag": float(np.imag(gain)),
        "evmPercent": float(evm),
        "evmPass10Percent": bool(evm < 10.0),
    }
    logger.info(
        "Reference EVM: observed=%s, lag=%d, gain=%.6g%+.6gj, EVM=%.9f%% (%s <10%%)",
        observedPath.name,
        lag,
        np.real(gain),
        np.imag(gain),
        evm,
        "PASS" if evm < 10.0 else "FAIL",
    )
    _saveArtifacts(result, refAligned, obsAligned, outputPrefix=args.output_prefix)


if __name__ == "__main__":
    main()
