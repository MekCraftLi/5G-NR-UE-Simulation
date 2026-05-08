import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pbch.pbchDecoder import PbchDecoder


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def _parseArgs(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce DigitalReceiver TX-reference PBCH EVM flow")
    parser.add_argument("--tx-path", type=str, default="data/txs0.npy")
    parser.add_argument("--rx-path", type=str, default=None)
    parser.add_argument("--output-prefix", type=str, default="digitalreceiver_repro")
    parser.add_argument("--nfft", type=int, default=2048)
    parser.add_argument("--cp", type=int, default=152)
    parser.add_argument("--symbol-length", type=int, default=2200)
    parser.add_argument("--ssb-bin-start", type=int, default=920, help="0-based fftshift bin start; MATLAB 921 -> 920")
    parser.add_argument("--pci", type=int, default=5)
    return parser.parse_args(argv)


def _loadComplexNpy(path: str) -> np.ndarray:
    data = np.load(path)
    return np.asarray(data, dtype=np.complex64).reshape(-1)


def _fftCorrelateLag(obs: np.ndarray, ref: np.ndarray) -> int:
    nfft = 1 << (len(obs) + len(ref) - 2).bit_length()
    corr = np.fft.ifft(np.fft.fft(obs, nfft) * np.fft.fft(np.conjugate(ref[::-1]), nfft))
    valid = np.abs(corr[: len(obs) + len(ref) - 1])
    peak = int(np.argmax(valid))
    return int(peak - (len(ref) - 1))


def _extractSsbGrid(signal: np.ndarray, start: int, nfft: int, cp: int, symbolLength: int, ssbBinStart: int) -> np.ndarray:
    grid = np.zeros((240, 4), dtype=np.complex64)
    for symbol in range(4):
        symbolStart = int(start) + symbol * int(symbolLength)
        usefulStart = symbolStart + int(cp)
        usefulEnd = usefulStart + int(nfft)
        if usefulStart < 0 or usefulEnd > len(signal):
            raise ValueError(
                f"SSB symbol out of range: symbol={symbol}, useful=[{usefulStart},{usefulEnd}), len={len(signal)}"
            )
        spectrum = np.fft.fftshift(np.fft.fft(signal[usefulStart:usefulEnd], int(nfft))).astype(np.complex64)
        grid[:, symbol] = spectrum[int(ssbBinStart):int(ssbBinStart) + 240]
    return grid


def _nearestQpsk(points: np.ndarray) -> np.ndarray:
    return ((np.sign(np.real(points)) + 1j * np.sign(np.imag(points))) / np.sqrt(2.0)).astype(np.complex64)


def _evmPercent(points: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    norm = np.sqrt(np.mean(np.abs(points) ** 2))
    pointsNorm = points / max(float(norm), 1e-12)
    ref = _nearestQpsk(pointsNorm)
    evm = 100.0 * np.sqrt(np.mean(np.abs(pointsNorm - ref) ** 2) / max(float(np.mean(np.abs(ref) ** 2)), 1e-12))
    return float(evm), pointsNorm.astype(np.complex64), ref.astype(np.complex64)


def _phaseCalibratedTxPbch(gridTx: np.ndarray, pci: int) -> tuple[np.ndarray, dict]:
    dmrsItems = PbchDecoder._dmrsReList(pci)
    dataItems = PbchDecoder._pbchDataReList(pci)
    dmrsSeq = PbchDecoder.generatePbchDmrs(pci, 0)

    dmrsOffsetBySymbol = {1: 0, 2: 60, 3: 84}
    eq = np.zeros(len(dataItems), dtype=np.complex64)
    evmBySymbol: dict[int, float] = {}

    for l in (1, 2, 3):
        dmrsForSymbol = [item for item in dmrsItems if item.l == l]
        dataForSymbol = [(idx, item) for idx, item in enumerate(dataItems) if item.l == l]
        scDmrs = np.asarray([item.k for item in dmrsForSymbol], dtype=np.float64)
        txDmrs = np.asarray([gridTx[item.k, item.l] for item in dmrsForSymbol], dtype=np.complex64)
        offset = dmrsOffsetBySymbol[l]
        refDmrs = dmrsSeq[offset:offset + len(dmrsForSymbol)]
        phase = np.angle(txDmrs / refDmrs)

        scData = np.asarray([item.k for _, item in dataForSymbol], dtype=np.float64)
        phaseInterp = np.interp(scData, scDmrs, phase).astype(np.float32)
        pts = np.asarray([gridTx[item.k, item.l] for _, item in dataForSymbol], dtype=np.complex64)
        ptsEq = (pts * np.exp(-1j * phaseInterp)).astype(np.complex64)
        for localIdx, (globalIdx, _) in enumerate(dataForSymbol):
            eq[globalIdx] = ptsEq[localIdx]
        evmBySymbol[l + 1] = _evmPercent(ptsEq)[0]

    evmTotal, eqNorm, ref = _evmPercent(eq)
    return eqNorm, {
        "evmPercent": evmTotal,
        "evmBySymbol": evmBySymbol,
        "reference": ref,
        "pbchCount": int(len(dataItems)),
    }


def _saveConstellation(eq: np.ndarray, evm: float, outputPrefix: str) -> Path:
    outDir = Path("output")
    outDir.mkdir(exist_ok=True)
    figPath = outDir / f"{outputPrefix}_pbch_final_constellation.png"
    qpsk = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(np.real(eq), np.imag(eq), s=18, alpha=0.75, label="PBCH")
    ax.scatter(np.real(qpsk), np.imag(qpsk), marker="x", s=140, c="red", linewidths=2.5, label="QPSK")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_xlabel("I")
    ax.set_ylabel("Q")
    ax.set_title(f"PBCH Constellation | DigitalReceiver repro | EVM={evm:.2f}%")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figPath, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return figPath


def main(argv: list[str] | None = None) -> None:
    args = _parseArgs(argv)
    tx = _loadComplexNpy(args.tx_path)
    if args.rx_path:
        rx = _loadComplexNpy(args.rx_path)
        lag = _fftCorrelateLag(rx, tx)
    else:
        rx = tx
        lag = 0

    gridTx = _extractSsbGrid(
        tx,
        start=0,
        nfft=args.nfft,
        cp=args.cp,
        symbolLength=args.symbol_length,
        ssbBinStart=args.ssb_bin_start,
    )
    _ = _extractSsbGrid(
        rx,
        start=lag,
        nfft=args.nfft,
        cp=args.cp,
        symbolLength=args.symbol_length,
        ssbBinStart=args.ssb_bin_start,
    )

    eq, metrics = _phaseCalibratedTxPbch(gridTx, args.pci)
    figPath = _saveConstellation(eq, float(metrics["evmPercent"]), args.output_prefix)

    out = {
        "method": "digitalreceiver_pbch_final_v2_tx_reference_repro",
        "txPath": str(Path(args.tx_path).resolve()),
        "rxPath": None if args.rx_path is None else str(Path(args.rx_path).resolve()),
        "lag": int(lag),
        "nfft": int(args.nfft),
        "cp": int(args.cp),
        "symbolLength": int(args.symbol_length),
        "ssbBinStart": int(args.ssb_bin_start),
        "ssbBinsMatlab": [int(args.ssb_bin_start + 1), int(args.ssb_bin_start + 240)],
        "pci": int(args.pci),
        "evmPercent": float(metrics["evmPercent"]),
        "evmBySymbol": {str(k): float(v) for k, v in metrics["evmBySymbol"].items()},
        "pbchCount": int(metrics["pbchCount"]),
        "constellation": str(figPath.resolve()),
    }
    outPath = Path("output") / f"{args.output_prefix}_result.json"
    outPath.write_text(json.dumps(out, ensure_ascii=False, indent=4), encoding="utf-8")

    logger.info("lag=%s", lag)
    logger.info(
        "EVM: S2=%.2f%%, S3=%.2f%%, S4=%.2f%%, Total=%.2f%%",
        out["evmBySymbol"]["2"],
        out["evmBySymbol"]["3"],
        out["evmBySymbol"]["4"],
        out["evmPercent"],
    )
    logger.info("Saved result: %s", outPath.resolve())
    logger.info("Saved constellation: %s", figPath.resolve())


if __name__ == "__main__":
    main()
