import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pbch.pbchDecoder import PbchDecoder
from common.config import createSsbConfigWithOverrides


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce DigitalReceiver/pbch_final_v2.m style fixed-parameter evaluation")
    p.add_argument("--rx-path", type=str, default="data/rxSignal.npy")
    p.add_argument("--tx-path", type=str, default="data/txs0.npy")
    p.add_argument("--nfft", type=int, default=2048)
    p.add_argument("--cp", type=int, default=152)
    p.add_argument("--sym-len", type=int, default=2200)
    p.add_argument("--ssb-start-bin-1based", type=int, default=921)
    p.add_argument("--pci", type=int, default=5)
    p.add_argument("--sample-rate", type=float, default=30.72e6)
    p.add_argument("--scs", type=int, default=15000)
    p.add_argument("--output-prefix", type=str, default="pbch_v2_style_repro")
    return p.parse_args()


def qpsk_evm_percent(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.complex64).reshape(-1)
    x_n = x / np.sqrt(max(float(np.mean(np.abs(x) ** 2)), 1e-12))
    ref = (np.sign(np.real(x_n)) + 1j * np.sign(np.imag(x_n))) / np.sqrt(2.0)
    evm = 100.0 * np.sqrt(np.mean(np.abs(x_n - ref) ** 2) / max(np.mean(np.abs(ref) ** 2), 1e-12))
    return float(evm)


def main() -> None:
    args = parse_args()
    rx = np.asarray(np.load(args.rx_path), dtype=np.complex128).reshape(-1)
    tx = np.asarray(np.load(args.tx_path), dtype=np.complex128).reshape(-1)

    # xcorr(rx, tx) style lag
    corr = signal.correlate(rx, tx, mode="full", method="fft")
    lags = np.arange(-len(tx) + 1, len(rx), dtype=np.int64)
    lag = int(lags[int(np.argmax(np.abs(corr)))])

    nfft = int(args.nfft)
    cp = int(args.cp)
    sym_len = int(args.sym_len)
    ssb_start_0b = int(args.ssb_start_bin_1based) - 1
    ssb_bins = np.arange(ssb_start_0b, ssb_start_0b + 240, dtype=np.int32)
    if ssb_bins[-1] >= nfft:
        raise ValueError("SSB bins exceed NFFT")

    ssb_tx = np.zeros((240, 4), dtype=np.complex64)
    ssb_rx = np.zeros((240, 4), dtype=np.complex64)
    for s in range(4):
        st = s * sym_len
        tx_use = tx[st + cp: st + cp + nfft]
        fr_tx = np.fft.fftshift(np.fft.fft(tx_use, nfft))
        ssb_tx[:, s] = fr_tx[ssb_bins]

        rs = lag + st + cp
        re = rs + nfft
        if rs < 0 or re > len(rx):
            raise ValueError(f"RX symbol range out of bounds for s={s}, rs={rs}, re={re}")
        fr_rx = np.fft.fftshift(np.fft.fft(rx[rs:re], nfft))
        ssb_rx[:, s] = fr_rx[ssb_bins]

    # v2-style self-cancel path: eq_i = tx_val then DMRS phase correction
    # We keep it explicit to compare with blind chain.
    try:
        import matlab.engine  # noqa
    except Exception:
        pass

    # For a robust comparable metric, compute "oracle" EVM directly from TX PBCH RE
    cfg = createSsbConfigWithOverrides(
        sampleRate=float(args.sample_rate),
        subcarrierSpacing=int(args.scs),
        fftSize=int(args.nfft),
        ssbSubcarrierOffset=int(ssb_start_0b - (nfft // 2 - 120)),
    )
    dec = PbchDecoder(cfg)
    n_id_cell = int(args.pci)
    dmrs_items = dec._dmrsReList(n_id_cell)  # noqa: SLF001
    data_items = dec._pbchDataReList(n_id_cell)  # noqa: SLF001
    dmrs_seq = dec.generatePbchDmrs(n_id_cell, 0)

    # oracle equalized PBCH = TX PBCH at data RE
    tx_pbch = np.asarray([ssb_tx[it.k, it.l] for it in data_items], dtype=np.complex64)
    oracle_evm = qpsk_evm_percent(tx_pbch)

    # MATLAB pbch_final_v2-style branch:
    # 1) H_tx = TX_DMRS / REF_DMRS (symbol-wise slicing 60/24/60)
    # 2) phase interpolation from DMRS SC to PBCH SC in each symbol (2/3/4 => l=1/2/3)
    # 3) eq_i = tx_val, then apply exp(-j*phase_interp)
    eq_v2 = np.zeros(len(data_items), dtype=np.complex64)
    data_by_symbol: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for i, it in enumerate(data_items):
        if it.l in data_by_symbol:
            data_by_symbol[it.l].append(i)
    dmrs_by_symbol: dict[int, list[int]] = {1: [], 2: [], 3: []}
    for i, it in enumerate(dmrs_items):
        if it.l in dmrs_by_symbol:
            dmrs_by_symbol[it.l].append(i)

    # MATLAB symbol numbering in script is 2/3/4 for PBCH symbols.
    # Corresponding l indices here are 1/2/3.
    ref_ranges = {
        1: (0, 60),    # 1:60
        2: (60, 84),   # 61:84
        3: (84, 144),  # 85:144
    }
    for l_idx in (1, 2, 3):
        dm_idx = dmrs_by_symbol[l_idx]
        pb_idx = data_by_symbol[l_idx]
        if len(dm_idx) == 0 or len(pb_idx) == 0:
            continue
        sc_d = np.asarray([dmrs_items[i].k for i in dm_idx], dtype=np.float64)
        tx_dmrs_sym = np.asarray([ssb_tx[dmrs_items[i].k, dmrs_items[i].l] for i in dm_idx], dtype=np.complex64)
        rs, re = ref_ranges[l_idx]
        ref_dmrs_sym = dmrs_seq[rs:re].astype(np.complex64)
        if len(ref_dmrs_sym) != len(tx_dmrs_sym):
            raise ValueError(
                f"DMRS count mismatch at l={l_idx}: tx_dmrs={len(tx_dmrs_sym)}, ref_dmrs={len(ref_dmrs_sym)}"
            )
        h_tx = tx_dmrs_sym / np.where(np.abs(ref_dmrs_sym) > 1e-12, ref_dmrs_sym, 1.0 + 0j)
        ph_dmrs = np.angle(h_tx).astype(np.float64)

        sc_p = np.asarray([data_items[i].k for i in pb_idx], dtype=np.float64)
        ph_interp = np.interp(sc_p, sc_d, ph_dmrs).astype(np.float64)
        tx_pbch_sym = np.asarray([ssb_tx[data_items[i].k, data_items[i].l] for i in pb_idx], dtype=np.complex64)
        eq_sym = tx_pbch_sym * np.exp(-1j * ph_interp).astype(np.complex64)
        for local_i, global_i in enumerate(pb_idx):
            eq_v2[global_i] = eq_sym[local_i]

    v2_style_evm = qpsk_evm_percent(eq_v2)
    evm_sym = {}
    for l_idx, label in zip((1, 2, 3), ("sym2", "sym3", "sym4")):
        idx = data_by_symbol[l_idx]
        if len(idx) > 0:
            evm_sym[label] = qpsk_evm_percent(eq_v2[np.asarray(idx, dtype=np.int32)])

    # blind-like at fixed params: derive channel from RX/TX DMRS and equalize RX PBCH
    tx_dmrs = np.asarray([ssb_tx[it.k, it.l] for it in dmrs_items], dtype=np.complex64)
    rx_dmrs = np.asarray([ssb_rx[it.k, it.l] for it in dmrs_items], dtype=np.complex64)
    h_dmrs = rx_dmrs / np.where(np.abs(tx_dmrs) > 1e-9, tx_dmrs, 1.0 + 0j)
    h_data = dec._interpolateChannel(dmrs_items, h_dmrs, data_items)  # noqa: SLF001
    rx_pbch = np.asarray([ssb_rx[it.k, it.l] for it in data_items], dtype=np.complex64)
    eq_pbch = rx_pbch / np.where(np.abs(h_data) > 1e-9, h_data, 1.0 + 0j)
    blind_fixed_evm = qpsk_evm_percent(eq_pbch)

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    fig_path = out_dir / f"{args.output_prefix}_constellation.png"
    json_path = out_dir / f"{args.output_prefix}.json"

    q = np.asarray([1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j], dtype=np.complex64) / np.sqrt(2.0)
    fig, ax = plt.subplots(figsize=(7, 7))
    eqn = eq_pbch / np.sqrt(max(float(np.mean(np.abs(eq_pbch) ** 2)), 1e-12))
    ax.scatter(np.real(eqn), np.imag(eqn), s=10, alpha=0.6, label="RX eq PBCH (fixed params)")
    ax.scatter(np.real(q), np.imag(q), marker="x", s=120, c="red", linewidths=2, label="QPSK")
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    lim = max(1.2, float(np.max(np.abs(np.concatenate([np.real(eqn), np.imag(eqn)])))) * 1.2)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Fixed-parameter PBCH eq | EVM={blind_fixed_evm:.2f}%")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    out = {
        "rxPath": str(Path(args.rx_path).resolve()),
        "txPath": str(Path(args.tx_path).resolve()),
        "nfft": int(args.nfft),
        "cp": int(args.cp),
        "symLen": int(args.sym_len),
        "ssbBins1Based": [int(args.ssb_start_bin_1based), int(args.ssb_start_bin_1based + 239)],
        "ssbSubcarrierOffset0Based": int(ssb_start_0b - (nfft // 2 - 120)),
        "pci": int(args.pci),
        "lagFromXcorrRxTx": int(lag),
        "oracleTxPbchEvmPercent": float(oracle_evm),
        "v2StyleTxPhaseCalibratedEvmPercent": float(v2_style_evm),
        "v2StylePerSymbolEvmPercent": evm_sym,
        "blindFixedParamPbchEvmPercent": float(blind_fixed_evm),
        "figure": str(fig_path.resolve()),
    }
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
