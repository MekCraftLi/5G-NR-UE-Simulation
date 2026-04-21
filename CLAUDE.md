# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 5G NR (New Radio) physical layer signal processing project implementing cell search and synchronization procedures per 3GPP TS 38.211 / 38.101-1 specifications. The project processes received baseband I/Q signals to detect Primary Synchronization Signals (PSS) and perform initial cell search.

## Running

```bash
cd src
python main.py
```

Input signal data is loaded from `data/rxSignal.npy` (NumPy binary format). The path in `main.py` is relative (`../data/rxSignal.npy`), so run from `src/`.

Dependencies: `numpy`, `scipy`, `matplotlib`

## Architecture

All source code is in `src/`. The pipeline follows the 5G NR cell search procedure:

1. **`main.py`** — Entry point. Defines the three key parameters (`SAMPLE_RATE`, `SCS`, `N_RB`) and drives the pipeline.
2. **`config.py`** — Shared dataclass configurations: `SsbConfig` (SSB physical layer params: FFT size, CP length, subcarrier spacing, sample rate, target band raster) and `BandRasterConfig`/`BandConstants` (per-band GSCN ranges per 3GPP TS 38.101-1 Table 5.4.3.3-1).
3. **`gscn.py`** — GSCN (Global Synchronization Raster Channel Number) to absolute frequency conversion. Implements the three-range formula from TS 38.101-1 Table 5.4.3.1-1 (<3 GHz, 3–24.25 GHz, >24.25 GHz).
4. **`cp_manager.py`** — Cyclic Prefix length computation per OFDM symbol index. Distinguishes long CP (symbol 0) vs normal CP, scaled from the 2048-FFT baseline.
5. **`synchronization.py`** — Core PSS detection: generates local m-sequence PSS templates for all 3 sector IDs (`N_ID_2`), then performs cross-correlation-based blind search across candidate GSCN frequencies. Returns timing offset, `N_ID_2`, frequency offset, and a 2D search matrix.
6. **`visualizer.py`** — Matplotlib visualization: 2D heatmap of PSS search (GSCN vs timing) and 1D correlation peak profile.

## Key Conventions

- Code comments and variable names are in Chinese — maintain this style when modifying.
- Naming uses PascalCase for classes, constants, and dataclass fields (not snake_case). Follow existing naming patterns.
- Physical layer constants are centralized in dedicated `*Constants` classes (`PssConstants`, `CpConstants`, `GscnConstants`, `BandConstants`).
- All frequency calculations follow 3GPP TS 38.211 and 38.101-1. References to specific tables/clauses are noted in comments.

## Reference Documents

3GPP specification documents are stored in `doc/` as `.docx` files (TS 38.101-1 and TS 38.211). Refer to these for protocol details when implementing new features.
