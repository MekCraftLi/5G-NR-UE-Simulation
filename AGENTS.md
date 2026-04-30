# Repository Guidelines

## Project Structure & Module Organization
Core code lives in `src/` and is organized by NR processing stage:
- `src/common/`: shared config and OFDM utilities (`config.py`, `cpManager.py`, `gscn.py`, `ofdm.py`)
- `src/pss/`: implemented PSS blind-search and plotting
- `src/sss/`, `src/pbch/`: reserved interfaces for next pipeline stages
- `src/main.py`: primary receiver flow entry point
- `src/run_tx_reference_flow.py`: smoke flow for TX reference waveform

Data and docs:
- `data/`: local signal inputs (for example `txs0.npy`, `.mat` references)
- `output/`: generated JSON/PNG artifacts
- `doc/`: 3GPP references and project notes
- `convert_docx_to_md.py`: DOCX to Markdown utility for `doc/`

## Build, Test, and Development Commands
This repository is script-driven (no build system file yet).
- `pip install numpy scipy matplotlib python-docx`: install runtime and doc conversion deps
- `cd src && python main.py`: run the full receiver pipeline (expects `../data/txs0.npy`)
- `python src/run_tx_reference_flow.py`: run TX-reference smoke validation
- `python convert_docx_to_md.py`: convert `doc/*.docx` into `doc/markdown/`

## Coding Style & Naming Conventions
Follow existing Python conventions in `src/`:
- 4-space indentation, PEP 8 import grouping (stdlib, third-party, local)
- `camelCase` for variables/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for module constants
- keep stage constants in dedicated `*Constants` classes
- use module logger pattern: `logger = logging.getLogger(__name__)`
- keep protocol references explicit in comments (TS 38.211 / TS 38.101-1 clause/table)

## Testing Guidelines
No dedicated `tests/` suite is committed yet. Until one is added:
- use `python src/run_tx_reference_flow.py` as the required smoke check
- verify regenerated artifacts under `output/` (JSON metrics + heatmap PNG)
- for algorithm changes, include before/after key values (`nId2`, `timingOffset`, peak) in PR notes

## Commit & Pull Request Guidelines
Recent history uses short, action-first Chinese subjects (for example: `添加 ...`, `实现 ...`, `重构 ...`). Keep commits focused and scoped to one change.

PRs should include:
- what stage/module changed and why
- linked issue/spec clause when applicable
- how you validated locally (exact command)
- updated output screenshots/files when visualization behavior changes
