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

All source code is in `src/`, organized by signal processing stage:

```
src/
├── main.py                     # 流水线总入口（仅编排，不含算法逻辑）
├── common/                     # 跨阶段共用模块
│   ├── config.py               # SsbConfig / OfdmConstants / BandConstants
│   ├── cpManager.py            # CP 长度计算 (CpManager)
│   ├── gscn.py                 # GSCN → 绝对频率转换
│   └── ofdm.py                 # OFDM 符号提取与 FFT 解调
├── pss/                        # 阶段 1: PSS 盲搜
│   ├── pssDetector.py          # PssConstants + PssDetector
│   └── pssVisualizer.py        # PssVisualizer
├── sss/                        # 阶段 2: SSS 检测（接口预留）
│   └── sssDetector.py          # SssConstants + SssDetector
└── pbch/                       # 阶段 3: PBCH 解码（接口预留）
    └── pbchDecoder.py          # PbchDecoder
```

Pipeline follows the 5G NR cell search procedure:
1. **PSS 盲搜** → N_ID_2, timingOffset, freqOffset
2. **SSS 检测** (预留) → N_ID_1, N_ID_cell
3. **PBCH 解码** (预留) → MIB, system frame number

## Code Style (全局最高优先级)

### 命名
- 变量和函数: **camelCase** (小驼峰) — 如 `timingOffset`, `detectPss`
- 常量和类名: **PascalCase** (大驼峰) — 如 `PssDetector`, `BandConstants`
- 私有方法: `_camelCase` 前缀 — 如 `_analyzePeaks`
- 模块级常量: `UPPER_SNAKE_CASE` — 如 `SAMPLE_RATE`
- dataclass 字段: `PascalCase` — 如 `FftSize`, `SampleRate`

### 注释
- 所有注释和文档使用中文
- docstring 统一使用三引号 `"""`
- 注释对齐: 同一代码块内的行内注释保持统一缩进
- 方法级 docstring 包含: 功能说明、参数说明、返回值说明
- 3GPP 规范引用标注具体条款号 (如 TS 38.211 Clause 7.4.2.2.1)

### Import 排序 (PEP 8 三段式)
```python
import os                    # 1. 标准库
import logging

import numpy as np           # 2. 第三方库
from scipy import signal

from common.config import ...  # 3. 本地模块
from pss.pssDetector import ...
```
各组之间空一行分隔，组内按字母序排列。

### 工程规范
- 所有模块使用 `logger = logging.getLogger(__name__)` 创建 logger，不直接调用 `logging.xxx()`
- 删除所有未使用的 import 和变量
- 消除重复推导逻辑
- 类名准确反映职责 (如 `PssDetector` 而非 `CellSearcher`)

## Key Conventions

- Physical layer constants are centralized in dedicated `*Constants` classes (`PssConstants`, `SssConstants`, `GscnConstants`, `BandConstants`, `OfdmConstants`).
- All frequency calculations follow 3GPP TS 38.211 and 38.101-1. References to specific tables/clauses are noted in comments.

## Reference Documents

3GPP specification documents are stored in `doc/` as `.docx` files (TS 38.101-1 and TS 38.211). Refer to these for protocol details when implementing new features.
