# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 5G NR (New Radio) physical layer signal processing project implementing cell search and synchronization procedures per 3GPP TS 38.211 / 38.101-1 specifications. The project processes received baseband I/Q signals to detect Primary Synchronization Signals (PSS) and perform initial cell search.

**Python Environment**: conda environment `nr_phy` — activate with `conda activate nr_phy` before running.

## Running

```bash
conda activate nr_phy
cd src
python main.py
```

### CLI Arguments

```bash
python main.py --config ../config/main.example.json       # 使用配置文件
python main.py --sample-rate 30.72e6 --scs 30000          # 命令行参数
python main.py --input-path ../data/rxSignal.npy          # 指定输入文件
python main.py --pss-only                                 # 仅运行 PSS 阶段
python main.py --disable-sss --disable-pbch               # 禁用后续阶段
python main.py --use-gpu                                  # 启用 GPU 加速 (PyTorch)
python main.py --no-gpu                                   # 强制 CPU 模式
```

主要参数:
- `--config`: JSON/YAML 配置文件路径 (参考 `config/main.example.json`)
- `--sample-rate`: 采样率 (Hz)，默认 30.72 MHz
- `--scs`: 子载波间隔 (Hz)，默认 30000 (30 kHz)
- `--pss-workers`: PSS 检测并行工作进程数
- `--pss-score-mode`: `raw` 或 `ncc` (归一化互相关)
- `--pss-adaptive-iterations`: 自适应精细搜索迭代次数
- `--pss-coarse-step/--medium-step/--fine-step`: 频率搜索步进 (Hz)
- `--sss-freq-base-mode`: SSS 频偏基准 (`pss`/`scs`/`zero`)
- `--sss-offset-candidates`: SSB 子载波偏移候选列表 (逗号分隔)
- `--pbch-residual-freq-search`: PBCH 残余 CFO 搜索范围 (Hz)
- `--progress-mode`: 进度显示 (`auto`/`rich`/`log`)

Input signal data is loaded from `data/rxSignal.npy` (NumPy binary format). The path in `main.py` is relative (`../data/rxSignal.npy`), so run from `src/`.

Dependencies: `numpy`, `scipy`, `matplotlib`. Optional: `rich` (进度条), `pyyaml` (YAML 配置), `torch` (GPU 加速).

### Validation

No dedicated test suite. Use the TX reference flow as a smoke check:
```bash
python src/run_tx_reference_flow.py
```
Verify regenerated artifacts under `output/` (JSON metrics + PNG plots).

## Architecture

All source code is in `src/`, organized by signal processing stage:

```
src/
├── main.py                     # 流水线总入口（仅编排，不含算法逻辑）
├── run_tx_reference_flow.py    # TX 参考波形验证流程
├── common/                     # 跨阶段共用模块
│   ├── config.py               # SsbConfig / OfdmConstants / BandConstants
│   ├── cpManager.py            # CP 长度计算 (CpManager)
│   ├── gscn.py                 # GSCN → 绝对频率转换
│   ├── bandwidthEstimator.py   # SCS/带宽检测 (CP 自相关)
│   └── ofdm.py                 # OFDM 符号提取与 FFT 解调
├── pss/                        # 阶段 1: PSS 盲搜
│   ├── pssBasebandSearcher.py  # 自适应基带频率搜索 (主入口，支持 GPU)
│   ├── pssDetector.py          # PSS 相关检测
│   ├── pssTemplateFactory.py   # PSS 序列生成
│   ├── pssFreqOffsetEstimator.py  # PSS LS 精细频偏估计
│   ├── cpFreqOffsetEstimator.py   # CP 相位频偏估计
│   └── pssVisualizer.py        # PssVisualizer
├── sss/                        # 阶段 2: SSS 检测
│   ├── sssDetector.py          # SssConstants + SssDetector
│   └── sssVisualizer.py        # SssVisualizer
└── pbch/                       # 阶段 3: PBCH 解码
    ├── pbchDecoder.py          # PbchDecoder (DM-RS 信道估计, QPSK 解调, EVM)
    └── pbchBchDecoder.py       # BCH polar 解码 + MIB 解析
```

Pipeline follows the 5G NR cell search procedure:
1. **带宽检测** (detectBandwidth) → SCS 推断、FFT/CP 参数验证
2. **PSS 盲搜** → N_ID_2, timingOffset, freqOffset (自适应多级频率搜索，可选 GPU 加速)
3. **PSS 后处理** → 半符号 FFO 估计、LS 相位拟合、CP 相位频偏估计
4. **SSS 检测** → N_ID_1, N_ID_cell (滑动相关 + 频域验证)
5. **PBCH 解码** → DM-RS 信道估计、QPSK 解调、EVM 评估
6. **BCH 解码** → Polar 译码、CRC 校验、MIB 解析

### 关键数据流

```
rxSignal (np.ndarray) ──► PssBasebandSearcher.searchAdaptive()
                              │
                              ├── freqOffsetParabolic (载波频偏)
                              ├── nId2 (0/1/2)
                              └── timingOffset (PSS 定时)
                                    │
                                    ▼
                         PSS 后处理: 半符号 FFO + LS 拟合 + CP 相位
                              │
                              ├── totalCfoHz (总频偏估计)
                              └── refinedFreqHz (精细频偏)
                                    │
                                    ▼
                         SssDetector.detectSss()
                              │
                              ├── nId1 (0~335)
                              ├── nIdCell = 3*nId1 + nId2
                              └── verifiedFreqCompHz (残余频偏)
                                    │
                                    ▼
                         PbchDecoder.decodePbch()
                              │
                              ├── pbchEq (均衡后符号)
                              ├── hardBits (硬判决比特)
                              └── evmPercent (EVM 指标)
                                    │
                                    ▼
                         PbchBchDecoder.decode()
                              │
                              ├── crcOk (CRC 校验结果)
                              └── mib (MIB 解析字段)
```

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
- `SsbConfig` dataclass holds runtime OFDM parameters (FFT size, CP length, sample rate, SCS).
- `CpManager` computes CP lengths per symbol (normal vs. long CP for slot boundary alignment).
- Output artifacts (plots, JSON results) go to `output/` directory.

## Reference Documents

3GPP specification documents are stored in `doc/` as `.docx` files (TS 38.101-1 and TS 38.211). Refer to these for protocol details when implementing new features.
