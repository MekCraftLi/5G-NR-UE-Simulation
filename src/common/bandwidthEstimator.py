"""
信号带宽与 SCS 检测模块

职责：通过 CP 自相关推断 SCS，进而计算 SSB 占用带宽
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BandwidthEstimate:
    bandwidthHz: float
    scsHz: int
    fftSize: int
    cpLength: int
    confidence: float   # 0~1, SCS 检测置信度


# SCS 假设及其对应的 OFDM 参数 (fs=30.72MHz)
# 其他采样率按 fs/fft = SCS 关系推导
_SCS_HYPOTHESES = [
    (15000, 2048, 144, 2192),
    (30000, 1024, 72, 1096),
    (60000, 512, 36, 548),
]

# SSB 固定占用 240 子载波 (20 PRB), TS 38.211 Clause 7.4.3.1
SSB_NUM_SUBCARRIERS = 240


def detectBandwidth(signal: np.ndarray, sampleRate: float) -> BandwidthEstimate:
    """
    通过 CP 自相关检测 SCS，返回 SSB 占用带宽

    方法:
      对每种 SCS 假设，计算 lag=FFT_size 的滑动自相关，
      按符号长度折叠累加后比较峰均比。
      CP 重复特性使得正确 SCS 下相关峰最强。

    返回:
      BandwidthEstimate: SSB 带宽 (=240×SCS)、检测到的 SCS、置信度
    """
    signalC64 = signal.astype(np.complex64)
    corrLen = min(50000, len(signalC64) - 3000)

    bestScore = 0.0
    bestHyp = None

    for scs, fftN, cpLen, symLen in _SCS_HYPOTHESES:
        # 按采样率缩放 FFT 点数
        scale = sampleRate / 30.72e6
        fftN_scaled = int(fftN * scale)
        cpLen_scaled = int(cpLen * scale)
        symLen_scaled = int(symLen * scale)

        if fftN_scaled < 64 or cpLen_scaled < 1:
            continue
        if symLen_scaled > corrLen:
            continue

        # 滑动 CP 自相关
        corr = np.zeros(corrLen, dtype=np.float64)
        for i in range(corrLen):
            a = signalC64[i:i + cpLen_scaled]
            b = signalC64[i + fftN_scaled:i + fftN_scaled + cpLen_scaled]
            corr[i] = np.abs(np.vdot(a, b))

        # 按符号长度折叠累加
        folded = np.zeros(symLen_scaled, dtype=np.float64)
        nFolds = 0
        for i in range(0, corrLen - symLen_scaled, symLen_scaled):
            folded += corr[i:i + symLen_scaled]
            nFolds += 1

        if nFolds == 0:
            continue

        peak = float(np.max(folded))
        mean = float(np.mean(folded))
        score = peak / mean if mean > 0 else 1.0

        if score > bestScore:
            bestScore = score
            bestHyp = (scs, fftN_scaled, cpLen_scaled, symLen_scaled)

    if bestHyp is None:
        return BandwidthEstimate(
            bandwidthHz=0.0, scsHz=0, fftSize=0, cpLength=0, confidence=0.0,
        )

    scs, fftN, cpLen, symLen = bestHyp
    bwHz = SSB_NUM_SUBCARRIERS * scs  # SSB 占用带宽

    # 置信度: score > 1.3 为高置信，1.1~1.3 中等，<1.1 低
    confidence = min(1.0, max(0.0, (bestScore - 1.05) / 0.35))

    return BandwidthEstimate(
        bandwidthHz=float(bwHz),
        scsHz=int(scs),
        fftSize=int(fftN),
        cpLength=int(cpLen),
        confidence=float(confidence),
    )


def inferScs(bwEstimate: BandwidthEstimate):
    """
    返回检测到的 SCS 及备选列表
    """
    detected = bwEstimate.scsHz
    if detected <= 0:
        return []

    candidates = [{
        "scsHz": detected,
        "confidence": bwEstimate.confidence,
    }]
    # 备选: 其他标准 SCS
    for altScs in [15e3, 30e3, 60e3]:
        if int(altScs) != int(detected):
            candidates.append({
                "scsHz": int(altScs),
                "confidence": 0.0,
            })

    candidates.sort(key=lambda c: c["confidence"], reverse=True)
    return candidates
