"""PSS 序列与 OFDM 时域模板构造。

规范锚点：
- TS 38.211 Clause 7.4.2.2.1：PSS 是长度为 127 的 m 序列：
  d_PSS(n) = 1 - 2*x[(n + 43*N_ID_2) mod 127], N_ID_2 in {0,1,2}.
- TS 38.211 Clause 7.4.3.1.1 / Table 7.4.3.1-1：PSS 映射在
  SS/PBCH block 符号 l=0 的中间 127 个子载波，也就是 k=56..182。

数据交接：
- `generatePssSequence(nId2)` 返回频域 BPSK PSS 序列。
- `buildPssTimeDomainTemplates(config)` 把该序列映射到配置好的 FFT 网格，
  执行 IFFT、补 CP、归一化，然后把模板交给 PSS 相关搜索器。
"""

import numpy as np

from common.config import SsbConfig


def generatePssSequence(nId2: int) -> np.ndarray:
    """按给定 N_ID_2 生成一条符合规范的 PSS 序列。

    输入 `nId2` 来自检测器假设。输出是即将插入 SS/PBCH 资源网格的
    127 子载波复数序列。
    """
    if nId2 not in (0, 1, 2):
        raise ValueError(f"N_ID_2 非法：{nId2}")

    # TS 38.211 给出的初始状态是 [x(6),...,x(0)] = [1,1,1,0,1,1,0]。
    # 下面数组按 x(0)..x(6) 存储，因此看起来是反向顺序。
    x = np.zeros(127 + 7, dtype=np.int8)
    x[:7] = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.int8)
    for n in range(127):
        x[n + 7] = (x[n + 4] + x[n]) & 1

    m = 43 * nId2
    seq = np.empty(127, dtype=np.float32)
    for n in range(127):
        seq[n] = 1.0 - 2.0 * x[(n + m) % 127]
    return seq.astype(np.complex64)


def buildPssTimeDomainTemplates(config: SsbConfig) -> dict[int, np.ndarray]:
    """构造带 CP 的归一化 PSS 时域相关模板。

    数据流：
    `config.FftSize/config.NormalCpLength/config.SsbSubcarrierOffset`
    -> 本地 OFDM 网格 -> IFFT -> `[CP | useful symbol]` 模板。

    PSS 检测器接收的是时域 IQ 采样，不是裸 RE 值，所以相关模板必须包含
    和真实 SS/PBCH 符号 0 一致的 IFFT 与 CP 几何。
    """
    fftSize = int(config.FftSize)
    # 本接收机中 SS/PBCH block 内的 PSS 按 normal CP 符号处理。
    cpLength = int(config.NormalCpLength)

    templates: dict[int, np.ndarray] = {}
    center = fftSize // 2
    pssStart = center - 63 + int(getattr(config, "SsbSubcarrierOffset", 0))

    for nId2 in (0, 1, 2):
        seq = generatePssSequence(nId2)
        freqDomain = np.zeros(fftSize, dtype=np.complex64)
        freqDomain[pssStart:pssStart + len(seq)] = seq
        timeNoCp = np.fft.ifft(np.fft.ifftshift(freqDomain)).astype(np.complex64)
        withCp = np.concatenate([timeNoCp[-cpLength:], timeNoCp]).astype(np.complex64)
        norm = float(np.linalg.norm(withCp))
        if norm <= 1e-12:
            raise RuntimeError("PSS 模板范数为零")
        templates[nId2] = (withCp / norm).astype(np.complex64)
    return templates
