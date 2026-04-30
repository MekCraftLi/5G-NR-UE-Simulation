import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager


def generatePssSequence(nId2: int) -> np.ndarray:
    if nId2 not in (0, 1, 2):
        raise ValueError(f"Invalid N_ID_2: {nId2}")

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
    fftSize = int(config.FftSize)
    cpManager = CpManager(config)
    cpLength = int(cpManager.normalCpLength)

    templates: dict[int, np.ndarray] = {}
    center = fftSize // 2
    pssStart = center - 63

    for nId2 in (0, 1, 2):
        seq = generatePssSequence(nId2)
        freqDomain = np.zeros(fftSize, dtype=np.complex64)
        freqDomain[pssStart:pssStart + len(seq)] = seq
        timeNoCp = np.fft.ifft(np.fft.ifftshift(freqDomain)).astype(np.complex64)
        withCp = np.concatenate([timeNoCp[-cpLength:], timeNoCp]).astype(np.complex64)
        norm = float(np.linalg.norm(withCp))
        if norm <= 1e-12:
            raise RuntimeError("PSS template norm is zero")
        templates[nId2] = (withCp / norm).astype(np.complex64)
    return templates
