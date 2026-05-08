import logging

import numpy as np

from common.config import SsbConfig
from common.cpManager import CpManager

logger = logging.getLogger(__name__)


class OfdmDemodulator:
    """OFDM FFT extraction helper for SS/PBCH block processing."""

    def __init__(self, config: SsbConfig):
        self.config = config
        self.cpManager = CpManager(config)
        self.fftSize = int(config.FftSize)

    def extractSymbol(self, rxSignal: np.ndarray, symbolStart: int, cpLength: int | None = None) -> np.ndarray:
        """Extract one OFDM symbol as fftshift(FFT(useful-part))."""
        cpLength = int(self.cpManager.normalCpLength if cpLength is None else cpLength)
        symbolStart = int(symbolStart)
        end = symbolStart + cpLength + self.fftSize
        if symbolStart < 0 or end > len(rxSignal):
            logger.warning(
                "OFDM symbol extraction out of range: start=%s, cp=%s, end=%s, signalLen=%s",
                symbolStart,
                cpLength,
                end,
                len(rxSignal),
            )
            return np.zeros(self.fftSize, dtype=np.complex64)

        useful = np.asarray(rxSignal[symbolStart + cpLength:symbolStart + cpLength + self.fftSize], dtype=np.complex64)
        return np.fft.fftshift(np.fft.fft(useful)).astype(np.complex64)

    def extractSsbSymbols(self, rxSignal: np.ndarray, ssbStart: int, cpLengths: list[int] | tuple[int, ...] | None = None) -> list[np.ndarray]:
        """Extract the four SS/PBCH block symbols."""
        symbols: list[np.ndarray] = []
        current = int(ssbStart)
        for symbolIndex in range(int(self.config.SsbSymbolCount)):
            cpLength = int(cpLengths[symbolIndex]) if cpLengths is not None else int(self.cpManager.getCpLength(symbolIndex))
            symbols.append(self.extractSymbol(rxSignal, current, cpLength=cpLength))
            current += self.fftSize + cpLength
        return symbols

    def getSssSymbolOffset(self, ssbStart: int, cpLengths: list[int] | tuple[int, ...] | None = None) -> int:
        """Return the absolute CP-start sample of SSS symbol inside an SS/PBCH block."""
        offset = 0
        for symbolIndex in range(2):
            cpLength = int(cpLengths[symbolIndex]) if cpLengths is not None else int(self.cpManager.getCpLength(symbolIndex))
            offset += self.fftSize + cpLength
        return int(ssbStart) + offset
