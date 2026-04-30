import logging

import numpy as np

from common.config import SsbConfig
from pss.pssTemplateFactory import buildPssTimeDomainTemplates

logger = logging.getLogger(__name__)


class PssFreqOffsetEstimator:
    def __init__(self, config: SsbConfig):
        self.fftSize = int(config.FftSize)
        self.cpLength = int(config.NormalCpLength)
        self.sampleRate = float(config.SampleRate)
        templatesWithCp = buildPssTimeDomainTemplates(config)
        self._templatesUseful: dict[int, np.ndarray] = {}
        for nId2, template in templatesWithCp.items():
            useful = template[self.cpLength:self.cpLength + self.fftSize]
            if len(useful) != self.fftSize:
                raise ValueError(
                    f"Invalid useful template length for N_ID_2={nId2}: {len(useful)}"
                )
            self._templatesUseful[int(nId2)] = useful.astype(np.complex64)

    def estimate(
        self,
        rxSignal: np.ndarray,
        nId2: int,
        timingOffset: int,
        baseFreqHz: float,
    ) -> dict | None:
        if nId2 not in self._templatesUseful:
            raise ValueError(f"Invalid nId2={nId2}")

        # Estimate CFO only on the useful PSS symbol period (exclude CP).
        template = self._templatesUseful[int(nId2)]
        templateLength = int(len(template))
        start = int(timingOffset) + self.cpLength
        end = start + templateLength
        if start < 0 or end > len(rxSignal):
            logger.warning(
                "Skip PSS-assisted frequency-offset LS fit: invalid timing window "
                f"(start={start}, end={end}, signalLen={len(rxSignal)})"
            )
            return None

        localIndex = np.arange(templateLength, dtype=np.float64)
        globalIndex = np.arange(start, end, dtype=np.float64)

        phase = np.exp(-1j * 2.0 * np.pi * float(baseFreqHz) * globalIndex / self.sampleRate).astype(np.complex64)
        compensatedWindow = (np.asarray(rxSignal[start:end], dtype=np.complex64) * phase).astype(np.complex64)
        product = compensatedWindow * np.conjugate(template)

        phaseSeries = np.unwrap(np.angle(product)).astype(np.float64)
        weights = np.abs(product).astype(np.float64)
        if len(weights) < 2:
            return None

        weightThreshold = float(np.quantile(weights, 0.2))
        validMask = weights >= weightThreshold
        if int(np.sum(validMask)) < 2:
            validMask = np.ones_like(weights, dtype=bool)

        x = localIndex[validMask]
        y = phaseSeries[validMask]
        w = weights[validMask]
        sqrtW = np.sqrt(np.maximum(w, 1e-12))

        design = np.column_stack((x, np.ones_like(x)))
        weightedDesign = design * sqrtW[:, np.newaxis]
        weightedY = y * sqrtW
        coeff, _, _, _ = np.linalg.lstsq(weightedDesign, weightedY, rcond=None)
        slope = float(coeff[0])
        intercept = float(coeff[1])

        fittedPhase = slope * localIndex + intercept
        residual = phaseSeries - fittedPhase

        yMean = float(np.average(y, weights=w))
        ssRes = float(np.sum(w * (y - (slope * x + intercept)) ** 2))
        ssTot = float(np.sum(w * (y - yMean) ** 2))
        rSquared = float(1.0 - ssRes / (ssTot + 1e-12))

        residualFreqHz = float(slope * self.sampleRate / (2.0 * np.pi))
        refinedFreqHz = float(baseFreqHz + residualFreqHz)

        return {
            "method": "pss_phase_lsq",
            "usefulOnly": True,
            "baseFreqHz": float(baseFreqHz),
            "residualFreqHz": residualFreqHz,
            "refinedFreqHz": refinedFreqHz,
            "slopeRadPerSample": slope,
            "interceptRad": intercept,
            "rSquared": rSquared,
            "sampleStart": start,
            "sampleCount": templateLength,
            "validSampleCount": int(np.sum(validMask)),
            "sampleIndex": localIndex.astype(np.int32),
            "phaseSamplesRad": phaseSeries.astype(np.float32),
            "fittedPhaseRad": fittedPhase.astype(np.float32),
            "residualPhaseRad": residual.astype(np.float32),
        }
