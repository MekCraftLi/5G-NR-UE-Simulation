import logging
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import savemat

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PbchBchDecodeAttempt:
    v: int
    prbsPhaseStart: int
    crcOk: bool
    crcRemainder: int
    crcParityErrorCount: int
    payloadBits: np.ndarray
    crcBits: np.ndarray
    infoBits: np.ndarray


class PbchBchDecoder:
    """PBCH/BCH decode pipeline with an official standards path.

    Notes:
    - The default path delegates PBCH/BCH decoding to MATLAB 5G Toolbox:
      nrPBCHDecode -> nrBCHDecode. This matches the official examples and
      avoids treating the old hand-written polar scaffold as normative.
    - The previous Python-only implementation is retained as a diagnostic
      fallback when MATLAB is unavailable.
    """

    PbchRateMatchedLength = 864
    PolarLengthN = 512
    PayloadLengthA = 32
    CrcLength = 24
    PolarInfoLengthK = PayloadLengthA + CrcLength

    # PBCH (N=512, K=56) input-bit positions in reliability order.
    # Reference: commonly used 3GPP-compatible PBCH profile.
    PbchPolarInfoIndicesOrdered = np.asarray(
        [
            441, 469, 247, 367, 253, 375, 444, 470, 483, 415, 485, 473,
            474, 254, 379, 431, 489, 486, 476, 439, 490, 463, 381, 497,
            492, 443, 382, 498, 445, 471, 500, 446, 475, 487, 504, 255,
            477, 491, 478, 383, 493, 499, 502, 494, 501, 447, 505, 506,
            479, 508, 495, 503, 507, 509, 510, 511,
        ],
        dtype=np.int32,
    )

    def __init__(
        self,
        preferMatlab: bool = True,
        matlabExecutable: str | None = None,
        listLength: int = 8,
        lssbCandidates: tuple[int, ...] = (4, 8),
        outputDir: str | Path = "output",
        matlabTimeoutSec: int = 180,
        allowPythonFallback: bool = True,
    ) -> None:
        self.preferMatlab = bool(preferMatlab)
        self.matlabExecutable = matlabExecutable or self._defaultMatlabExecutable()
        self.listLength = int(listLength)
        self.lssbCandidates = tuple(int(x) for x in lssbCandidates if int(x) in (4, 8, 64))
        if not self.lssbCandidates:
            self.lssbCandidates = (4, 8)
        self.outputDir = Path(outputDir)
        self.matlabTimeoutSec = int(matlabTimeoutSec)
        self.allowPythonFallback = bool(allowPythonFallback)

    def decode(
        self,
        pbchEq: np.ndarray,
        nIdCell: int,
        iSsbBar: int,
        noiseVar: float | None = None,
        outputPrefix: str | None = None,
    ) -> dict[str, Any]:
        if self.preferMatlab:
            official = self._decodeMatlabOfficial(
                pbchEq=pbchEq,
                nIdCell=nIdCell,
                iSsbBar=iSsbBar,
                noiseVar=noiseVar,
                outputPrefix=outputPrefix,
            )
            if official.get("matlabRan", False):
                return official
            if not self.allowPythonFallback:
                return official

            fallback = self._decodePythonFallback(
                pbchEq=pbchEq,
                nIdCell=nIdCell,
                iSsbBar=iSsbBar,
                noiseVar=noiseVar,
            )
            fallback["officialDecodeAttempt"] = official
            fallback["method"] = "python_fallback_after_matlab_official_unavailable"
            return fallback

        return self._decodePythonFallback(
            pbchEq=pbchEq,
            nIdCell=nIdCell,
            iSsbBar=iSsbBar,
            noiseVar=noiseVar,
        )

    def _decodeMatlabOfficial(
        self,
        pbchEq: np.ndarray,
        nIdCell: int,
        iSsbBar: int,
        noiseVar: float | None,
        outputPrefix: str | None,
    ) -> dict[str, Any]:
        self.outputDir.mkdir(parents=True, exist_ok=True)
        prefix = self._artifactPrefix(outputPrefix)
        inputMat = self.outputDir / f"{prefix}_bch_official_input.mat"
        outputJson = self.outputDir / f"{prefix}_bch_official_result.json"
        outputMat = self.outputDir / f"{prefix}_bch_official_result.mat"

        sigma2 = float(noiseVar) if noiseVar is not None and float(noiseVar) >= 0.0 else 1e-10
        pbchSymbols = np.asarray(pbchEq, dtype=np.complex128).reshape(-1, 1)
        savemat(
            inputMat,
            {
                "pbchEq": pbchSymbols,
                "nIdCell": np.asarray([[int(nIdCell)]], dtype=np.float64),
                "iSsbBar": np.asarray([[int(iSsbBar)]], dtype=np.float64),
                "noiseVar": np.asarray([[sigma2]], dtype=np.float64),
                "listLength": np.asarray([[self.listLength]], dtype=np.float64),
                "lssbCandidates": np.asarray(self.lssbCandidates, dtype=np.float64),
            },
        )

        repoRoot = Path(__file__).resolve().parents[2]
        matlabScript = repoRoot / "matlab_official_pbch_bch_decode.m"
        if not matlabScript.exists():
            return self._officialFailure(
                "matlab_official",
                f"Missing MATLAB bridge script: {matlabScript}",
                inputMat,
                outputJson,
                outputMat,
            )

        command = (
            f"cd({self._matlabQuote(repoRoot)}); "
            f"matlab_official_pbch_bch_decode("
            f"{self._matlabQuote(inputMat)}, "
            f"{self._matlabQuote(outputJson)}, "
            f"{self._matlabQuote(outputMat)});"
        )
        try:
            proc = subprocess.run(
                [self.matlabExecutable, "-batch", command],
                cwd=str(repoRoot),
                capture_output=True,
                text=True,
                timeout=self.matlabTimeoutSec,
                check=False,
            )
        except Exception as exc:
            return self._officialFailure(
                "matlab_official",
                str(exc),
                inputMat,
                outputJson,
                outputMat,
            )

        if proc.returncode != 0 or not outputJson.exists():
            return self._officialFailure(
                "matlab_official",
                f"MATLAB exited with code {proc.returncode}",
                inputMat,
                outputJson,
                outputMat,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        try:
            import json

            result = json.loads(outputJson.read_text(encoding="utf-8"))
        except Exception as exc:
            return self._officialFailure(
                "matlab_official",
                f"Could not parse MATLAB output JSON: {exc}",
                inputMat,
                outputJson,
                outputMat,
                stdout=proc.stdout,
                stderr=proc.stderr,
            )

        result["matlabRan"] = True
        result["failureStage"] = "" if bool(result.get("crcOk", False)) else result.get("failureStage", "crc")
        bestAttempt = result.get("bestAttempt", {}) if isinstance(result.get("bestAttempt", {}), dict) else {}
        if isinstance(bestAttempt.get("mib", {}), dict):
            result["mib"] = bestAttempt["mib"]
        if "hrf" in bestAttempt:
            result["hrf"] = bestAttempt.get("hrf")
        if "sfnLsb4" in bestAttempt:
            result["sfnLsb4"] = bestAttempt.get("sfnLsb4")
        result["artifactFiles"] = {
            "inputMat": str(inputMat.resolve()),
            "resultJson": str(outputJson.resolve()),
            "resultMat": str(outputMat.resolve()),
        }
        result["matlabStdoutTail"] = self._tailText(proc.stdout)
        result["matlabStderrTail"] = self._tailText(proc.stderr)
        return result

    def _decodePythonFallback(
        self,
        pbchEq: np.ndarray,
        nIdCell: int,
        iSsbBar: int,
        noiseVar: float | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "method": "python_pbch_bch_decode_diagnostic_fallback",
            "nIdCell": int(nIdCell),
            "iSsbBar": int(iSsbBar),
            "crcOk": False,
            "failureStage": "",
            "implementedStages": {
                "llr": True,
                "descramble": True,
                "rateRecovery": True,
                "polar": True,
                "crc": True,
                "mibParse": True,
                "limitations": [
                    "PBCH payload deinterleaving and known-bit handling are simplified",
                    "Polar input bit deinterleaver uses a fixed PBCH profile",
                    "Rate recovery currently uses repetition combining only",
                    "Use MATLAB official path for standards-compliant decoding",
                ],
            },
        }

        try:
            llr = self._qpskSoftLlr(pbchEq, noiseVar=noiseVar)
        except Exception as exc:
            result["failureStage"] = "llr"
            result["error"] = str(exc)
            return result

        # Try likely v hypotheses and two PRBS phase-start models.
        vPrimary = int(iSsbBar) % 8
        vOrder = [vPrimary]
        vMod4 = int(iSsbBar) % 4
        if vMod4 not in vOrder:
            vOrder.append(vMod4)
        for v in range(8):
            if v not in vOrder:
                vOrder.append(v)

        attempts: list[PbchBchDecodeAttempt] = []
        for v in vOrder:
            for phaseStart in (v * self.PbchRateMatchedLength, v * self.PayloadLengthA):
                try:
                    attempt = self._decodeOnce(
                        llr=llr,
                        nIdCell=int(nIdCell),
                        v=int(v),
                        prbsPhaseStart=int(phaseStart),
                    )
                    attempts.append(attempt)
                except Exception:
                    continue

        if not attempts:
            result["failureStage"] = "descramble"
            result["error"] = "No valid PBCH/BCH decode attempts"
            return result

        # Best effort selection: CRC pass first, then smaller CRC syndrome.
        attemptsSorted = sorted(
            attempts,
            key=lambda item: (
                0 if item.crcOk else 1,
                int(item.crcParityErrorCount),
                int(item.crcRemainder),
            ),
        )
        best = attemptsSorted[0]

        payloadBits = best.payloadBits.astype(np.uint8)
        mib = self._parseMib(payloadBits[:24], payloadBits[24:32])

        result.update(
            {
                "crcOk": bool(best.crcOk),
                "failureStage": "" if bool(best.crcOk) else "crc",
                "vUsed": int(best.v),
                "prbsPhaseStart": int(best.prbsPhaseStart),
                "payloadBits": payloadBits.tolist(),
                "mibBits": payloadBits[:24].tolist(),
                "extraTimingBits": payloadBits[24:32].tolist(),
                "mib": mib,
                "sfnMsb6": int(mib["systemFrameNumberMsb6"]),
                "hrf": int(mib["halfFrameBit"]),
                "crcRemainder": int(best.crcRemainder),
                "crcParityErrorCount": int(best.crcParityErrorCount),
                "attempts": [
                    {
                        "v": int(item.v),
                        "prbsPhaseStart": int(item.prbsPhaseStart),
                        "crcOk": bool(item.crcOk),
                        "crcRemainder": int(item.crcRemainder),
                        "crcParityErrorCount": int(item.crcParityErrorCount),
                    }
                    for item in attemptsSorted[:16]
                ],
            }
        )
        return result

    @staticmethod
    def _defaultMatlabExecutable() -> str:
        envValue = os.environ.get("MATLAB_EXECUTABLE")
        if envValue:
            return envValue
        default = Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe")
        if default.exists():
            return str(default)
        return "matlab"

    @staticmethod
    def _matlabQuote(path: str | Path) -> str:
        text = str(path).replace("\\", "/").replace("'", "''")
        return f"'{text}'"

    @staticmethod
    def _tailText(text: str, maxChars: int = 4000) -> str:
        if not text:
            return ""
        return text[-maxChars:]

    @staticmethod
    def _artifactPrefix(outputPrefix: str | None) -> str:
        if outputPrefix:
            cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(outputPrefix))
            return cleaned.strip("_") or f"pbch_{uuid.uuid4().hex[:8]}"
        return f"pbch_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _officialFailure(
        stage: str,
        error: str,
        inputMat: Path,
        outputJson: Path,
        outputMat: Path,
        stdout: str = "",
        stderr: str = "",
    ) -> dict[str, Any]:
        return {
            "method": "matlab_5g_toolbox_nrPBCHDecode_nrBCHDecode",
            "officialChain": True,
            "matlabRan": False,
            "crcOk": False,
            "failureStage": stage,
            "error": error,
            "artifactFiles": {
                "inputMat": str(inputMat.resolve()),
                "resultJson": str(outputJson.resolve()),
                "resultMat": str(outputMat.resolve()),
            },
            "matlabStdoutTail": PbchBchDecoder._tailText(stdout),
            "matlabStderrTail": PbchBchDecoder._tailText(stderr),
        }

    def _decodeOnce(self, llr: np.ndarray, nIdCell: int, v: int, prbsPhaseStart: int) -> PbchBchDecodeAttempt:
        llrDescrambled = self._descramblePbchLlr(llr, nIdCell=nIdCell, v=v, phaseStart=prbsPhaseStart)
        llrPolar = self._rateRecoverPolar(llrDescrambled, self.PolarLengthN)
        uHat = self._polarScDecode(llrPolar)
        infoBits = uHat[self.PbchPolarInfoIndicesOrdered]
        if len(infoBits) != self.PolarInfoLengthK:
            raise ValueError("Polar info-bit extraction length mismatch")

        payloadBits = infoBits[:self.PayloadLengthA]
        crcBits = infoBits[self.PayloadLengthA:self.PayloadLengthA + self.CrcLength]
        crcRemainder = self._crc24cRemainder(infoBits)
        crcOk = bool(crcRemainder == 0)
        crcParityErrorCount = int(bin(int(crcRemainder) & 0xFFFFFF).count("1"))
        return PbchBchDecodeAttempt(
            v=int(v),
            prbsPhaseStart=int(prbsPhaseStart),
            crcOk=crcOk,
            crcRemainder=int(crcRemainder),
            crcParityErrorCount=int(crcParityErrorCount),
            payloadBits=np.asarray(payloadBits, dtype=np.uint8),
            crcBits=np.asarray(crcBits, dtype=np.uint8),
            infoBits=np.asarray(infoBits, dtype=np.uint8),
        )

    def _qpskSoftLlr(self, pbchEq: np.ndarray, noiseVar: float | None = None) -> np.ndarray:
        symbols = np.asarray(pbchEq, dtype=np.complex64).reshape(-1)
        if len(symbols) == 0:
            raise ValueError("Empty PBCH equalized symbols")
        if len(symbols) != self.PbchRateMatchedLength // 2:
            logger.warning(
                "Unexpected PBCH symbol count=%s (expected=%s)",
                len(symbols),
                self.PbchRateMatchedLength // 2,
            )
        sigma2 = float(noiseVar) if noiseVar is not None and float(noiseVar) > 0.0 else 1.0
        scale = float(np.sqrt(2.0) / max(sigma2, 1e-12))
        llrI = scale * np.real(symbols).astype(np.float64)
        llrQ = scale * np.imag(symbols).astype(np.float64)
        llr = np.empty(2 * len(symbols), dtype=np.float64)
        llr[0::2] = llrI
        llr[1::2] = llrQ
        return llr

    def _descramblePbchLlr(self, llr: np.ndarray, nIdCell: int, v: int, phaseStart: int = 0) -> np.ndarray:
        seq = self._pbchPrbs(nIdCell=nIdCell, v=v, length=len(llr), phaseStart=phaseStart)
        mask = (1.0 - 2.0 * seq.astype(np.float64))
        return np.asarray(llr, dtype=np.float64) * mask

    @staticmethod
    def _goldSequence(cInit: int, length: int) -> np.ndarray:
        nc = 1600
        total = nc + int(length) + 31
        x1 = np.zeros(total, dtype=np.uint8)
        x2 = np.zeros(total, dtype=np.uint8)
        x1[0] = 1
        for i in range(31):
            x2[i] = (int(cInit) >> i) & 1
        for n in range(total - 31):
            x1[n + 31] = (x1[n + 3] + x1[n]) & 1
            x2[n + 31] = (x2[n + 3] + x2[n + 2] + x2[n + 1] + x2[n]) & 1
        return ((x1[nc:nc + length] + x2[nc:nc + length]) & 1).astype(np.uint8)

    def _pbchPrbs(self, nIdCell: int, v: int, length: int, phaseStart: int = 0) -> np.ndarray:
        # 38.211 PBCH PRBS uses NCellID plus a v-dependent phase.
        # Here we keep cInit=NCellID and model v via sequence start shift.
        offset = max(0, int(phaseStart))
        seq = self._goldSequence(cInit=int(nIdCell), length=int(length) + offset)
        return seq[offset:offset + int(length)]

    @staticmethod
    def _rateRecoverPolar(llr: np.ndarray, n: int) -> np.ndarray:
        llr = np.asarray(llr, dtype=np.float64).reshape(-1)
        out = np.zeros(int(n), dtype=np.float64)
        for i, value in enumerate(llr):
            out[i % int(n)] += float(value)
        return out

    def _polarScDecode(self, llr: np.ndarray) -> np.ndarray:
        llr = np.asarray(llr, dtype=np.float64).reshape(-1)
        if len(llr) != self.PolarLengthN:
            raise ValueError(f"Polar LLR length must be {self.PolarLengthN}, got {len(llr)}")
        frozenMask = np.ones(self.PolarLengthN, dtype=bool)
        frozenMask[self.PbchPolarInfoIndicesOrdered] = False
        return self._scDecodeRecursive(llr, frozenMask).astype(np.uint8)

    def _scDecodeRecursive(self, llr: np.ndarray, frozenMask: np.ndarray) -> np.ndarray:
        n = int(len(llr))
        if n == 1:
            if bool(frozenMask[0]):
                return np.asarray([0], dtype=np.uint8)
            return np.asarray([0 if float(llr[0]) >= 0.0 else 1], dtype=np.uint8)

        half = n // 2
        a = np.asarray(llr[:half], dtype=np.float64)
        b = np.asarray(llr[half:], dtype=np.float64)
        leftLlr = np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))
        uLeft = self._scDecodeRecursive(leftLlr, frozenMask[:half]).astype(np.uint8)

        rightLlr = b + (1.0 - 2.0 * uLeft.astype(np.float64)) * a
        uRight = self._scDecodeRecursive(rightLlr, frozenMask[half:]).astype(np.uint8)

        u = np.empty(n, dtype=np.uint8)
        u[:half] = np.bitwise_xor(uLeft, uRight)
        u[half:] = uRight
        return u

    @staticmethod
    def _crc24cRemainder(bits: np.ndarray) -> int:
        poly = 0x1864CFB
        reg = 0
        for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
            reg = ((reg << 1) | int(bit & 1)) & 0x1FFFFFF
            if reg & 0x1000000:
                reg ^= poly
        return int(reg & 0xFFFFFF)

    @staticmethod
    def _bitsToInt(bits: np.ndarray) -> int:
        out = 0
        for bit in np.asarray(bits, dtype=np.uint8).reshape(-1):
            out = (out << 1) | int(bit & 1)
        return int(out)

    def _parseMib(self, mibBits24: np.ndarray, extraBits8: np.ndarray) -> dict[str, Any]:
        b = np.asarray(mibBits24, dtype=np.uint8).reshape(-1)
        e = np.asarray(extraBits8, dtype=np.uint8).reshape(-1)
        if len(b) < 24:
            raise ValueError("MIB parse requires at least 24 bits")
        if len(e) < 8:
            e = np.pad(e, (0, max(0, 8 - len(e))), mode="constant")

        systemFrameNumberMsb6 = self._bitsToInt(b[1:7])
        subCarrierSpacingCommonBit = int(b[7])
        ssbSubcarrierOffsetLsb4 = self._bitsToInt(b[8:12])
        dmrsTypeABit = int(b[12])
        pdcchConfigSib1 = self._bitsToInt(b[13:21])
        cellBarredBit = int(b[21])
        intraFreqReselectionBit = int(b[22])
        spareBits = b[23:24].astype(np.uint8)

        sfnLsb4 = self._bitsToInt(e[0:4])
        halfFrameBit = int(e[4])
        ssbIndexLsb3 = self._bitsToInt(e[5:8])
        sfn10 = int((systemFrameNumberMsb6 << 4) | sfnLsb4)

        return {
            "systemFrameNumberMsb6": int(systemFrameNumberMsb6),
            "systemFrameNumberLsb4": int(sfnLsb4),
            "systemFrameNumber10": int(sfn10),
            "subCarrierSpacingCommon": "scs15or60" if subCarrierSpacingCommonBit == 0 else "scs30or120",
            "ssbSubcarrierOffsetLsb4": int(ssbSubcarrierOffsetLsb4),
            "dmrsTypeAPosition": "pos2" if dmrsTypeABit == 0 else "pos3",
            "pdcchConfigSIB1": int(pdcchConfigSib1),
            "cellBarred": "barred" if cellBarredBit == 0 else "notBarred",
            "intraFreqReselection": "allowed" if intraFreqReselectionBit == 0 else "notAllowed",
            "spareBits": spareBits.astype(int).tolist(),
            "halfFrameBit": int(halfFrameBit),
            "ssbIndexLsb3": int(ssbIndexLsb3),
            "note": "MIB parse uses simplified payload ordering assumptions",
        }
