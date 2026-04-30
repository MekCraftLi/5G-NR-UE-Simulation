from dataclasses import dataclass
import re

from common.gscn import GscnRaster


@dataclass(frozen=True)
class GscnRasterEntry:
    band: str
    scsHz: int
    ssbCase: str
    bandwidthClass: str
    gscnList: tuple[int, ...]
    sourceSpec: str


_RAW_GT3_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("n1", "15 kHz", "Case A", "5279 – <1> – 5419"),
    ("n2", "15 kHz", "Case A", "4829 – <1> – 4969"),
    ("n3", "15 kHz", "Case A", "4517 – <1> – 4693"),
    ("n5", "15 kHz", "Case A", "2177 – <1> – 2230"),
    ("n5", "30 kHz", "Case B", "2183 – <1> – 2224"),
    ("n7", "15 kHz", "Case A", "6554 – <1> – 6718"),
    ("n8", "15 kHz", "Case A", "2318 – <1> – 2395"),
    ("n12", "15 kHz", "Case A", "1828 – <1> – 1858"),
    ("n13", "15 kHz", "Case A", "1871 – <1> – 1885"),
    ("n14", "15 kHz", "Case A", "1901 – <1> – 1915"),
    ("n18", "15 kHz", "Case A", "2156 – <1> – 2182"),
    ("n20", "15 kHz", "Case A", "1982 – <1> – 2047"),
    ("n24", "15 kHz", "Case A", "3818 – <1> – 3892"),
    ("n24", "30 kHz", "Case B", "3824 – <1> – 3886"),
    ("n25", "15 kHz", "Case A", "4829 – <1> – 4981"),
    ("n26", "15 kHz", "Case A", "2153 – <1> – 2230"),
    ("n28", "15 kHz", "Case A", "1901 – <1> – 2002"),
    ("n29", "15 kHz", "Case A", "1798 – <1> – 1813"),
    ("n30", "15 kHz", "Case A", "5879 – <1> – 5893"),
    ("n31", "15kHz", "Case A", "1161 – <1> – 1162"),
    ("n34", "15 kHz", "Case A", "NOTE 5"),
    ("n34", "30 kHz", "Case C", "5036 – <1> – 5050"),
    ("n38", "15 kHz", "Case A", "NOTE 2"),
    ("n38", "30 kHz", "Case C", "6437 – <1> – 6538"),
    ("n39", "15 kHz", "Case A", "NOTE 6"),
    ("n39", "30 kHz", "Case C", "4712 – <1> – 4789"),
    ("n40", "30 kHz", "Case C", "5762 – <1> – 5989"),
    ("n41", "15 kHz", "Case A", "6246 – <3> – 6717"),
    ("n41", "30 kHz", "Case C", "6252 – <3> – 6714"),
    ("n463", "30 kHz", "Case C", "8993 – <1> – 9530"),
    ("n48", "30 kHz", "Case C", "7884 – <1> – 7982"),
    ("n50", "30 kHz", "Case C", "3590 – <1> – 3781"),
    ("n51", "15 kHz", "Case A", "3572 – <1> – 3574"),
    ("n53", "15 kHz", "Case A", "6215 – <1> – 6232"),
    ("n53", "30 KHz", "Case C", "6221 – <1> – 6226"),
    ("n54", "15 kHz", "Case A", "4181 – <1> – 4182"),
    ("n65", "15 kHz", "Case A", "5279 – <1> – 5494"),
    ("n66", "15 kHz", "Case A", "5279 – <1> – 5494"),
    ("n66", "30 kHz", "Case B", "5285 – <1> – 5488"),
    ("n67", "15 kHz", "Case A", "1850 – <1> – 1888"),
    ("n68", "15 kHz", "Case A", "1888 – <1> – 1951"),
    ("n70", "15 kHz", "Case A", "4993 – <1> – 5044"),
    ("n71", "15 kHz", "Case A", "1547 – <1> – 1624"),
    ("n72", "15 kHz", "Case A", "1157 – <1> – 1159"),
    ("n74", "15 kHz", "Case A", "3692 – <1> – 3790"),
    ("n75", "15 kHz", "Case A", "3584 – <1> – 3787"),
    ("n76", "15 kHz", "Case A", "3572 – <1> – 3574"),
    ("n77", "30 kHz", "Case C", "7711 – <1> – 8329"),
    ("n78", "30 kHz", "Case C", "7711 – <1> – 8051"),
    ("n79", "30 kHz", "Case C", "8480 – <16> – 88807"),
    ("n85", "15 kHz", "Case A", "1826 – <1> – 1858"),
    ("n87", "15 kHz", "Case A", "1055 – <1> – 1057"),
    ("n88", "15 kHz", "Case A", "1061 – <1> – 1062"),
    ("n90", "15 kHz", "Case A", "6246 – <1> – 671710"),
    ("n90", "30 kHz", "Case C", "6252 – <1> – 6714"),
    ("n91", "15 kHz", "Case A", "3572 – <1> – 3574"),
    ("n92", "15 kHz", "Case A", "3584 – <1> – 3787"),
    ("n93", "15 kHz", "Case A", "3572 – <1> – 3574"),
    ("n94", "15 kHz", "Case A", "3584 – <1> – 3787"),
    ("n964", "30 kHz", "Case C", "9531 – <1> – 10363"),
    ("n100", "15 kHz", "Case A", "2303 – <1> – 2307, 4163812"),
    ("n101", "15 kHz", "Case A", "4754 – <1> – 4768"),
    ("n101", "30 kHz", "Case C", "4760 – <1> – 4764"),
    ("n1029", "30 kHz", "Case C", "9531 – <1> – 9877"),
    ("n104", "30 kHz", "Case C", "9882 – <7> – 10358"),
    ("n105", "15 kHz", "Case A", "1535 – <1> – 1624"),
    ("n109", "15 kHz", "Case A", "3584 – <1> – 3787"),
)

_RAW_3MHZ_TABLE: tuple[tuple[str, str, str, str], ...] = (
    ("n5", "15 kHz", "Case A", "30987 – <1> – 31100"),
    ("n12", "15 kHz", "Case A", "30288 – <1> – 30359"),
    ("n26", "15 kHz", "Case A", "30937 – <1> – 31100"),
    ("n28", "15 kHz", "Case A", "30432 – <1> – 30644"),
    ("n31", "15 kHz", "Case A", "28955 – <1> – 28967"),
    ("n72", "15 kHz", "Case A", "28947 – <1> – 28959"),
    ("n85", "15 kHz", "Case A", "30282 – <1> – 30359"),
    ("n87", "15 kHz", "Case A", "28743 – <1> – 28754"),
    ("n88", "15 kHz", "Case A", "28752 – <1> – 28764"),
    ("n100", "15 kHz", "Case A", "31240 – <1> – 31242, 31244 – <1> – 31253, 416372"),
    ("n106", "15 kHz", "Case A", "31317 – <1> – 31329"),
    ("n110", "15 kHz", "Case A", "33802 – <1> – 33804"),
)


_NOTE_EXPANSION_GT3 = {
    "NOTE 2": [6432, 6443, 6457, 6468, 6479, 6493, 6507, 6518, 6532, 6543],
    "NOTE 5": [5032, 5043, 5054],
    "NOTE 6": [4707, 4715, 4718, 4729, 4732, 4743, 4747, 4754, 4761, 4768, 4772, 4782, 4786, 4793],
}

_BAND_NORMALIZATION = {
    "n463": "n46",
    "n964": "n96",
    "n1029": "n102",
}


# TS 38.101-1 Table 5.3.2-1 (FR1 common NRB to channel-BW mapping)
_CHANNEL_BW_BY_NRB = {
    15000: {15: 3, 25: 5, 52: 10, 79: 15, 106: 20, 133: 25, 160: 30, 188: 35, 216: 40, 242: 45, 270: 50},
    30000: {11: 5, 24: 10, 38: 15, 51: 20, 65: 25, 78: 30, 92: 35, 106: 40, 119: 45, 133: 50, 162: 60, 189: 70, 217: 80, 245: 90, 273: 100},
    60000: {11: 10, 18: 15, 24: 20, 31: 25, 38: 30, 44: 35, 51: 40, 58: 45, 65: 50, 79: 60, 93: 70, 107: 80, 121: 90, 135: 100},
}


def _parseScsHz(text: str) -> int:
    num = int(re.findall(r"\d+", text)[0])
    return num * 1000


def _sanitizeGscnNumber(value: int) -> int:
    # Remove trailing footnote digits from malformed markdown exports (e.g., 88807 -> 8880).
    while value > 50000:
        value //= 10
    return value


def _parseGscnSpec(spec: str, noteExpansion: dict[str, list[int]]) -> list[int]:
    spec = spec.strip()
    if spec in noteExpansion:
        return list(noteExpansion[spec])

    values: list[int] = []
    for part in [item.strip() for item in spec.split(",") if item.strip()]:
        nums = [_sanitizeGscnNumber(int(x)) for x in re.findall(r"\d+", part)]
        if len(nums) >= 3 and "<" in part:
            first, step, last = nums[0], nums[1], nums[2]
            values.extend(list(range(first, last + 1, step)))
        elif len(nums) >= 1:
            values.append(nums[0])
    return values


def _buildEntries(rawRows: tuple[tuple[str, str, str, str], ...], bandwidthClass: str) -> tuple[GscnRasterEntry, ...]:
    entries: list[GscnRasterEntry] = []
    for band, scsText, ssbCase, gscnSpec in rawRows:
        normBand = _BAND_NORMALIZATION.get(band, band)
        scsHz = _parseScsHz(scsText)
        gscnList = _parseGscnSpec(gscnSpec, _NOTE_EXPANSION_GT3 if bandwidthClass == "gt3mhz" else {})
        entries.append(
            GscnRasterEntry(
                band=normBand,
                scsHz=scsHz,
                ssbCase=ssbCase,
                bandwidthClass=bandwidthClass,
                gscnList=tuple(sorted(set(gscnList))),
                sourceSpec=gscnSpec,
            )
        )
    return tuple(entries)


GSCN_RASTER_GT3MHZ: tuple[GscnRasterEntry, ...] = _buildEntries(_RAW_GT3_TABLE, "gt3mhz")
GSCN_RASTER_3MHZ: tuple[GscnRasterEntry, ...] = _buildEntries(_RAW_3MHZ_TABLE, "bw3mhz")
ALL_GSCN_RASTER_ENTRIES: tuple[GscnRasterEntry, ...] = GSCN_RASTER_GT3MHZ + GSCN_RASTER_3MHZ


def calculateOccupiedBandwidthHz(prbCount: int, scsHz: int) -> float:
    return float(prbCount * 12 * scsHz)


def inferChannelBandwidthMHz(prbCount: int, scsHz: int) -> float | None:
    mapping = _CHANNEL_BW_BY_NRB.get(scsHz, {})
    return mapping.get(prbCount)


def selectBandwidthClass(prbCount: int, scsHz: int) -> str:
    channelBwMHz = inferChannelBandwidthMHz(prbCount, scsHz)
    if channelBwMHz is not None:
        return "bw3mhz" if abs(channelBwMHz - 3.0) < 1e-9 else "gt3mhz"

    occupiedBwMHz = calculateOccupiedBandwidthHz(prbCount, scsHz) / 1e6
    return "bw3mhz" if occupiedBwMHz <= 3.0 else "gt3mhz"


def getRasterEntries(prbCount: int, scsHz: int) -> tuple[list[GscnRasterEntry], dict]:
    bwClass = selectBandwidthClass(prbCount, scsHz)
    table = GSCN_RASTER_3MHZ if bwClass == "bw3mhz" else GSCN_RASTER_GT3MHZ
    entries = [item for item in table if item.scsHz == scsHz]
    if not entries:
        raise ValueError(f"No GSCN raster entries for SCS={scsHz} Hz in bandwidth class {bwClass}")

    meta = {
        "bandwidthClass": bwClass,
        "occupiedBandwidthHz": calculateOccupiedBandwidthHz(prbCount, scsHz),
        "occupiedBandwidthMHz": calculateOccupiedBandwidthHz(prbCount, scsHz) / 1e6,
        "inferredChannelBandwidthMHz": inferChannelBandwidthMHz(prbCount, scsHz),
        "caseSet": sorted(set(item.ssbCase for item in entries)),
    }
    return entries, meta


def buildGscnMatrix(prbCount: int, scsHz: int) -> tuple[list[tuple[int, float]], list[GscnRasterEntry], dict]:
    entries, meta = getRasterEntries(prbCount, scsHz)
    gscnSet = set()
    for entry in entries:
        gscnSet.update(entry.gscnList)

    gscnList = sorted(gscnSet)
    validGscnList = [(gscn, GscnRaster.getAbsoluteFrequency(gscn)) for gscn in gscnList]
    return validGscnList, entries, meta


def findCasesByGscn(gscn: int, entries: list[GscnRasterEntry]) -> str:
    matches = []
    for entry in entries:
        if gscn in entry.gscnList:
            matches.append(f"{entry.band}/{entry.ssbCase}")
    return ", ".join(sorted(set(matches))) if matches else "Unknown"
