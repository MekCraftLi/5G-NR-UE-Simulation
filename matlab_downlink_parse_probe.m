function matlab_downlink_parse_probe(outputPrefix, inputMatPath, inputVarName)
%MATLAB_DOWNLINK_PARSE_PROBE Continue RX parsing after verified PBCH/MIB.
%
% This probe intentionally starts from the narrow PBCH/BCH parameters that
% already pass CRC, then follows the MATLAB 5G Toolbox / HDL example style:
%   CORESET0 tables -> PDCCH SI-RNTI blind decode -> DCI 1_0 parse
%   -> SIB1 PDSCH/DL-SCH decode.

if nargin < 1 || isempty(outputPrefix)
    outputPrefix = 'downlink_parse_probe';
end
if nargin < 2 || isempty(inputMatPath)
    inputMatPath = fullfile(pwd, 'data', 'rxSignal_from_npy.mat');
end
if nargin < 3 || isempty(inputVarName)
    inputVarName = '';
end

repoRoot = pwd;
exampleRoot = fullfile(matlabroot, 'toolbox', 'whdl', 'whdlexamples', 'whdlexamples');
if exist(exampleRoot, 'dir') == 7
    addpath(exampleRoot);
end

outDir = fullfile(repoRoot, 'output');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

rxMatPath = char(inputMatPath);
if exist(rxMatPath, 'file') ~= 2
    error('Missing %s. Generate it from data/rxSignal.npy first.', rxMatPath);
end
rxIn = load(rxMatPath);
if isempty(inputVarName)
    if isfield(rxIn, 'rxSignal')
        inputVarName = 'rxSignal';
    elseif isfield(rxIn, 'txSignal')
        inputVarName = 'txSignal';
    elseif isfield(rxIn, 'txs0')
        inputVarName = 'txs0';
    else
        fields = fieldnames(rxIn);
        fields = fields(~startsWith(fields, '__'));
        if isempty(fields)
            error('No signal variable found in %s', rxMatPath);
        end
        inputVarName = fields{1};
    end
end
rx = rxIn.(char(inputVarName))(:);

% Known-good PBCH/BCH narrow candidate from output/pbch_narrow_official...
p = struct();
p.sampleRate = 122.88e6;
p.scsSSB = 30;
p.scsCommon = 30;
p.nfft = 4096;
p.nCellID = 0;
p.ssbIndex = 0;
p.ssbStart = 6;
p.freqCompHz = 30288.0;
p.ssbSubcarrierOffset = -7;
p.lssb = 4;
p.hrf = 0;
p.nFrame = 400;
p.kSSB = 13;
p.dmrsTypeAPosition = 3;
p.pdcchConfigSIB1 = 170;
p.ssbPattern = 'Case C';
p.standardSsbSampleOffset = standardSsbSampleOffset( ...
    p.ssbPattern, p.ssbIndex, p.sampleRate, p.nfft, p.scsSSB);
p.frameStartEstimate = p.ssbStart - p.standardSsbSampleOffset;

mib = struct();
mib.NFrame = p.nFrame;
mib.SubcarrierSpacingCommon = p.scsCommon;
mib.k_SSB = p.kSSB;
mib.DMRSTypeAPosition = p.dmrsTypeAPosition;
mib.PDCCHConfigSIB1 = p.pdcchConfigSIB1;
mib.CellBarred = false;
mib.IntraFreqReselection = true;

result = struct();
result.method = 'matlab_5g_toolbox_coreset0_pdcch_pdsch_sib1_probe';
result.officialExamplePath = exampleRoot;
result.inputMat = rxMatPath;
result.inputVarName = char(inputVarName);
result.inputSampleCount = double(numel(rx));
result.parameters = p;
result.mib = mib;
result.stage = 'start';
result.success = false;
result.pdcchCrcOk = false;
result.sib1CrcOk = false;
result.error = '';
result.attempts = repmat(emptyAttempt(), 0, 1);

ctrlResourceSet = floor(p.pdcchConfigSIB1 / 16);
searchSpaceZero = mod(p.pdcchConfigSIB1, 16);
scsPair = [p.scsSSB p.scsCommon];
minChanBWCandidates = [5 10 40];

% Frequency hypotheses are deliberately narrow. The first value is derived
% from the SSB FFT-bin placement and the CORESET0 example formula. Extra
% values cover the opposite sign convention and no-Point-A-shift fallback.
baseFreqFromSsbBin = p.freqCompHz + p.ssbSubcarrierOffset * p.scsSSB * 1e3;
directFrameStarts = unique([p.ssbStart + (-8:2:8), 0, 4, 6, 8]);
standardFrameStarts = unique(round(p.frameStartEstimate) + (-16:4:16));

attemptIndex = 0;
bestScore = -Inf;
bestAttempt = emptyAttempt();

for minChanBW = minChanBWCandidates
    try
        [coresetNRB, coresetDuration, coresetOffset, muxPattern] = ...
            nrhdlexamples.coreset0.resources(ctrlResourceSet, scsPair, minChanBW, p.kSSB);
        [coresetNSlot, firstSymbol, inNextFrame] = ...
            nrhdlexamples.coreset0.timingOccasion(searchSpaceZero, p.ssbIndex, scsPair, muxPattern, coresetDuration, p.nFrame);
    catch ex
        attemptIndex = attemptIndex + 1;
        attempt = emptyAttempt();
        attempt.minChanBW = minChanBW;
        attempt.failureStage = 'coreset0_config';
        attempt.error = ex.message;
        result.attempts(attemptIndex) = attempt;
        continue;
    end

    if any(~isfinite([coresetNRB coresetDuration coresetOffset muxPattern coresetNSlot firstSymbol]))
        attemptIndex = attemptIndex + 1;
        attempt = emptyAttempt();
        attempt.minChanBW = minChanBW;
        attempt.failureStage = 'coreset0_config';
        attempt.error = 'CORESET0 table returned NaN for this minimum channel bandwidth';
        result.attempts(attemptIndex) = attempt;
        continue;
    end

    if p.lssb == 64
        scsKSSB = p.scsCommon;
    else
        scsKSSB = 15;
    end
    ssbToSib1Hz = 120 * p.scsSSB * 1e3 ...
        + p.scsCommon * 1e3 * 12 * coresetOffset ...
        + p.kSSB * scsKSSB * 1e3 ...
        - p.scsCommon * 1e3 * 12 * coresetNRB / 2;

    freqCenterA = baseFreqFromSsbBin + ssbToSib1Hz;
    freqCenterB = p.freqCompHz - ssbToSib1Hz;
    freqCenterC = p.freqCompHz;
    freqCenters = unique(round([freqCenterA, freqCenterB, freqCenterC] / 1000) * 1000);
    freqOffsets = -60000:15000:60000;

    gridFirstSlots = [zeros(1, numel(directFrameStarts)) repmat([0 1], 1, numel(standardFrameStarts))];
    frameStarts = [directFrameStarts(:).' reshape(repmat(standardFrameStarts(:).', 2, 1), 1, [])];

    for startIdx = 1:numel(frameStarts)
        timingStart = frameStarts(startIdx);
        gridFirstSlot = gridFirstSlots(startIdx);
        for freqCenter = freqCenters
            for freqDelta = freqOffsets
                freqHz = freqCenter + freqDelta;
                attemptIndex = attemptIndex + 1;
                attempt = tryDownlinkAttempt( ...
                    rx, p, mib, minChanBW, coresetNRB, coresetDuration, ...
                    coresetOffset, muxPattern, coresetNSlot, firstSymbol, ...
                    inNextFrame, timingStart, gridFirstSlot, freqHz);
                result.attempts(attemptIndex) = attempt;

                score = double(attempt.pdcchCrcOk) * 1e9 ...
                    + double(attempt.sib1CrcOk) * 1e12 ...
                    + attempt.bestMeanAbsDciCw ...
                    - double(~isempty(attempt.error)) * 1e3;
                if score > bestScore
                    bestScore = score;
                    bestAttempt = attempt;
                end

                if attempt.sib1CrcOk
                    break;
                end
            end
            if bestAttempt.sib1CrcOk
                break;
            end
        end
        if bestAttempt.sib1CrcOk
            break;
        end
    end
    if bestAttempt.sib1CrcOk
        break;
    end
end

result.bestAttempt = bestAttempt;
result.success = logical(bestAttempt.sib1CrcOk);
result.pdcchCrcOk = logical(bestAttempt.pdcchCrcOk);
result.sib1CrcOk = logical(bestAttempt.sib1CrcOk);
if result.sib1CrcOk
    result.stage = 'sib1_crc_pass';
elseif result.pdcchCrcOk
    result.stage = 'pdcch_crc_pass_sib1_crc_fail';
else
    result.stage = 'pdcch_crc_fail';
end

jsonPath = fullfile(outDir, [char(outputPrefix) '_result.json']);
matPath = fullfile(outDir, [char(outputPrefix) '_result.mat']);
save(matPath, 'result');
writeJson(jsonPath, result);

fprintf('Downlink parse probe done: stage=%s, PDCCH CRC=%d, SIB1 CRC=%d\n', ...
    result.stage, result.pdcchCrcOk, result.sib1CrcOk);
fprintf('  JSON: %s\n', jsonPath);
end

function attempt = tryDownlinkAttempt(rx, p, mib, minChanBW, coresetNRB, coresetDuration, coresetOffset, muxPattern, coresetNSlot, firstSymbol, inNextFrame, timingStart, gridFirstSlot, freqHz)
attempt = emptyAttempt();
attempt.minChanBW = double(minChanBW);
attempt.coresetNRB = double(coresetNRB);
attempt.coresetDuration = double(coresetDuration);
attempt.coresetOffset = double(coresetOffset);
attempt.muxPattern = double(muxPattern);
attempt.coresetNSlot = double(coresetNSlot);
attempt.firstSymbol = double(firstSymbol);
attempt.inNextFrame = logical(inNextFrame);
attempt.timingStart = double(timingStart);
attempt.gridFirstSlot = double(gridFirstSlot);
attempt.freqCompHz = double(freqHz);

try
    grid = demodCoresetGrid(rx, p, coresetNRB, muxPattern, timingStart, gridFirstSlot, freqHz);
    attempt.gridRms = finiteScalar(rms(abs(grid(:))));
    attempt.gridPeak = finiteScalar(max(abs(grid(:))));
catch ex
    attempt.failureStage = 'ofdm_demod';
    attempt.error = ex.message;
    return;
end

try
    [dci, pdcchInfo] = decodeSiPdcch(grid, p, mib, coresetNRB, coresetDuration, muxPattern, coresetNSlot, gridFirstSlot);
    attempt.pdcchCrcOk = logical(pdcchInfo.crcOk);
    attempt.failureStage = pdcchInfo.failureStage;
    attempt.bestMeanAbsDciCw = finiteScalar(pdcchInfo.bestMeanAbsDciCw);
    if isfield(pdcchInfo, 'lastCandidateError')
        attempt.lastCandidateError = pdcchInfo.lastCandidateError;
    end
    attempt.bestAggregationLevel = double(pdcchInfo.bestAggregationLevel);
    attempt.bestCandidateIndex = double(pdcchInfo.bestCandidateIndex);
    attempt.dciSlot = double(pdcchInfo.nSlot);
    attempt.firstOrSecondSlot = double(pdcchInfo.firstOrSecondSlot);
    attempt.dci = dci;
catch ex
    attempt.failureStage = 'pdcch_decode';
    attempt.error = ex.message;
    return;
end

if ~attempt.pdcchCrcOk
    return;
end

try
    [pdsch, k0] = nrhdlexamples.sib1.configuration(dci, coresetNRB, mib.DMRSTypeAPosition, muxPattern);
    pdschSlotIndex = attempt.firstOrSecondSlot + k0;
    symbolsPerSlot = 14;
    slotCols = (1:symbolsPerSlot) + symbolsPerSlot * pdschSlotIndex;
    if max(slotCols) > size(grid, 2)
        error('Decoded DCI schedules PDSCH outside demodulated grid: slotIndex=%d K0=%d', pdschSlotIndex, k0);
    end
    rxSlotGrid = grid(:, slotCols, :);
    [sib1Bits, sib1CRC] = nrhdlexamples.pdschDecoding( ...
        rxSlotGrid, p.nCellID, mib, coresetNRB, dci, attempt.dciSlot, muxPattern);
    attempt.sib1CrcOk = ~logical(sib1CRC);
    attempt.sib1Crc = logical(sib1CRC);
    attempt.sib1BitCount = double(numel(sib1Bits));
    attempt.sib1Bits = double(sib1Bits(:)).';
    attempt.pdschK0 = double(k0);
    attempt.pdschPRBStart = double(min(pdsch.PRBSet));
    attempt.pdschPRBLength = double(numel(pdsch.PRBSet));
    attempt.pdschSymbolStart = double(pdsch.SymbolAllocation(1));
    attempt.pdschSymbolLength = double(pdsch.SymbolAllocation(2));
    attempt.pdschModulation = char(pdsch.Modulation);
    if attempt.sib1CrcOk
        attempt.failureStage = '';
    else
        attempt.failureStage = 'sib1_crc';
    end
catch ex
    attempt.failureStage = 'pdsch_decode';
    attempt.error = ex.message;
end
end

function grid = demodCoresetGrid(rx, p, coresetNRB, muxPattern, timingStart, gridFirstSlot, freqHz)
if muxPattern == 1
    numSlots = 2;
else
    numSlots = 1;
end
slotLength = round(0.001 * p.sampleRate / (p.scsCommon / 15));
needed = numSlots * slotLength;
start = round(timingStart + gridFirstSlot * slotLength);
if start < 0 || start + needed > numel(rx)
    error('Demodulation range out of bounds: start=%d length=%d signalLen=%d', start, needed, numel(rx));
end
idx = (start + 1):(start + needed);
n = (idx(:) - 1);
wave = rx(idx) .* exp(-1i * 2 * pi * double(freqHz) * n / p.sampleRate);
grid = nrOFDMDemodulate( ...
    wave, coresetNRB, p.scsCommon, gridFirstSlot, ...
    'SampleRate', p.sampleRate, ...
    'Nfft', p.nfft, ...
    'CyclicPrefixFraction', 0.5);
if isempty(grid)
    error('nrOFDMDemodulate returned an empty grid');
end
grid = grid ./ sqrt(p.nfft);
end

function [dci, info] = decodeSiPdcch(rx2SlotGrid, p, mib, coresetNRB, coresetDuration, muxPattern, coresetNSlot, gridFirstSlot)
scsCommon = mib.SubcarrierSpacingCommon;
scsPair = [p.scsSSB scsCommon];
pdcch = nrhdlexamples.coreset0.configuration( ...
    p.ssbIndex, mib, scsPair, p.nCellID, coresetNRB, coresetDuration, muxPattern);

c0Carrier = nrCarrierConfig;
c0Carrier.SubcarrierSpacing = scsCommon;
c0Carrier.NStartGrid = pdcch.NStartBWP;
c0Carrier.NSizeGrid = pdcch.NSizeBWP;
c0Carrier.NFrame = mib.NFrame;
c0Carrier.NCellID = p.nCellID;

[dcispec1_0, numDCIBits] = nrhdlexamples.coreset0.dciFieldsSize(coresetNRB);
siRNTI = 65535;
polarListLength = 8;
symbolsPerSlot = 14;
if muxPattern == 1
    monSlots = [coresetNSlot coresetNSlot + 1];
else
    monSlots = coresetNSlot;
end

info = struct();
info.crcOk = false;
info.failureStage = 'pdcch_crc';
info.bestMeanAbsDciCw = 0;
info.bestAggregationLevel = NaN;
info.bestCandidateIndex = NaN;
info.nSlot = gridFirstSlot;
info.firstOrSecondSlot = 0;
info.numDCIBits = numDCIBits;
dci = nrhdlexamples.coreset0.parseDCI(dcispec1_0, zeros(numDCIBits, 1));

for mSlot = 0:(numel(monSlots) - 1)
    candidateSlot = gridFirstSlot + mSlot;
    if ~ismember(candidateSlot, monSlots)
        continue;
    end
    c0Carrier.NSlot = candidateSlot;
    if numel(monSlots) == 2
        pdcch.SearchSpace.SlotPeriodAndOffset(2) = monSlots(mSlot + 1);
    end
    [pdcchInd, pdcchDmrsSym, pdcchDmrsInd] = nrPDCCHSpace(c0Carrier, pdcch);
    slotCols = (1:symbolsPerSlot) + symbolsPerSlot * mSlot;
    rxSlotGrid = rx2SlotGrid(:, slotCols, :);
    scale = max(abs(rxSlotGrid(:)));
    if scale > 0
        rxSlotGrid = rxSlotGrid / scale;
    end

    for aLev = 1:length(pdcch.SearchSpace.NumCandidates)
        numCandidatesAL = pdcch.SearchSpace.NumCandidates(aLev);
        for cIdx = 1:numCandidatesAL
            try
                [hest, nVar] = nrChannelEstimate( ...
                    rxSlotGrid, pdcchDmrsInd{aLev}(:, cIdx), pdcchDmrsSym{aLev}(:, cIdx));
                [pdcchRxSym, pdcchHest] = nrExtractResources(pdcchInd{aLev}(:, cIdx), rxSlotGrid, hest);
                pdcchEqSym = nrEqualizeMMSE(pdcchRxSym, pdcchHest, nVar);
                dcicw = nrPDCCHDecode(pdcchEqSym, p.nCellID, 0, nVar);
                meanAbsDciCw = finiteScalar(mean(abs(dcicw)));
                if meanAbsDciCw > info.bestMeanAbsDciCw
                    info.bestMeanAbsDciCw = meanAbsDciCw;
                    info.bestAggregationLevel = aggregationLevelFromIndex(aLev);
                    info.bestCandidateIndex = cIdx - 1;
                    info.nSlot = c0Carrier.NSlot;
                    info.firstOrSecondSlot = mSlot;
                end
                [tempDCIBits, tempDCICRC] = nrDCIDecode(dcicw, numDCIBits, polarListLength, siRNTI);
                if ~logical(tempDCICRC(1))
                    info.crcOk = true;
                    info.failureStage = '';
                    info.bestMeanAbsDciCw = meanAbsDciCw;
                    info.bestAggregationLevel = aggregationLevelFromIndex(aLev);
                    info.bestCandidateIndex = cIdx - 1;
                    info.nSlot = c0Carrier.NSlot;
                    info.firstOrSecondSlot = mSlot;
                    dci = nrhdlexamples.coreset0.parseDCI(dcispec1_0, double(tempDCIBits));
                    return;
                end
            catch exCand
                if isempty(info.failureStage) || strcmp(info.failureStage, 'pdcch_crc')
                    info.failureStage = 'pdcch_candidate_error';
                    info.lastCandidateError = exCand.message;
                end
                % Keep scanning other candidates. Individual candidate errors
                % are expected when hypotheses are off-grid.
            end
        end
    end
end
end

function value = aggregationLevelFromIndex(idx)
levels = [1 2 4 8 16];
if idx >= 1 && idx <= numel(levels)
    value = levels(idx);
else
    value = NaN;
end
end

function attempt = emptyAttempt()
attempt = struct();
attempt.minChanBW = NaN;
attempt.coresetNRB = NaN;
attempt.coresetDuration = NaN;
attempt.coresetOffset = NaN;
attempt.muxPattern = NaN;
attempt.coresetNSlot = NaN;
attempt.firstSymbol = NaN;
attempt.inNextFrame = false;
attempt.timingStart = NaN;
attempt.gridFirstSlot = NaN;
attempt.freqCompHz = NaN;
attempt.gridRms = 0;
attempt.gridPeak = 0;
attempt.pdcchCrcOk = false;
attempt.sib1CrcOk = false;
attempt.sib1Crc = true;
attempt.failureStage = '';
attempt.error = '';
attempt.bestMeanAbsDciCw = 0;
attempt.lastCandidateError = '';
attempt.bestAggregationLevel = NaN;
attempt.bestCandidateIndex = NaN;
attempt.dciSlot = NaN;
attempt.firstOrSecondSlot = NaN;
attempt.dci = struct();
attempt.pdschK0 = NaN;
attempt.pdschPRBStart = NaN;
attempt.pdschPRBLength = NaN;
attempt.pdschSymbolStart = NaN;
attempt.pdschSymbolLength = NaN;
attempt.pdschModulation = '';
attempt.sib1BitCount = 0;
attempt.sib1Bits = [];
end

function out = finiteScalar(value)
out = double(value);
if isempty(out) || ~isfinite(out)
    out = 0;
else
    out = out(1);
end
end

function sampleOffset = standardSsbSampleOffset(ssbPattern, ssbIndex, sampleRate, nfft, scsSSB)
if exist('nrhdlexamples.coreset0', 'class') == 8
    if strcmpi(ssbPattern, 'Case C')
        lmax = 8;
    elseif strcmpi(ssbPattern, 'Case B')
        lmax = 8;
    elseif strcmpi(ssbPattern, 'Case A')
        lmax = 8;
    else
        lmax = 64;
    end
    startSymbols = nrhdlexamples.coreset0.ssbStartSymbols(ssbPattern, lmax);
    ssbSymbol = startSymbols(ssbIndex + 1);
else
    if strcmpi(ssbPattern, 'Case C')
        ssbSymbol = 2;
    elseif strcmpi(ssbPattern, 'Case B')
        ssbSymbol = 4;
    else
        ssbSymbol = 2;
    end
end

if nargin < 5 || isempty(scsSSB)
    if strcmpi(ssbPattern, 'Case A')
        scsSSB = 15;
    elseif strcmpi(ssbPattern, 'Case B') || strcmpi(ssbPattern, 'Case C')
        scsSSB = 30;
    elseif strcmpi(ssbPattern, 'Case D')
        scsSSB = 120;
    else
        scsSSB = 240;
    end
end

if nargin < 4 || isempty(nfft)
    nfft = sampleRate / (scsSSB * 1e3);
end
nfft = round(double(nfft));
normalCp = round(144 * nfft / 2048);
mu = round(log2(double(scsSSB) / 15));
slotDuration = 1e-3 / (2 ^ mu);
totalSamplesPerSlot = round(double(sampleRate) * slotDuration);
longCp = round(totalSamplesPerSlot - 14 * nfft - 13 * normalCp);
if ~isfinite(longCp) || longCp <= 0
    longCp = round(160 * nfft / 2048);
end

sampleOffset = 0;
for sym = 0:(ssbSymbol - 1)
    if mod(sym, 14) == 0
        sampleOffset = sampleOffset + longCp + nfft;
    else
        sampleOffset = sampleOffset + normalCp + nfft;
    end
end
end

function writeJson(path, value)
jsonText = jsonencode(value, PrettyPrint=true);
fid = fopen(path, 'w');
if fid < 0
    error('Could not open output JSON: %s', path);
end
cleanupObj = onCleanup(@() fclose(fid));
fwrite(fid, jsonText, 'char');
end
