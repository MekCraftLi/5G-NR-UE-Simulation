function matlab_pbch_mib_oracle()
% MATLAB PBCH/MIB oracle check for data/rxSignal.npy
% Uses current project candidate assumptions:
% nfft=2048, cp=152, ssbSubcarrierOffset=16, NCellID=5.

outDir = fullfile('output');
if ~exist(outDir, 'dir')
    mkdir(outDir);
end

rxPath = fullfile('data', 'rxSignal.npy');
if exist('readNPY', 'file') ~= 2
    error('readNPY not found on MATLAB path');
end
rx = readNPY(rxPath);
rx = rx(:);

fs = 30.72e6;
scs = 15;
nfft = 2048;
cp = 152;
ncellid = 5;
ssbSubcarrierOffset = 16; % 0-based

nrbSSB = 20;
refGrid = zeros(nrbSSB * 12, 2);

% PSS uses N_ID_2 from NCellID.
nid2 = mod(ncellid, 3);
refGrid(nrPSSIndices, 2) = nrPSS(nid2);
[timingOffset, corr] = nrTimingEstimate(rx, nrbSSB, scs, 0, refGrid, 'SampleRate', fs);

rxAligned = rx(1 + timingOffset:end);
rxGrid = nrOFDMDemodulate(rxAligned, nrbSSB, scs, 0, 'SampleRate', fs);

% Find strongest PSS symbol in current slot.
pssIdx = nrPSSIndices;
pssSc = mod(pssIdx - 1, 240) + 1;
pssCorr = zeros(1, size(rxGrid, 2));
for s = 1:size(rxGrid, 2)
    pssCorr(s) = abs(mean(rxGrid(pssSc, s) .* conj(nrPSS(nid2))));
end
[~, pssSym] = max(pssCorr);
if pssSym + 3 > size(rxGrid, 2)
    error('Not enough symbols for one SSB block');
end
ssbGrid = rxGrid(:, pssSym:pssSym + 3);

% SSS verify
sssIdx = nrSSSIndices;
sssRx = ssbGrid(sssIdx);
bestSss = -inf;
bestNid1 = 0;
for nid1 = 0:335
    c = abs(mean(sssRx .* conj(nrSSS(3 * nid1 + nid2))));
    if c > bestSss
        bestSss = c;
        bestNid1 = nid1;
    end
end
ncellidBlind = 3 * bestNid1 + nid2;

% Force oracle NCellID as requested, but keep blind estimate in result.
ncellidUsed = ncellid;
dmrsIdx = nrPBCHDMRSIndices(ncellidUsed);

% Scan iSSBbar by DMRS correlation.
dmrsCorr = zeros(1, 8);
for ibar = 0:7
    dmrsRef = nrPBCHDMRS(ncellidUsed, ibar);
    dmrsCorr(ibar + 1) = abs(mean(ssbGrid(dmrsIdx) .* conj(dmrsRef)));
end
[~, bestIbarIdx] = max(dmrsCorr);
iSsbBar = bestIbarIdx - 1;

% Channel estimate + PBCH equalization.
refGridSSB = zeros(240, 4);
refGridSSB(dmrsIdx) = nrPBCHDMRS(ncellidUsed, iSsbBar);
refGridSSB(sssIdx) = nrSSS(ncellidUsed);
[hest, nVar] = nrChannelEstimate(ssbGrid, refGridSSB, 'AveragingWindow', [1 1]);

[pbchIdx, ~] = nrPBCHIndices(ncellidUsed);
[pbchEq, csi] = nrEqualizeMMSE(ssbGrid(pbchIdx), hest(pbchIdx), nVar);

% PBCH decode (returns BCH coded bits soft values / LLR style).
v = mod(iSsbBar, 8);
bchSoft = nrPBCHDecode(pbchEq(:), ncellidUsed, v, nVar);

% BCH decode + CRC + MIB parse.
% R2024b official signature:
% [scrblk,errFlag,trblk,lsbofsfn,hrf,msbidxoffset] =
%     nrBCHDecode(softbits,listLength,lssb,ncellid)
crcOk = false;
mibBits = [];
extraOut = struct();
errMsg = '';
try
    listLength = 8;
    lssb = 8;
    [scrblk, errFlag, trblk, lsbofsfn, hrf, msbidxoffset] = nrBCHDecode(bchSoft, listLength, lssb, ncellidUsed);
    crcOk = ~logical(errFlag);
    mibBits = trblk;
    extraOut.scrblk = scrblk;
    extraOut.lssb = lssb;
    extraOut.listLength = listLength;
    extraOut.lsbofsfn = lsbofsfn;
    extraOut.hrf = hrf;
    extraOut.msbidxoffset = msbidxoffset;
catch ex1
    try
        listLength = 8;
        lssb = 4;
        [scrblk, errFlag, trblk, lsbofsfn, hrf, msbidxoffset] = nrBCHDecode(bchSoft, listLength, lssb, ncellidUsed);
        crcOk = ~logical(errFlag);
        mibBits = trblk;
        extraOut.scrblk = scrblk;
        extraOut.lssb = lssb;
        extraOut.listLength = listLength;
        extraOut.lsbofsfn = lsbofsfn;
        extraOut.hrf = hrf;
        extraOut.msbidxoffset = msbidxoffset;
    catch ex2
        errMsg = sprintf('nrBCHDecode failed: %s | %s', ex1.message, ex2.message);
    end
end

mibInfo = struct();
if ~isempty(mibBits)
    mibInfo.rawMibBits = double(mibBits(:)).';
    mibInfo.sfnMsb6 = bit2int(double(mibBits(2:7)), 6);
    if isfield(extraOut, 'lsbofsfn')
        mibInfo.sfnLsb4 = bit2int(double(extraOut.lsbofsfn(:)), 4);
        mibInfo.sfn10 = mibInfo.sfnMsb6 * 16 + mibInfo.sfnLsb4;
    end
    mibInfo.subCarrierSpacingCommonBit = double(mibBits(8));
    mibInfo.ssbSubcarrierOffsetLsb4 = bit2int(double(mibBits(9:12)), 4);
    mibInfo.dmrsTypeAPosition = 2 + double(mibBits(13));
    mibInfo.pdcchConfigSIB1 = bit2int(double(mibBits(14:21)), 8);
    mibInfo.cellBarredBit = double(mibBits(22));
    mibInfo.intraFreqReselectionBit = double(mibBits(23));
    mibInfo.spareBit = double(mibBits(24));
end

result = struct();
result.timestamp = datestr(now, 'yyyy-mm-dd HH:MM:SS');
result.rxPath = which(rxPath);
result.fs = fs;
result.scskHz = scs;
result.nfft = nfft;
result.cp = cp;
result.ssbSubcarrierOffset = ssbSubcarrierOffset;
result.nCellIdOracle = ncellid;
result.nCellIdBlind = ncellidBlind;
result.nid2 = nid2;
result.pssTimingOffset = timingOffset;
result.pssSymbolInSlot = pssSym - 1;
result.bestSssNid1 = bestNid1;
result.dmrsCorr = dmrsCorr;
result.iSsbBar = iSsbBar;
result.v = v;
result.noiseVar = nVar;
result.pbchEqCount = length(pbchEq);
result.csiMean = mean(abs(csi));
result.crcOk = logical(crcOk);
result.mibBits = mibBits;
result.mibInfo = mibInfo;
result.error = errMsg;
result.matlabVersion = version;

save(fullfile(outDir, 'matlab_pbch_mib_oracle.mat'), 'result', 'pbchEq', 'bchSoft');

fprintf('MATLAB PBCH/MIB oracle done.\n');
fprintf('  NCellID(oracle/blind) = %d / %d\n', ncellid, ncellidBlind);
fprintf('  iSsbBar=%d, v=%d, CRC=%d\n', iSsbBar, v, crcOk);
if ~isempty(errMsg)
    fprintf('  BCH decode warning: %s\n', errMsg);
end
end
