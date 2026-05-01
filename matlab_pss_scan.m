%% PSS 盲搜频率扫描 — MATLAB 参考实现
% 对比 Python 代码，排除数学实现错误
% 3GPP TS 38.211 Clause 7.4.2.2.1

clear; close all;

%% 参数
fs = 30.72e6;           % 采样率
scs = 15000;            % 子载波间隔
Nfft = fs / scs;        % FFT 点数 = 2048
cpLen = 144;            % 正常 CP
symLen = Nfft + cpLen;  % 含 CP 的符号长度 = 2192

%% 加载信号 (TX 参考信号)
load('data/txs0_seg.mat', 'txSignal');
rxSignal = txSignal(:);  % 列向量
fprintf('TX Signal: %d samples, %.3f ms\n', length(rxSignal), length(rxSignal)/fs*1000);

%% 生成 PSS 序列 (TS 38.211 Clause 7.4.2.2.1)
pssTemplates = cell(1, 3);  % 时域模板 (含 CP)
pssFreqDom = cell(1, 3);    % 频域序列 (用于频域相关)

for nId2 = 0:2
    % m-sequence
    x = zeros(1, 127+7, 'int8');
    x(1:7) = [0 1 1 0 1 1 1];
    for n = 1:127
        x(n+7) = mod(x(n+4) + x(n), 2);
    end

    % PSS 序列 (BPSK)
    m = 43 * nId2;
    pssSeq = zeros(1, 127);
    for n = 1:127
        pssSeq(n) = 1 - 2 * double(x(mod(n-1 + m, 127) + 1));
    end

    % 频域放置: 子载波 -63 到 +63 (共 127 个), DC 载波不用于 PSS
    freqDom = zeros(Nfft, 1);
    center = Nfft/2 + 1;  % MATLAB 1-based: DC at Nfft/2+1
    freqDom(center-63:center+63) = pssSeq;

    % 时域 OFDM 符号 (IFFT)
    timeNoCp = ifft(ifftshift(freqDom)) * Nfft;  % *Nfft 保持能量
    % 加 CP
    timeWithCp = [timeNoCp(end-cpLen+1:end); timeNoCp];
    % 归一化
    timeWithCp = timeWithCp / norm(timeWithCp);

    pssTemplates{nId2+1} = timeWithCp;
    pssFreqDom{nId2+1} = freqDom;

    fprintf('N_ID_2=%d: template length=%d, norm=%.3f\n', ...
        nId2, length(timeWithCp), norm(timeWithCp));
end

%% 频率扫描
freqStart = -2000;     % Hz (TX信号CFO较小)
freqEnd   = 2000;
freqStep  = 200;       % Hz
freqGrid  = freqStart:freqStep:freqEnd;
nFreqs    = length(freqGrid);

fprintf('Frequency scan: %d points, step=%.0f Hz, range=[%.1f, %.1f] kHz\n', ...
    nFreqs, freqStep, freqStart/1e3, freqEnd/1e3);

% 预计算 FFT 卷积参数
templateLen = length(pssTemplates{1});
convLen = length(rxSignal) + templateLen - 1;
corrNfft = 2^nextpow2(convLen);
validStart = templateLen;
validEnd = length(rxSignal) + 1;

fprintf('corrNfft=%d, valid range=[%d, %d]\n', corrNfft, validStart, validEnd);

% 预计算模板 FFT (时域反转 → 共轭 → FFT)
templateFFTs = cell(1, 3);
for nId2 = 0:2
    kernel = conj(flip(pssTemplates{nId2+1}));
    templateFFTs{nId2+1} = fft(kernel, corrNfft);
end

% 频率扫描
peakMatrix = zeros(nFreqs, 3);  % [freqIdx, nId2]
timingMatrix = zeros(nFreqs, 3);
bestPeak = 0;
bestFreq = 0;
bestNid2 = 0;
bestTiming = 0;

tStart = tic;
fprintf('Scanning...\n');
for iFreq = 1:nFreqs
    freqHz = freqGrid(iFreq);

    % 频率补偿
    t = (0:length(rxSignal)-1)';
    phase = exp(-1j * 2 * pi * freqHz * t / fs);
    compensated = rxSignal .* phase;

    % FFT
    sigFft = fft(compensated, corrNfft);

    for nId2 = 0:2
        % 互相关 (IFft of product)
        corrFull = ifft(sigFft .* templateFFTs{nId2+1});
        corrValid = abs(corrFull(validStart:validEnd));

        [peakVal, peakIdx] = max(corrValid);
        peakMatrix(iFreq, nId2+1) = peakVal;
        timingMatrix(iFreq, nId2+1) = peakIdx;

        if peakVal > bestPeak
            bestPeak = peakVal;
            bestFreq = freqHz;
            bestNid2 = nId2;
            bestTiming = peakIdx;
        end
    end

    if mod(iFreq, max(1, floor(nFreqs/10))) == 0 || iFreq == nFreqs
        fprintf('  %d/%d (%.0f%%), best: freq=%.0f Hz, N_ID_2=%d, peak=%.2f\n', ...
            iFreq, nFreqs, 100*iFreq/nFreqs, bestFreq, bestNid2, bestPeak);
    end
end
elapsed = toc(tStart);
fprintf('Scan done: %.1f s\n', elapsed);

%% 结果
fprintf('\n===== BEST RESULT =====\n');
fprintf('  N_ID_2:       %d\n', bestNid2);
fprintf('  Frequency:    %.2f Hz (%.3f kHz)\n', bestFreq, bestFreq/1e3);
fprintf('  Timing:       %d samples\n', bestTiming);
fprintf('  Peak:         %.4f\n', bestPeak);
fprintf('  Peak ratio:   %.2f\n', bestPeak / median(peakMatrix(:)));

%% 绘图
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250]};

% -- 图 1: 所有 N_ID_2 的频率扫描热力图 --
figure('Position', [100 100 1400 500]);
hold on;
for nId2 = 0:2
    plot(freqGrid/1e3, peakMatrix(:, nId2+1), '.-', ...
        'Color', colors{nId2+1}, 'LineWidth', 1.2, ...
        'MarkerSize', 8, 'DisplayName', sprintf('N_{ID}^{(2)}=%d', nId2));
end
xline(bestFreq/1e3, 'r--', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('Best: %.0f Hz (N_{ID}^{(2)}=%d)', bestFreq, bestNid2));
xlabel('Frequency (kHz)');
ylabel('Peak Correlation');
title('MATLAB PSS Frequency Scan');
legend('Location', 'best'); grid on;
saveas(gcf, 'output/matlab_tx_pss_freq_scan.png');

% -- 图 2: 最佳频率处的各 N_ID_2 互相关 --
figure('Position', [100 100 1400 800]);
for nId2 = 0:2
    % 在最佳频率重算互相关
    t = (0:length(rxSignal)-1)';
    phase = exp(-1j * 2 * pi * bestFreq * t / fs);
    compensated = rxSignal .* phase;
    sigFft = fft(compensated, corrNfft);
    corrFull = ifft(sigFft .* templateFFTs{nId2+1});
    corrValid = abs(corrFull(validStart:validEnd));

    subplot(2, 2, nId2+1);
    scanRange = min(15000, length(corrValid));
    plot(0:scanRange-1, corrValid(1:scanRange), 'Color', colors{nId2+1});
    hold on;
    [~, pk] = max(corrValid(1:scanRange));
    plot(pk, corrValid(pk), 'r^', 'MarkerSize', 8);
    title(sprintf('N_{ID}^{(2)}=%d at %.0f Hz | peak=%.1f @ %d', ...
        nId2, bestFreq, corrValid(pk), pk));
    xlabel('Timing offset (samples)'); ylabel('|Correlation|');
    grid on;
end

% 叠加
subplot(2, 2, 4); hold on;
for nId2 = 0:2
    t = (0:length(rxSignal)-1)';
    phase = exp(-1j * 2 * pi * bestFreq * t / fs);
    compensated = rxSignal .* phase;
    sigFft = fft(compensated, corrNfft);
    corrFull = ifft(sigFft .* templateFFTs{nId2+1});
    corrValid = abs(corrFull(validStart:validEnd));
    scanRange = min(15000, length(corrValid));
    plot(0:scanRange-1, corrValid(1:scanRange), 'Color', colors{nId2+1}, ...
        'DisplayName', sprintf('N_{ID}^{(2)}=%d', nId2));
end
plot(bestTiming, bestPeak, 'r^', 'MarkerSize', 10, 'LineWidth', 2);
title(sprintf('Overlay at best freq=%.0f Hz', bestFreq));
xlabel('Timing offset (samples)'); ylabel('|Correlation|');
legend; grid on;
saveas(gcf, 'output/matlab_tx_pss_correlation.png');

fprintf('\nPlots saved to output/\n');
fprintf('===== DONE =====\n');
