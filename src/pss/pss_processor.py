import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

try:
    from src.config import AppConfig, GlobalConfig
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.append(str(repo_root))
    from src.config import AppConfig, GlobalConfig


console = Console()


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("pss_processor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RichHandler(console=console, show_path=False, rich_tracebacks=True)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return logger


class PssProcessor:
    def __init__(
        self,
        config: GlobalConfig = GlobalConfig(),
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.logger = logger

    def __m_sequence_generator(self, n_id_2) -> np.ndarray:
        if n_id_2 not in [0, 1, 2]:
            raise ValueError("n_id_2 not in [0, 1, 2]")

        # 生成长度127 m序列
        reg_7 = np.array([0, 1, 1, 0, 1, 1, 1], dtype=np.int8)
        x = np.zeros(127 + 7, dtype=np.int8)
        x[:7] = reg_7

        for n in range(127):
            x[n + 7] = (x[n + 4] + x[n]) & 1

        return x

    """
    @brief 本地PSS频偏矩阵生成
    """

    def generate_pss_bank(self, n_id_2: int) -> np.ndarray:
        if self.logger:
            self.logger.info(f"[PSS] generate_pss_bank start (N_ID_2={n_id_2})")

        # 得到M序列
        x = self.__m_sequence_generator(n_id_2)
        m = 43 * n_id_2

        # 得到频域PSS
        seq = np.empty(127, dtype=np.int8)
        indices = (np.arange(127) + m) % 127
        seq = 1.0 - 2.0 * x[indices]

        # 构建PSS频偏矩阵
        c0 = self.config.fft_size // 2
        delta_k = [x for x in range(50)]

        bank = np.zeros((50, self.config.fft_size), dtype=np.complex64)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Building PSS bank", total=len(delta_k))

            # 遍历子载波偏移位置
            for i, dk in enumerate(delta_k):
                c = c0 + delta_k[i] - 25
                if c - 63 < 0 or c + 64 > self.config.fft_size:
                    progress.advance(task)
                    continue

                grid = np.zeros(self.config.fft_size, dtype=np.complex64)
                grid[c - 63 : c + 64] = seq

                # 将原本的m序列放在IFFT的窗口中然后进行IFFT变换，得到时域
                bank[i] = np.fft.ifft(np.fft.ifftshift(grid))
                progress.advance(task)

        if self.logger:
            self.logger.info(f"[PSS] generate_pss_bank done, bank shape={bank.shape}")

        return bank

    def search_pss(self, signal: np.ndarray, time_view_len: int = 8800) -> dict:
        bank_0 = self.generate_pss_bank(0)
        bank_1 = self.generate_pss_bank(1)
        bank_2 = self.generate_pss_bank(2)

        banks = {
            0: bank_0,
            1: bank_1,
            2: bank_2,
        }
        # 构建频偏序列
        delta_k = np.arange(50) - 25
        pss_results = {}

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("[magenta]XCORR N_ID_2=0/1/2", total=3)

            # 进行本地PSS与信号的滑动互相关搜索
            # valid模式，意味着是本地PSS的第一个值与信号的第一个值对齐之后进行
            for n_id_2 in [0, 1, 2]:
                # 进行滑动互相关, 使用fftconvolve，加速卷积运算
                C = scipy.signal.fftconvolve(
                    signal[None, :],
                    np.conj(banks[n_id_2][:, ::-1]),
                    mode="valid",
                    axes=-1,
                )

                # 将互相关结果归一化
                M = np.abs(C)
                template_energy = np.sum(np.abs(banks[n_id_2]) ** 2, axis=1)
                window_energy = np.convolve(
                    np.abs(signal) ** 2, np.ones(self.config.fft_size), mode="valid"
                )
                denom = np.sqrt(template_energy[:, None] * window_energy[None, :])
                M_norm = np.zeros_like(M)
                np.divide(M, denom, out=M_norm, where=denom > 0)
                M = M_norm

                peak_idx = np.argmax(M, axis=1)
                peak_value = M[np.arange(M.shape[0]), peak_idx]
                best_row = int(np.argmax(peak_value))
                best_metric = float(peak_value[best_row])
                best_peak_idx = int(peak_idx[best_row])
                best_timing = best_peak_idx - (self.config.fft_size - 1)

                pss_results[n_id_2] = {
                    "peak_idx": peak_idx,
                    "peak_value": peak_value,
                    "best_row": best_row,
                    "best_metric": best_metric,
                    "best_peak_idx": best_peak_idx,
                    "best_timing": best_timing,
                    "time_curve": M[best_row, :time_view_len].copy(),
                }

                if self.logger:
                    self.logger.info(
                        f"[MAIN][N_ID_2={n_id_2}] best_row={best_row}, "
                        f"delta_k={best_row - 120}, best_metric={best_metric:.6f}, "
                        f"peak_idx={best_peak_idx}, timing={best_timing}"
                    )
                progress.advance(task)

                del C
                del M

        best_n_id_2 = max(
            pss_results.keys(), key=lambda k: pss_results[k]["best_metric"]
        )
        if self.logger:
            self.logger.info(
                f"[MAIN] overall best N_ID_2={best_n_id_2}, "
                f"metric={pss_results[best_n_id_2]['best_metric']:.6f}, "
                f"timing={pss_results[best_n_id_2]['best_timing']}"
            )

        return {
            "banks": banks,
            "delta_k": delta_k,
            "pss_results": pss_results,
            "best_n_id_2": best_n_id_2,
        }

    def freq_offset_detect(self):
        pass


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("[MAIN] pss_processor started")

    pss_processor = PssProcessor(logger=logger)

    data_path = (
        Path(__file__).resolve().parents[2]
        / AppConfig().io_cfg.data_dir
        / "txsig0_frame0.npy"
    )
    signal = np.load(data_path)

    signal_size = len(signal)
    logger.info(f"[MAIN] signal loaded, size={signal_size}")

    time_view_len = 8800
    search_result = pss_processor.search_pss(signal=signal, time_view_len=time_view_len)
    delta_k = search_result["delta_k"]
    pss_results = search_result["pss_results"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for col, n_id_2 in enumerate([0, 1, 2]):
        result = pss_results[n_id_2]

        axes[0, col].plot(delta_k, result["peak_value"], lw=1.0)
        axes[0, col].axvline(result["best_row"] - 25, color="r", ls="--", lw=1.0)
        axes[0, col].set_title(f"N_ID_2={n_id_2} Frequency Scan")
        axes[0, col].set_xlabel("delta_k")
        axes[0, col].set_ylabel("peak correlation")
        axes[0, col].grid(True, alpha=0.3)

        axes[1, col].plot(np.arange(time_view_len), result["time_curve"], lw=1.0)
        axes[1, col].set_title(f"N_ID_2={n_id_2} Time Scan (0~{time_view_len - 1})")
        axes[1, col].set_xlabel("sample index")
        axes[1, col].set_ylabel("|correlation|")
        axes[1, col].grid(True, alpha=0.3)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parents[2] / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "pss_nid2_freq_time_scan.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"[MAIN] figure saved: {fig_path}")

    # CP检测

    times = int(pss_processor.config.fft_size / 1024)

    cp_len = times * 88
    logger.info(f"[CP] cp_len={cp_len}")

    cp_head = signal[:cp_len]
    cp_tail = signal[
        pss_processor.config.fft_size : pss_processor.config.fft_size + cp_len
    ]

    corr_raw = scipy.signal.correlate(signal[: times * 2000], cp_head, mode="valid")
    window_energy = np.convolve(
        np.abs(signal[: times * 2000]) ** 2, np.ones(len(cp_head)), mode="valid"
    )
    template_energy = np.sum(np.abs(cp_head) ** 2)
    corr_norm = np.abs(corr_raw) / np.sqrt(window_energy * template_energy)
    plt.plot(corr_norm)
    plt.show()
