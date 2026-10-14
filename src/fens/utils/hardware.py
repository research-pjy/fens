"""
Hardware & Environment Verification for the `rama` L40S Compute Node.

Follows `L40S Execution Context.md`:
  - Hostname check (`rama` vs local development machine)
  - Conda environment check (`qwen-vl`)
  - GPU detection (NVIDIA L40S, ~44 GiB usable VRAM)
  - BF16 hardware support verification
  - System RAM (~247 GiB) and disk space inspection
  - Global deterministic seeding
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import sys
from typing import Optional

import numpy as np


@dataclass
class EnvironmentReport:
    """Summary of the runtime hardware and software environment."""

    hostname: str
    conda_env: str
    python_version: str
    torch_version: str
    cuda_available: bool
    cuda_version: str
    gpu_name: str
    gpu_count: int
    vram_total_gib: float
    vram_free_gib: float
    bf16_supported: bool
    system_ram_gib: float
    disk_free_gib: float

    def format_banner(self) -> str:
        lines = [
            "=" * 72,
            "  FENS EXECUTION ENVIRONMENT REPORT",
            "=" * 72,
            f"  Hostname         : {self.hostname} {'(L40S Remote Node)' if 'rama' in self.hostname.lower() else '(Local / Dev Machine)'}",
            f"  Conda Env        : {self.conda_env}",
            f"  Python / PyTorch : Python {self.python_version} | PyTorch {self.torch_version}",
            f"  CUDA Available   : {self.cuda_available} (CUDA {self.cuda_version})",
            f"  GPU Device       : {self.gpu_name} (count={self.gpu_count})",
            f"  GPU VRAM         : {self.vram_free_gib:.2f} GiB free / {self.vram_total_gib:.2f} GiB total",
            f"  BF16 Supported   : {self.bf16_supported}",
            f"  System RAM       : {self.system_ram_gib:.1f} GiB",
            f"  Free Disk Space  : {self.disk_free_gib:.1f} GiB",
            "=" * 72,
        ]
        return "\n".join(lines)


def set_global_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, and PyTorch (if installed)."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _get_system_ram_gib() -> float:
    """Reads total system RAM in GiB from /proc/meminfo or os.sysconf."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024.0 ** 2)
    except OSError:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024.0 ** 3)
    except (ValueError, AttributeError):
        pass
    return 0.0


def inspect_execution_environment(work_dir: Path | str = ".") -> EnvironmentReport:
    """Inspects and returns a complete `EnvironmentReport` for the current machine."""
    hostname = socket.gethostname()
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "(none)")
    py_ver = platform.python_version()

    disk_usage = shutil.disk_usage(Path(work_dir).resolve())
    disk_free_gib = disk_usage.free / (1024.0 ** 3)
    sys_ram_gib = _get_system_ram_gib()

    try:
        import torch

        torch_ver = torch.__version__
        cuda_avail = bool(torch.cuda.is_available())
        cuda_ver = str(torch.version.cuda or "N/A")
        if cuda_avail:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            vram_free_gib = free_b / (1024.0 ** 3)
            vram_total_gib = total_b / (1024.0 ** 3)
            bf16_ok = bool(torch.cuda.is_bf16_supported())
        else:
            gpu_count = 0
            gpu_name = "CPU Only"
            vram_free_gib = 0.0
            vram_total_gib = 0.0
            bf16_ok = False
    except ImportError:
        torch_ver = "(not installed)"
        cuda_avail = False
        cuda_ver = "N/A"
        gpu_count = 0
        gpu_name = "(torch not installed)"
        vram_free_gib = 0.0
        vram_total_gib = 0.0
        bf16_ok = False

    return EnvironmentReport(
        hostname=hostname,
        conda_env=conda_env,
        python_version=py_ver,
        torch_version=torch_ver,
        cuda_available=cuda_avail,
        cuda_version=cuda_ver,
        gpu_name=gpu_name,
        gpu_count=gpu_count,
        vram_total_gib=vram_total_gib,
        vram_free_gib=vram_free_gib,
        bf16_supported=bf16_ok,
        system_ram_gib=sys_ram_gib,
        disk_free_gib=disk_free_gib,
    )
