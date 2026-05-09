"""Device and accelerator detection utilities."""
import os
import platform
import subprocess
import sys
from typing import Literal, Union


def detect_accelerator(accelerator: str = "auto") -> str:
    """Detect and validate the accelerator type.

    Args:
        accelerator: The accelerator to use. Can be "auto", "cpu", "cuda", or "mps".

    Returns:
        The resolved accelerator string.

    Raises:
        ValueError: If accelerator is not a valid option.
    """
    accelerator = accelerator.lower() if isinstance(accelerator, str) else accelerator

    valid_accelerators = ("auto", "cpu", "cuda", "mps")
    if accelerator not in valid_accelerators:
        raise ValueError(
            f"accelerator must be one of {valid_accelerators}, got '{accelerator}'"
        )

    if accelerator == "auto":
        if check_cuda_with_nvidia_smi() > 0:
            return "cuda"
        if _check_mps_available():
            return "mps"
        return "cpu"

    return accelerator


def check_cuda_with_nvidia_smi() -> int:
    """Check for CUDA devices using nvidia-smi command.

    Returns:
        Number of available CUDA devices, respecting CUDA_VISIBLE_DEVICES.
    """
    try:
        output = subprocess.check_output(["nvidia-smi", "-L"]).decode("utf-8").strip()
        devices = [line for line in output.split("\n") if line.startswith("GPU")]
        device_ids = [line.split(":")[0].split()[1] for line in devices]

        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_set = set(visible_devices.split(","))
            device_ids = [d for d in device_ids if d in visible_set]

        return len(device_ids)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0


def _check_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available.

    Returns:
        True if MPS is available, False otherwise.
    """
    if platform.processor() not in ("arm", "arm64"):
        return False

    try:
        import torch
        return torch.backends.mps.is_available()
    except (ImportError, AttributeError):
        return False


def detect_device_count(accelerator: str) -> int:
    """Detect the number of available devices for the given accelerator.

    Args:
        accelerator: The accelerator type ("cpu", "cuda", "mps").

    Returns:
        Number of available devices.
    """
    if accelerator == "cuda":
        return check_cuda_with_nvidia_smi() or 1
    return 1


def get_device_identifiers(
    accelerator: str,
    num_devices: int,
    workers_per_device: int,
) -> list[str]:
    """Generate device identifier strings for worker processes.

    Args:
        accelerator: The accelerator type ("cpu", "cuda", "mps").
        num_devices: Number of devices to use.
        workers_per_device: Number of workers per device.

    Returns:
        List of device identifier strings.
    """
    if accelerator == "cpu":
        return ["cpu"] * workers_per_device

    identifiers = []
    for device_idx in range(num_devices):
        for _ in range(workers_per_device):
            identifiers.append(f"{accelerator}:{device_idx}")
    return identifiers
