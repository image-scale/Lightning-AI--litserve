"""Tests for device/accelerator detection."""
import os
import pytest
from unittest.mock import patch, MagicMock

from aiserver.devices import (
    detect_accelerator,
    detect_device_count,
    get_device_identifiers,
    check_cuda_with_nvidia_smi,
)


class TestDetectAccelerator:
    """Tests for accelerator detection."""

    def test_explicit_cpu_returns_cpu(self):
        result = detect_accelerator("cpu")
        assert result == "cpu"

    def test_explicit_cuda_returns_cuda(self):
        result = detect_accelerator("cuda")
        assert result == "cuda"

    def test_explicit_mps_returns_mps(self):
        result = detect_accelerator("mps")
        assert result == "mps"

    def test_case_insensitive(self):
        assert detect_accelerator("CPU") == "cpu"
        assert detect_accelerator("CUDA") == "cuda"
        assert detect_accelerator("MPS") == "mps"

    def test_invalid_accelerator_raises_error(self):
        with pytest.raises(ValueError) as exc_info:
            detect_accelerator("invalid")
        assert "must be one of" in str(exc_info.value)

    def test_auto_returns_cuda_when_available(self):
        with patch("aiserver.devices.check_cuda_with_nvidia_smi", return_value=2):
            result = detect_accelerator("auto")
            assert result == "cuda"

    def test_auto_returns_cpu_when_no_gpu(self):
        with patch("aiserver.devices.check_cuda_with_nvidia_smi", return_value=0):
            with patch("aiserver.devices._check_mps_available", return_value=False):
                result = detect_accelerator("auto")
                assert result == "cpu"

    def test_auto_returns_mps_when_available(self):
        with patch("aiserver.devices.check_cuda_with_nvidia_smi", return_value=0):
            with patch("aiserver.devices._check_mps_available", return_value=True):
                result = detect_accelerator("auto")
                assert result == "mps"


class TestCheckCudaWithNvidiaSmi:
    """Tests for CUDA detection via nvidia-smi."""

    def test_returns_device_count_from_nvidia_smi(self):
        mock_output = b"GPU 0: NVIDIA GeForce RTX 3090\nGPU 1: NVIDIA GeForce RTX 3090\n"
        with patch("subprocess.check_output", return_value=mock_output):
            result = check_cuda_with_nvidia_smi()
            assert result == 2

    def test_returns_zero_when_nvidia_smi_not_found(self):
        with patch("subprocess.check_output", side_effect=FileNotFoundError):
            result = check_cuda_with_nvidia_smi()
            assert result == 0

    def test_returns_zero_on_subprocess_error(self):
        import subprocess
        with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, "nvidia-smi")):
            result = check_cuda_with_nvidia_smi()
            assert result == 0

    def test_respects_cuda_visible_devices(self):
        mock_output = b"GPU 0: NVIDIA GeForce RTX 3090\nGPU 1: NVIDIA GeForce RTX 3090\nGPU 2: NVIDIA GeForce RTX 3090\n"
        with patch("subprocess.check_output", return_value=mock_output):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}):
                result = check_cuda_with_nvidia_smi()
                assert result == 2

    def test_single_visible_device(self):
        mock_output = b"GPU 0: NVIDIA GeForce RTX 3090\nGPU 1: NVIDIA GeForce RTX 3090\n"
        with patch("subprocess.check_output", return_value=mock_output):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                result = check_cuda_with_nvidia_smi()
                assert result == 1


class TestDetectDeviceCount:
    """Tests for device count detection."""

    def test_cuda_returns_nvidia_smi_count(self):
        with patch("aiserver.devices.check_cuda_with_nvidia_smi", return_value=4):
            result = detect_device_count("cuda")
            assert result == 4

    def test_cpu_returns_one(self):
        result = detect_device_count("cpu")
        assert result == 1

    def test_mps_returns_one(self):
        result = detect_device_count("mps")
        assert result == 1

    def test_auto_devices_with_cuda(self):
        with patch("aiserver.devices.detect_accelerator", return_value="cuda"):
            with patch("aiserver.devices.check_cuda_with_nvidia_smi", return_value=2):
                result = detect_device_count("cuda")
                assert result == 2


class TestGetDeviceIdentifiers:
    """Tests for generating device identifier strings."""

    def test_cpu_single_worker(self):
        result = get_device_identifiers("cpu", 1, 1)
        assert result == ["cpu"]

    def test_cpu_multiple_workers(self):
        result = get_device_identifiers("cpu", 1, 3)
        assert result == ["cpu", "cpu", "cpu"]

    def test_cuda_single_device(self):
        result = get_device_identifiers("cuda", 1, 1)
        assert result == ["cuda:0"]

    def test_cuda_multiple_devices(self):
        result = get_device_identifiers("cuda", 2, 1)
        assert result == ["cuda:0", "cuda:1"]

    def test_cuda_multiple_workers_per_device(self):
        result = get_device_identifiers("cuda", 2, 2)
        assert result == ["cuda:0", "cuda:0", "cuda:1", "cuda:1"]

    def test_mps_single_device(self):
        result = get_device_identifiers("mps", 1, 1)
        assert result == ["mps:0"]

    def test_mps_multiple_workers(self):
        result = get_device_identifiers("mps", 1, 2)
        assert result == ["mps:0", "mps:0"]


class TestMPSDetection:
    """Tests for MPS (Apple Silicon) detection."""

    def test_mps_available_with_torch(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("platform.processor", return_value="arm"):
                from aiserver.devices import _check_mps_available
                result = _check_mps_available()
                assert result is True

    def test_mps_not_available_without_torch(self):
        with patch.dict("sys.modules", {"torch": None}):
            from aiserver import devices
            import importlib
            importlib.reload(devices)
            result = devices._check_mps_available()
            assert result is False

    def test_mps_not_available_on_non_arm(self):
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch("platform.processor", return_value="x86_64"):
                from aiserver.devices import _check_mps_available
                result = _check_mps_available()
                assert result is False


class TestServerIntegration:
    """Tests that server uses device detection module."""

    def test_server_resolves_accelerator_auto(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        with patch("aiserver.server.detect_accelerator", return_value="cpu") as mock_detect:
            api = SimpleAPI()
            server = InferenceServer(api, accelerator="auto")
            mock_detect.assert_called_with("auto")
            assert server._accelerator == "cpu"

    def test_server_resolves_device_count(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        with patch("aiserver.server.detect_accelerator", return_value="cpu"):
            with patch("aiserver.server.detect_device_count", return_value=1) as mock_count:
                api = SimpleAPI()
                server = InferenceServer(api, accelerator="auto", devices="auto")
                mock_count.assert_called_with("cpu")

    def test_server_uses_get_device_identifiers(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", devices=1, workers_per_device=2)
        assert server._accelerator == "cpu"
        assert server._devices == 1
