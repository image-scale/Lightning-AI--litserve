"""Tests for the InferenceServer class."""
import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from aiserver import InferenceAPI, InferenceServer


class SimpleAPI(InferenceAPI):
    """Simple test API implementation."""

    def setup(self, device):
        self.model = lambda x: x ** 2
        self.device = device

    def decode_request(self, request):
        return request["input"]

    def predict(self, x):
        return self.model(x)

    def encode_response(self, output):
        return {"output": output}


class SlowAPI(InferenceAPI):
    """API that takes time to respond."""

    def setup(self, device):
        self.device = device

    def predict(self, x):
        time.sleep(0.5)
        return x * 2


class ErrorAPI(InferenceAPI):
    """API that raises errors."""

    def setup(self, device):
        self.device = device

    def predict(self, x):
        raise ValueError("Test error")


class TestServerInit:
    """Tests for InferenceServer initialization."""

    def test_default_parameters(self):
        api = SimpleAPI()
        server = InferenceServer(api)
        assert server.timeout == 30
        assert server.workers_per_device == 1
        assert server.model_metadata is None

    def test_custom_parameters(self):
        api = SimpleAPI()
        server = InferenceServer(
            api,
            accelerator="cpu",
            devices=2,
            workers_per_device=2,
            timeout=60,
            model_metadata={"name": "test-model"},
        )
        assert server.timeout == 60
        assert server.workers_per_device == 2
        assert server.model_metadata == {"name": "test-model"}

    def test_timeout_disabled(self):
        api = SimpleAPI()
        server = InferenceServer(api, timeout=False)
        assert server.timeout == -1

    def test_invalid_workers_per_device(self):
        api = SimpleAPI()
        with pytest.raises(ValueError, match="workers_per_device must be >= 1"):
            InferenceServer(api, workers_per_device=0)

    def test_api_path_from_api(self):
        api = SimpleAPI(api_path="/inference")
        server = InferenceServer(api)
        assert server.api.api_path == "/inference"


class TestServerAccelerator:
    """Tests for accelerator resolution."""

    def test_cpu_accelerator(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu")
        assert server._accelerator == "cpu"

    def test_auto_accelerator_no_torch(self):
        api = SimpleAPI()
        with patch.dict('sys.modules', {'torch': None}):
            server = InferenceServer(api, accelerator="auto")
            assert server._accelerator == "cpu"

    def test_device_identifiers_cpu(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", workers_per_device=2)
        devices = server._get_device_identifiers()
        assert devices == ["cpu", "cpu"]

    def test_device_identifiers_cuda(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cuda", devices=2, workers_per_device=1)
        devices = server._get_device_identifiers()
        assert devices == ["cuda:0", "cuda:1"]


class TestServerEndpoints:
    """Tests for server endpoints."""

    @pytest.fixture
    def server(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu")
        return server

    def test_index_endpoint(self, server):
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.get("/")
        assert response.status_code == 200
        assert response.text == "aiserver running"

    def test_health_not_ready(self, server):
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.get("/health")
        assert response.status_code == 503
        assert response.text == "not ready"

    def test_info_endpoint(self, server):
        server.model_metadata = {"version": "1.0"}
        client = TestClient(server.app, raise_server_exceptions=False)
        response = client.get("/info")
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == {"version": "1.0"}
        assert data["server"]["accelerator"] == "cpu"
        assert data["server"]["api_path"] == "/predict"

    def test_custom_api_path(self):
        api = SimpleAPI(api_path="/classify")
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["api_path"] == "/classify"


class TestServerValidation:
    """Tests for server input validation."""

    def test_invalid_port_low(self):
        api = SimpleAPI()
        server = InferenceServer(api)
        with pytest.raises(ValueError, match="port must be between"):
            server.run(port=1000)

    def test_invalid_port_high(self):
        api = SimpleAPI()
        server = InferenceServer(api)
        with pytest.raises(ValueError, match="port must be between"):
            server.run(port=70000)


class TestServerWorkers:
    """Tests for worker management."""

    def test_worker_status_tracking(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", workers_per_device=1)
        assert server._worker_status is None

    def test_resolve_devices_auto_cpu(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", devices="auto")
        assert server._devices == 1


class TestServerIntegration:
    """Integration tests with mock workers."""

    def test_predict_endpoint_registered(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu")

        routes = [route.path for route in server.app.routes]
        assert "/predict" in routes
        assert "/health" in routes
        assert "/info" in routes
        assert "/" in routes

    def test_info_shows_timeout(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", timeout=45)
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["timeout"] == 45

    def test_info_shows_workers(self):
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", workers_per_device=3)
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["workers_per_device"] == 3
