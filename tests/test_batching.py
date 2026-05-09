"""Tests for request batching functionality."""
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from aiserver import InferenceAPI, InferenceServer


class BatchedAPI(InferenceAPI):
    """API that processes batches."""

    def setup(self, device):
        self.device = device
        self.call_count = 0

    def decode_request(self, request):
        return request["value"]

    def predict(self, batch):
        self.call_count += 1
        if isinstance(batch, list):
            return [x * 2 for x in batch]
        return batch * 2

    def encode_response(self, output):
        return {"result": output}


class TestBatchConfiguration:
    """Tests for batch configuration."""

    def test_default_batch_size(self):
        api = BatchedAPI()
        assert api.max_batch_size == 1

    def test_custom_batch_size(self):
        api = BatchedAPI(max_batch_size=8)
        assert api.max_batch_size == 8

    def test_default_batch_timeout(self):
        api = BatchedAPI()
        assert api.batch_timeout == 0.0

    def test_custom_batch_timeout(self):
        api = BatchedAPI(max_batch_size=4, batch_timeout=0.5)
        assert api.batch_timeout == 0.5


class TestBatchingInfoEndpoint:
    """Tests for batch info in server info endpoint."""

    def test_info_shows_batch_size(self):
        api = BatchedAPI(max_batch_size=4)
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["max_batch_size"] == 4

    def test_info_shows_batch_timeout(self):
        api = BatchedAPI(max_batch_size=4, batch_timeout=0.5)
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["batch_timeout"] == 0.5

    def test_info_shows_default_batch_settings(self):
        api = BatchedAPI()
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["max_batch_size"] == 1
        assert info["server"]["batch_timeout"] == 0.0


class TestBatchCollectionLogic:
    """Tests for batch collection helper function."""

    def test_api_batch_method(self):
        api = BatchedAPI(max_batch_size=4)
        api.pre_setup()

        inputs = [1, 2, 3, 4]
        batched = api.batch(inputs)
        assert batched == inputs

    def test_api_unbatch_method(self):
        api = BatchedAPI(max_batch_size=4)
        api.pre_setup()

        outputs = [2, 4, 6, 8]
        unbatched = api.unbatch(outputs)
        assert unbatched == outputs


class TestBatchProcessing:
    """Tests for batch processing flow."""

    def test_batched_api_predict_with_list(self):
        api = BatchedAPI(max_batch_size=4)
        api.pre_setup()
        api.setup("cpu")

        batch = [1, 2, 3, 4]
        result = api.predict(batch)
        assert result == [2, 4, 6, 8]

    def test_full_batch_flow(self):
        api = BatchedAPI(max_batch_size=4)
        api.pre_setup()
        api.setup("cpu")

        requests = [{"value": 1}, {"value": 2}, {"value": 3}]

        decoded = [api.decode_request(r) for r in requests]
        assert decoded == [1, 2, 3]

        batched = api.batch(decoded)
        output = api.predict(batched)
        unbatched = api.unbatch(output)

        responses = [api.encode_response(o) for o in unbatched]
        assert responses == [{"result": 2}, {"result": 4}, {"result": 6}]


class TestBatchSizes:
    """Tests for different batch size configurations."""

    def test_batch_size_2(self):
        api = BatchedAPI(max_batch_size=2)
        api.pre_setup()
        api.setup("cpu")

        assert api.max_batch_size == 2

    def test_batch_size_4(self):
        api = BatchedAPI(max_batch_size=4)
        api.pre_setup()
        api.setup("cpu")

        assert api.max_batch_size == 4

    def test_batch_size_8(self):
        api = BatchedAPI(max_batch_size=8)
        api.pre_setup()
        api.setup("cpu")

        assert api.max_batch_size == 8


class TestBatchTimeoutConfiguration:
    """Tests for batch timeout settings."""

    def test_zero_timeout_immediate_processing(self):
        api = BatchedAPI(max_batch_size=4, batch_timeout=0.0)
        assert api.batch_timeout == 0.0

    def test_nonzero_timeout(self):
        api = BatchedAPI(max_batch_size=4, batch_timeout=0.1)
        assert api.batch_timeout == 0.1

    def test_invalid_negative_timeout(self):
        with pytest.raises(ValueError, match="batch_timeout must be greater than or equal to 0"):
            BatchedAPI(max_batch_size=4, batch_timeout=-1.0)


class TestServerBatchingEndpoint:
    """Tests for server with batching enabled."""

    def test_server_with_batched_api(self):
        api = BatchedAPI(max_batch_size=4, batch_timeout=0.1)
        server = InferenceServer(api, accelerator="cpu")

        assert server.api.max_batch_size == 4
        assert server.api.batch_timeout == 0.1

    def test_predict_endpoint_exists_with_batching(self):
        api = BatchedAPI(max_batch_size=4)
        server = InferenceServer(api, accelerator="cpu")

        routes = [route.path for route in server.app.routes]
        assert "/predict" in routes
