"""Tests for streaming response support."""
import inspect
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from aiserver import InferenceAPI, InferenceServer


class StreamingAPI(InferenceAPI):
    """API that yields streaming responses."""

    def setup(self, device):
        self.device = device
        self.sentence = "Hello world this is streaming"

    def decode_request(self, request):
        return request.get("prompt", "")

    def predict(self, x):
        words = f"prompt={x} output={self.sentence}".split()
        for word in words:
            yield word

    def encode_response(self, output):
        return {"token": output}


class NonStreamingAPI(InferenceAPI):
    """Regular non-streaming API."""

    def setup(self, device):
        self.device = device

    def predict(self, x):
        return x * 2


class TestStreamingConfiguration:
    """Tests for streaming configuration."""

    def test_default_stream_false(self):
        api = NonStreamingAPI()
        assert api.stream is False

    def test_stream_enabled(self):
        api = StreamingAPI(stream=True)
        assert api.stream is True

    def test_stream_property_setter(self):
        api = NonStreamingAPI()
        api.stream = True
        assert api.stream is True


class TestStreamingAPIBehavior:
    """Tests for streaming API behavior."""

    def test_streaming_predict_is_generator(self):
        api = StreamingAPI(stream=True)
        api.pre_setup()
        api.setup("cpu")

        result = api.predict("test")
        assert hasattr(result, '__iter__')
        assert hasattr(result, '__next__')

    def test_streaming_predict_yields_values(self):
        api = StreamingAPI(stream=True)
        api.pre_setup()
        api.setup("cpu")

        results = list(api.predict("hi"))
        assert len(results) > 1
        assert "prompt=hi" in results

    def test_streaming_encode_response(self):
        api = StreamingAPI(stream=True)
        api.pre_setup()
        api.setup("cpu")

        encoded = api.encode_response("word")
        assert encoded == {"token": "word"}


class TestStreamingInfoEndpoint:
    """Tests for stream status in info endpoint."""

    def test_info_shows_stream_enabled(self):
        api = StreamingAPI(stream=True)
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["stream"] is True

    def test_info_shows_stream_disabled(self):
        api = NonStreamingAPI()
        server = InferenceServer(api, accelerator="cpu")
        client = TestClient(server.app, raise_server_exceptions=False)

        info = client.get("/info").json()
        assert info["server"]["stream"] is False


class TestStreamingServerSetup:
    """Tests for server setup with streaming."""

    def test_server_with_streaming_api(self):
        api = StreamingAPI(stream=True)
        server = InferenceServer(api, accelerator="cpu")
        assert server.api.stream is True

    def test_predict_endpoint_registered_for_streaming(self):
        api = StreamingAPI(stream=True)
        server = InferenceServer(api, accelerator="cpu")

        routes = [route.path for route in server.app.routes]
        assert "/predict" in routes


class TestNonStreamingStillWorks:
    """Tests that non-streaming APIs continue to work."""

    def test_non_streaming_api_works(self):
        api = NonStreamingAPI()
        api.pre_setup()
        api.setup("cpu")

        result = api.predict(5)
        assert result == 10

    def test_server_with_non_streaming_api(self):
        api = NonStreamingAPI()
        server = InferenceServer(api, accelerator="cpu")
        assert server.api.stream is False


class TestStreamingUnbatch:
    """Tests for streaming unbatch behavior."""

    def test_streaming_unbatch_is_generator(self):
        api = StreamingAPI(stream=True)
        api.pre_setup()

        def generate():
            yield [1, 2, 3]
            yield [4, 5, 6]

        result = api.unbatch(generate())
        assert inspect.isgenerator(result)

    def test_streaming_unbatch_yields_correctly(self):
        api = StreamingAPI(stream=True)
        api.pre_setup()

        def generate():
            yield [1, 2, 3]
            yield [4, 5, 6]

        results = list(api.unbatch(generate()))
        assert results == [[1, 2, 3], [4, 5, 6]]
