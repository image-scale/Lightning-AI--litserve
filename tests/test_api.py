"""Tests for the InferenceAPI base class."""
import json
import pytest
import numpy as np
import torch

from aiserver import InferenceAPI


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


class TestInferenceAPIInit:
    """Tests for InferenceAPI initialization and validation."""

    def test_default_parameters(self):
        api = SimpleAPI()
        assert api.max_batch_size == 1
        assert api.batch_timeout == 0.0
        assert api.api_path == "/predict"
        assert api.stream is False

    def test_custom_parameters(self):
        api = SimpleAPI(
            max_batch_size=8,
            batch_timeout=0.5,
            api_path="/inference",
            stream=True,
        )
        assert api.max_batch_size == 8
        assert api.batch_timeout == 0.5
        assert api.api_path == "/inference"
        assert api.stream is True

    def test_invalid_batch_size_zero(self):
        with pytest.raises(ValueError, match="max_batch_size must be greater than 0"):
            SimpleAPI(max_batch_size=0)

    def test_invalid_batch_size_negative(self):
        with pytest.raises(ValueError, match="max_batch_size must be greater than 0"):
            SimpleAPI(max_batch_size=-5)

    def test_invalid_batch_timeout_negative(self):
        with pytest.raises(ValueError, match="batch_timeout must be greater than or equal to 0"):
            SimpleAPI(batch_timeout=-1.0)

    def test_invalid_api_path_no_slash(self):
        with pytest.raises(ValueError, match="api_path must start with '/'"):
            SimpleAPI(api_path="predict")


class TestInferenceAPIMethods:
    """Tests for InferenceAPI core methods."""

    def test_setup_called(self):
        api = SimpleAPI()
        api.setup("cpu")
        assert api.device == "cpu"
        assert api.model(2) == 4

    def test_decode_request(self):
        api = SimpleAPI()
        request = {"input": 5.0}
        result = api.decode_request(request)
        assert result == 5.0

    def test_predict(self):
        api = SimpleAPI()
        api.setup("cpu")
        result = api.predict(3)
        assert result == 9

    def test_predict_not_implemented(self):
        api = InferenceAPI()
        with pytest.raises(NotImplementedError, match="predict is not implemented"):
            api.predict(1)

    def test_encode_response(self):
        api = SimpleAPI()
        result = api.encode_response(16)
        assert result == {"output": 16}

    def test_default_decode_request_passthrough(self):
        api = InferenceAPI()
        data = {"key": "value"}
        assert api.decode_request(data) == data

    def test_default_encode_response_passthrough(self):
        api = InferenceAPI()
        data = {"result": 42}
        assert api.encode_response(data) == data


class TestInferenceAPIBatching:
    """Tests for batching functionality."""

    def test_batch_list_passthrough(self):
        api = SimpleAPI()
        inputs = [1, 2, 3, 4]
        result = api.batch(inputs)
        assert result == inputs

    def test_batch_torch_tensors(self):
        api = SimpleAPI()
        tensors = [torch.tensor([1, 2]), torch.tensor([3, 4])]
        result = api.batch(tensors)
        expected = torch.stack(tensors)
        assert torch.all(result == expected)

    def test_batch_numpy_arrays(self):
        api = SimpleAPI()
        arrays = [np.array([1, 2]), np.array([3, 4])]
        result = api.batch(arrays)
        expected = np.stack(arrays)
        assert np.all(result == expected)

    def test_batch_empty_list(self):
        api = SimpleAPI()
        result = api.batch([])
        assert result == []

    def test_unbatch_not_configured_raises_error(self):
        api = SimpleAPI()
        with pytest.raises(ValueError, match="Default implementation for unbatch method was not found"):
            api.unbatch([1, 2, 3])

    def test_unbatch_after_pre_setup(self):
        api = SimpleAPI(max_batch_size=4)
        api.pre_setup()
        result = api.unbatch([1, 2, 3, 4])
        assert result == [1, 2, 3, 4]

    def test_unbatch_warns_on_string(self):
        api = SimpleAPI(max_batch_size=4)
        api.pre_setup()
        with pytest.warns(UserWarning, match="returned a string instead of a list"):
            api.unbatch("hello")

    def test_unbatch_warns_on_dict(self):
        api = SimpleAPI(max_batch_size=4)
        api.pre_setup()
        with pytest.warns(UserWarning, match="returned a dict instead of a list"):
            api.unbatch({"a": 1, "b": 2})


class TestInferenceAPIStreaming:
    """Tests for streaming functionality."""

    def test_stream_property_getter_setter(self):
        api = SimpleAPI()
        assert api.stream is False
        api.stream = True
        assert api.stream is True

    def test_streaming_unbatch(self):
        api = SimpleAPI(stream=True)
        api.pre_setup()

        def generate_outputs():
            yield [1, 2, 3]
            yield [4, 5, 6]

        result = list(api.unbatch(generate_outputs()))
        assert result == [[1, 2, 3], [4, 5, 6]]


class TestInferenceAPIFormatResponse:
    """Tests for format_encoded_response method."""

    def test_format_dict_as_json(self):
        api = SimpleAPI()
        data = {"output": 4.0}
        result = api.format_encoded_response(data)
        assert result == '{"output": 4.0}\n'

    def test_format_dict_with_nested_data(self):
        api = SimpleAPI()
        data = {"results": [1, 2, 3], "meta": {"count": 3}}
        result = api.format_encoded_response(data)
        parsed = json.loads(result.strip())
        assert parsed == data

    def test_format_non_dict_passthrough(self):
        api = SimpleAPI()
        data = [1, 2, 3, 4]
        result = api.format_encoded_response(data)
        assert result == data

    def test_format_string_passthrough(self):
        api = SimpleAPI()
        data = "hello world"
        result = api.format_encoded_response(data)
        assert result == data


class TestInferenceAPIProperties:
    """Tests for API properties."""

    def test_device_property(self):
        api = SimpleAPI()
        assert api.device is None
        api.device = "cuda:0"
        assert api.device == "cuda:0"

    def test_api_path_property(self):
        api = SimpleAPI()
        assert api.api_path == "/predict"
        api.api_path = "/classify"
        assert api.api_path == "/classify"


class TestInferenceAPIPreSetup:
    """Tests for pre_setup configuration."""

    def test_pre_setup_non_streaming(self):
        api = SimpleAPI(stream=False)
        api.pre_setup()
        assert api._default_unbatch == api._unbatch_no_stream

    def test_pre_setup_streaming(self):
        api = SimpleAPI(stream=True)
        api.pre_setup()
        assert api._default_unbatch == api._unbatch_stream


class TestInferenceAPIEndToEnd:
    """End-to-end tests for the full inference flow."""

    def test_simple_inference_flow(self):
        api = SimpleAPI()
        api.pre_setup()
        api.setup("cpu")

        request = {"input": 4.0}
        decoded = api.decode_request(request)
        prediction = api.predict(decoded)
        response = api.encode_response(prediction)

        assert response == {"output": 16.0}

    def test_batched_inference_flow(self):
        api = SimpleAPI(max_batch_size=4)
        api.pre_setup()
        api.setup("cpu")

        requests = [{"input": 1}, {"input": 2}, {"input": 3}]
        decoded = [api.decode_request(r) for r in requests]

        batched = api.batch(decoded)
        assert batched == [1, 2, 3]

        predictions = [api.predict(x) for x in batched]

        unbatched = api.unbatch(predictions)
        assert unbatched == [1, 4, 9]

        responses = [api.encode_response(p) for p in unbatched]
        assert responses == [{"output": 1}, {"output": 4}, {"output": 9}]
