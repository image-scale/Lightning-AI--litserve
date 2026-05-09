"""Core inference API base class for defining model inference logic."""
import json
import warnings
from abc import ABC
from typing import Any, Optional

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False


class InferenceAPI(ABC):
    """Base class for defining AI model inference logic.

    Users subclass this and implement at minimum:
    - setup(device): Load and initialize the model
    - predict(x): Run model inference on input

    Optional overrides:
    - decode_request(request): Transform HTTP request to model input
    - encode_response(output): Transform model output to HTTP response
    - batch(inputs): Combine multiple inputs for batched inference
    - unbatch(output): Split batched output into individual results
    """

    _stream: bool = False
    _device: Optional[str] = None
    _callback_runner: Optional[Any] = None
    request_timeout: Optional[float] = None

    def __init__(
        self,
        max_batch_size: int = 1,
        batch_timeout: float = 0.0,
        api_path: str = "/predict",
        stream: bool = False,
    ):
        """Initialize the inference API with configuration options.

        Args:
            max_batch_size: Maximum number of requests to batch together.
                Must be > 0. Defaults to 1.
            batch_timeout: Maximum time (seconds) to wait for batch to fill.
                Must be >= 0. Defaults to 0.0.
            api_path: URL endpoint path. Must start with "/". Defaults to "/predict".
            stream: Enable streaming responses. Defaults to False.
        """
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than 0")

        if batch_timeout < 0:
            raise ValueError("batch_timeout must be greater than or equal to 0")

        if not api_path.startswith("/"):
            raise ValueError(
                "api_path must start with '/'. "
                "Please provide a valid api path like '/predict', '/classify', or '/v1/predict'"
            )

        self._api_path = api_path
        self._stream = stream
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self._default_unbatch = None

    def setup(self, device: str) -> None:
        """Initialize the model and resources.

        Override this method to load your model and any resources needed.
        This is called once per worker process.

        Args:
            device: Device identifier (e.g., "cpu", "cuda:0", "mps")
        """
        pass

    def decode_request(self, request: Any, **kwargs) -> Any:
        """Transform HTTP request to model input format.

        Override to customize how incoming requests are transformed
        before being passed to predict().

        Args:
            request: The incoming request data (usually dict from JSON)

        Returns:
            Transformed input ready for predict()
        """
        return request

    def predict(self, x: Any, **kwargs) -> Any:
        """Run model inference on the input.

        This is the main inference method that must be implemented.

        Args:
            x: Input data (output from decode_request or batch)

        Returns:
            Model prediction output

        Raises:
            NotImplementedError: If not overridden
        """
        raise NotImplementedError("predict is not implemented")

    def encode_response(self, output: Any, **kwargs) -> Any:
        """Transform model output to HTTP response format.

        Override to customize how model outputs are transformed
        before being returned to the client.

        Args:
            output: Model prediction output

        Returns:
            Transformed output ready for HTTP response
        """
        return output

    def batch(self, inputs: list) -> Any:
        """Combine multiple inputs into a batched input.

        Override to customize batching behavior. Default stacks
        torch tensors and numpy arrays.

        Args:
            inputs: List of individual inputs

        Returns:
            Batched input
        """
        if not inputs:
            return inputs

        first = inputs[0]

        if hasattr(first, "__torch_function__"):
            import torch
            return torch.stack(inputs)

        if first.__class__.__name__ == "ndarray":
            import numpy
            return numpy.stack(inputs)

        return inputs

    def _unbatch_no_stream(self, output: Any) -> list:
        """Default unbatch for non-streaming mode."""
        if isinstance(output, str):
            warnings.warn(
                "The 'predict' method returned a string instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly.",
                UserWarning,
            )
        elif isinstance(output, dict):
            warnings.warn(
                "The 'predict' method returned a dict instead of a list of predictions. "
                "When batching is enabled, 'predict' must return a list to handle multiple inputs correctly.",
                UserWarning,
            )
        return list(output)

    def _unbatch_stream(self, output_stream):
        """Default unbatch for streaming mode."""
        for output in output_stream:
            yield list(output)

    def unbatch(self, output: Any) -> list:
        """Split batched output into individual results.

        Override to customize unbatching behavior.

        Args:
            output: Batched model output

        Returns:
            List of individual outputs
        """
        if self._default_unbatch is None:
            raise ValueError(
                "Default implementation for unbatch method was not found. "
                "Please implement the unbatch method."
            )
        return self._default_unbatch(output)

    def format_encoded_response(self, data: Any) -> Any:
        """Format encoded response for transmission.

        Converts dicts to JSON strings with newline, Pydantic models to JSON,
        and returns other types unchanged.

        Args:
            data: Encoded response data

        Returns:
            Formatted response ready for transmission
        """
        if isinstance(data, dict):
            return json.dumps(data) + "\n"
        if HAS_PYDANTIC and isinstance(data, BaseModel):
            return data.model_dump_json() + "\n"
        return data

    def pre_setup(self) -> None:
        """Called before setup to configure internal state."""
        if self._stream:
            self._default_unbatch = self._unbatch_stream
        else:
            self._default_unbatch = self._unbatch_no_stream

    @property
    def stream(self) -> bool:
        """Whether streaming mode is enabled."""
        return self._stream

    @stream.setter
    def stream(self, value: bool) -> None:
        self._stream = value

    @property
    def device(self) -> Optional[str]:
        """The device this API is running on."""
        return self._device

    @device.setter
    def device(self, value: str) -> None:
        self._device = value

    @property
    def api_path(self) -> str:
        """The URL endpoint path for this API."""
        return self._api_path

    @api_path.setter
    def api_path(self, value: str) -> None:
        self._api_path = value

    def set_callback_runner(self, runner) -> None:
        """Set the callback runner for this API.

        Args:
            runner: A CallbackRunner instance to use for callbacks.
        """
        self._callback_runner = runner

    def predict_with_callbacks(self, x: Any, **kwargs) -> Any:
        """Run predict with callbacks triggered before and after.

        Args:
            x: Input data (output from decode_request or batch)

        Returns:
            Model prediction output
        """
        if self._callback_runner:
            self._callback_runner.trigger_event("on_before_predict", x)

        result = self.predict(x, **kwargs)

        if self._callback_runner:
            self._callback_runner.trigger_event("on_after_predict", result)

        return result
