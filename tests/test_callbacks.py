"""Tests for callback system."""
import pytest
from unittest.mock import MagicMock, patch, call

from aiserver.callbacks import Callback, CallbackRunner


class TestCallback:
    """Tests for Callback base class."""

    def test_callback_has_before_setup(self):
        cb = Callback()
        assert hasattr(cb, "on_before_setup")
        assert callable(cb.on_before_setup)

    def test_callback_has_after_setup(self):
        cb = Callback()
        assert hasattr(cb, "on_after_setup")
        assert callable(cb.on_after_setup)

    def test_callback_has_before_predict(self):
        cb = Callback()
        assert hasattr(cb, "on_before_predict")
        assert callable(cb.on_before_predict)

    def test_callback_has_after_predict(self):
        cb = Callback()
        assert hasattr(cb, "on_after_predict")
        assert callable(cb.on_after_predict)

    def test_callback_has_before_decode_request(self):
        cb = Callback()
        assert hasattr(cb, "on_before_decode_request")
        assert callable(cb.on_before_decode_request)

    def test_callback_has_after_decode_request(self):
        cb = Callback()
        assert hasattr(cb, "on_after_decode_request")
        assert callable(cb.on_after_decode_request)

    def test_callback_has_before_encode_response(self):
        cb = Callback()
        assert hasattr(cb, "on_before_encode_response")
        assert callable(cb.on_before_encode_response)

    def test_callback_has_after_encode_response(self):
        cb = Callback()
        assert hasattr(cb, "on_after_encode_response")
        assert callable(cb.on_after_encode_response)

    def test_callback_has_server_start(self):
        cb = Callback()
        assert hasattr(cb, "on_server_start")
        assert callable(cb.on_server_start)

    def test_callback_has_server_end(self):
        cb = Callback()
        assert hasattr(cb, "on_server_end")
        assert callable(cb.on_server_end)

    def test_default_methods_do_nothing(self):
        cb = Callback()
        cb.on_before_setup()
        cb.on_after_setup()
        cb.on_before_predict()
        cb.on_after_predict()
        cb.on_server_start()
        cb.on_server_end()


class TestCallbackRunner:
    """Tests for CallbackRunner."""

    def test_init_empty(self):
        runner = CallbackRunner()
        assert runner._callbacks == []

    def test_init_with_single_callback(self):
        cb = Callback()
        runner = CallbackRunner(callbacks=cb)
        assert len(runner._callbacks) == 1

    def test_init_with_callback_list(self):
        cb1 = Callback()
        cb2 = Callback()
        runner = CallbackRunner(callbacks=[cb1, cb2])
        assert len(runner._callbacks) == 2

    def test_invalid_callback_raises_error(self):
        with pytest.raises(ValueError, match="Invalid callback type"):
            CallbackRunner(callbacks="not a callback")

    def test_trigger_event_calls_all_callbacks(self):
        cb1 = MagicMock(spec=Callback)
        cb2 = MagicMock(spec=Callback)
        runner = CallbackRunner(callbacks=[cb1, cb2])

        runner.trigger_event("on_before_setup")

        cb1.on_before_setup.assert_called_once()
        cb2.on_before_setup.assert_called_once()

    def test_trigger_event_passes_args(self):
        cb = MagicMock(spec=Callback)
        runner = CallbackRunner(callbacks=cb)

        runner.trigger_event("on_before_predict", "arg1", key="value")

        cb.on_before_predict.assert_called_once_with("arg1", key="value")

    def test_error_in_callback_does_not_stop_others(self):
        cb1 = MagicMock(spec=Callback)
        cb1.on_before_setup.side_effect = RuntimeError("test error")
        cb2 = MagicMock(spec=Callback)

        runner = CallbackRunner(callbacks=[cb1, cb2])
        runner.trigger_event("on_before_setup")

        cb2.on_before_setup.assert_called_once()


class TestCustomCallback:
    """Tests for custom callback implementations."""

    def test_custom_callback_inherits(self):
        class MyCallback(Callback):
            def on_before_predict(self, x):
                self.value = x * 2

        cb = MyCallback()
        cb.on_before_predict(5)
        assert cb.value == 10

    def test_multiple_custom_callbacks(self):
        class LoggingCallback(Callback):
            def __init__(self):
                self.events = []

            def on_before_setup(self):
                self.events.append("before_setup")

            def on_after_setup(self):
                self.events.append("after_setup")

        cb = LoggingCallback()
        runner = CallbackRunner(callbacks=cb)

        runner.trigger_event("on_before_setup")
        runner.trigger_event("on_after_setup")

        assert cb.events == ["before_setup", "after_setup"]


class TestServerCallbackIntegration:
    """Tests for server callback integration."""

    def test_server_accepts_callbacks_parameter(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        cb = Callback()
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", callbacks=cb)
        assert server._callback_runner is not None

    def test_server_accepts_callback_list(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        cb1 = Callback()
        cb2 = Callback()
        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu", callbacks=[cb1, cb2])
        assert len(server._callback_runner._callbacks) == 2

    def test_server_without_callbacks(self):
        from aiserver import InferenceAPI, InferenceServer

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        api = SimpleAPI()
        server = InferenceServer(api, accelerator="cpu")
        assert server._callback_runner is not None
        assert len(server._callback_runner._callbacks) == 0


class TestAPICallbackIntegration:
    """Tests for API callback integration."""

    def test_api_accepts_callback_runner(self):
        from aiserver import InferenceAPI

        class SimpleAPI(InferenceAPI):
            def predict(self, x):
                return x

        api = SimpleAPI()
        runner = CallbackRunner()
        api.set_callback_runner(runner)
        assert api._callback_runner is runner

    def test_api_triggers_callbacks_on_predict(self):
        from aiserver import InferenceAPI

        class SimpleAPI(InferenceAPI):
            def setup(self, device):
                self.device = device

            def predict(self, x):
                return x * 2

        cb = MagicMock(spec=Callback)
        runner = CallbackRunner(callbacks=cb)

        api = SimpleAPI()
        api.pre_setup()
        api.setup("cpu")
        api.set_callback_runner(runner)

        api.predict_with_callbacks(5)

        cb.on_before_predict.assert_called()
        cb.on_after_predict.assert_called()
