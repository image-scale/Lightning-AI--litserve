"""Callback system for server lifecycle events."""
import logging
from abc import ABC
from typing import Union

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base class for callbacks.

    Subclass this to implement custom callbacks for server lifecycle events.
    Override any methods you want to hook into.
    """

    def on_before_setup(self, *args, **kwargs):
        """Called before setup is started."""
        pass

    def on_after_setup(self, *args, **kwargs):
        """Called after setup is completed."""
        pass

    def on_before_decode_request(self, *args, **kwargs):
        """Called before request decoding."""
        pass

    def on_after_decode_request(self, *args, **kwargs):
        """Called after request decoding."""
        pass

    def on_before_encode_response(self, *args, **kwargs):
        """Called before response encoding."""
        pass

    def on_after_encode_response(self, *args, **kwargs):
        """Called after response encoding."""
        pass

    def on_before_predict(self, *args, **kwargs):
        """Called before prediction."""
        pass

    def on_after_predict(self, *args, **kwargs):
        """Called after prediction."""
        pass

    def on_server_start(self, *args, **kwargs):
        """Called when server starts."""
        pass

    def on_server_end(self, *args, **kwargs):
        """Called when server shuts down."""
        pass


class CallbackRunner:
    """Manages and triggers callbacks.

    Args:
        callbacks: A single Callback or list of Callbacks to manage.
    """

    def __init__(self, callbacks: Union[Callback, list[Callback], None] = None):
        self._callbacks: list[Callback] = []
        if callbacks:
            self._add_callbacks(callbacks)

    def _add_callbacks(self, callbacks: Union[Callback, list[Callback]]):
        """Add callbacks to the runner."""
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        for callback in callbacks:
            if not isinstance(callback, Callback):
                raise ValueError(f"Invalid callback type: {callback}")
        self._callbacks.extend(callbacks)

    def trigger_event(self, event_name: str, *args, **kwargs):
        """Trigger an event on all registered callbacks.

        Args:
            event_name: The name of the callback method to call.
            *args: Positional arguments to pass to the callback.
            **kwargs: Keyword arguments to pass to the callback.
        """
        for callback in self._callbacks:
            try:
                getattr(callback, event_name)(*args, **kwargs)
            except Exception:
                logger.exception(
                    f"Error in callback '{callback}' during event '{event_name}'"
                )
