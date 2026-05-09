"""AI Server - High-performance AI model serving library."""

from aiserver.api import InferenceAPI
from aiserver.server import InferenceServer

__version__ = "0.1.0"

__all__ = [
    "InferenceAPI",
    "InferenceServer",
]
