"""Inference server for hosting AI model APIs."""
import asyncio
import copy
import logging
import multiprocessing as mp
import os
import pickle
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any, Literal, Optional, Union

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from aiserver.api import InferenceAPI
from aiserver.devices import detect_accelerator, detect_device_count, get_device_identifiers

logger = logging.getLogger(__name__)

mp.allow_connection_pickling()


class APIStatus:
    """Status codes for API responses."""
    OK = "OK"
    ERROR = "ERROR"
    FINISH_STREAMING = "FINISH_STREAMING"


class WorkerStatus:
    """Status codes for worker setup."""
    STARTING = "starting"
    READY = "ready"
    ERROR = "error"


@dataclass
class ResponseItem:
    """Buffer item for holding response data."""
    event: asyncio.Event
    response: Any = None
    response_queue: Optional[deque] = None


def _inference_loop(
    api: InferenceAPI,
    device: str,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    worker_status: dict,
    request_timeout: float,
):
    """Main inference worker loop running in a separate process."""
    try:
        api.pre_setup()
        api.setup(device)
        api.device = device
        worker_status[worker_id] = WorkerStatus.READY
    except Exception as e:
        logger.exception(f"Error setting up worker {worker_id}: {e}")
        worker_status[worker_id] = WorkerStatus.ERROR
        return

    if api.stream:
        _streaming_inference_loop(api, worker_id, request_queue, response_queue, request_timeout)
    elif api.max_batch_size > 1:
        _batched_inference_loop(api, worker_id, request_queue, response_queue, request_timeout)
    else:
        _single_inference_loop(api, worker_id, request_queue, response_queue, request_timeout)


def _single_inference_loop(
    api: InferenceAPI,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    request_timeout: float,
):
    """Single request inference loop."""
    while True:
        try:
            request_data = request_queue.get(timeout=1.0)
            if request_data is None:
                logger.debug(f"Worker {worker_id} received shutdown signal")
                return

            uid, timestamp, payload = request_data
        except Empty:
            continue
        except Exception:
            continue

        if request_timeout and request_timeout > 0:
            elapsed = time.monotonic() - timestamp
            if elapsed > request_timeout:
                logger.warning(f"Request {uid} timed out after {elapsed:.2f}s")
                response_queue.put((uid, (HTTPException(504, "Request timed out"), APIStatus.ERROR)))
                continue

        try:
            x = api.decode_request(payload)
            y = api.predict(x)
            y_enc = api.encode_response(y)
            response_queue.put((uid, (y_enc, APIStatus.OK)))
        except HTTPException as e:
            response_queue.put((uid, (e, APIStatus.ERROR)))
        except Exception as e:
            logger.exception(f"Error processing request {uid}: {e}")
            response_queue.put((uid, (pickle.dumps(e), APIStatus.ERROR)))


def _collect_batch(
    api: InferenceAPI,
    request_queue: Queue,
    request_timeout: float,
) -> tuple[list, list]:
    """Collect a batch of requests from the queue.

    Returns:
        Tuple of (valid_requests, timed_out_uids) where valid_requests is a list
        of (uid, payload) tuples and timed_out_uids is a list of (uid,) tuples.
    """
    batch = []
    timed_out = []
    start_time = time.monotonic()
    end_time = start_time + api.batch_timeout

    while len(batch) < api.max_batch_size:
        remaining_time = end_time - time.monotonic() if api.batch_timeout > 0 else 0

        if api.batch_timeout > 0 and remaining_time <= 0 and len(batch) > 0:
            break

        try:
            timeout = min(remaining_time, 0.01) if api.batch_timeout > 0 else 0.01
            request_data = request_queue.get(timeout=timeout)

            if request_data is None:
                return None, None

            uid, timestamp, payload = request_data

            if request_timeout and request_timeout > 0:
                if time.monotonic() - timestamp > request_timeout:
                    timed_out.append(uid)
                    continue

            batch.append((uid, payload))

        except Empty:
            if api.batch_timeout == 0 and len(batch) > 0:
                break
            if api.batch_timeout > 0 and time.monotonic() >= end_time:
                break
            continue

    return batch, timed_out


def _batched_inference_loop(
    api: InferenceAPI,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    request_timeout: float,
):
    """Batched inference loop that processes multiple requests together."""
    while True:
        batch, timed_out = _collect_batch(api, request_queue, request_timeout)

        if batch is None:
            logger.debug(f"Worker {worker_id} received shutdown signal")
            return

        for uid in timed_out:
            logger.warning(f"Request {uid} timed out in batch collection")
            response_queue.put((uid, (HTTPException(504, "Request timed out"), APIStatus.ERROR)))

        if not batch:
            continue

        uids = [item[0] for item in batch]
        payloads = [item[1] for item in batch]

        try:
            decoded = [api.decode_request(p) for p in payloads]
            batched_input = api.batch(decoded)
            batched_output = api.predict(batched_input)
            unbatched = api.unbatch(batched_output)

            if len(unbatched) != len(uids):
                logger.error(
                    f"Batch size mismatch: got {len(unbatched)} outputs for {len(uids)} inputs"
                )
                for uid in uids:
                    response_queue.put((uid, (HTTPException(500, "Batch size mismatch"), APIStatus.ERROR)))
                continue

            for uid, output in zip(uids, unbatched):
                y_enc = api.encode_response(output)
                response_queue.put((uid, (y_enc, APIStatus.OK)))

        except HTTPException as e:
            for uid in uids:
                response_queue.put((uid, (e, APIStatus.ERROR)))
        except Exception as e:
            logger.exception(f"Error processing batch: {e}")
            for uid in uids:
                response_queue.put((uid, (pickle.dumps(e), APIStatus.ERROR)))


def _streaming_inference_loop(
    api: InferenceAPI,
    worker_id: int,
    request_queue: Queue,
    response_queue: Queue,
    request_timeout: float,
):
    """Streaming inference loop that yields responses incrementally."""
    while True:
        try:
            request_data = request_queue.get(timeout=1.0)
            if request_data is None:
                logger.debug(f"Worker {worker_id} received shutdown signal")
                return

            uid, timestamp, payload = request_data
        except Empty:
            continue
        except Exception:
            continue

        if request_timeout and request_timeout > 0:
            elapsed = time.monotonic() - timestamp
            if elapsed > request_timeout:
                logger.warning(f"Request {uid} timed out after {elapsed:.2f}s")
                response_queue.put((uid, (HTTPException(504, "Request timed out"), APIStatus.ERROR)))
                continue

        try:
            x = api.decode_request(payload)
            output_stream = api.predict(x)

            for output in output_stream:
                y_enc = api.encode_response(output)
                if hasattr(y_enc, '__iter__') and hasattr(y_enc, '__next__'):
                    for chunk in y_enc:
                        formatted = api.format_encoded_response(chunk)
                        response_queue.put((uid, (formatted, APIStatus.OK)))
                else:
                    formatted = api.format_encoded_response(y_enc)
                    response_queue.put((uid, (formatted, APIStatus.OK)))

            response_queue.put((uid, (None, APIStatus.FINISH_STREAMING)))

        except HTTPException as e:
            response_queue.put((uid, (e, APIStatus.ERROR)))
        except Exception as e:
            logger.exception(f"Error processing streaming request {uid}: {e}")
            response_queue.put((uid, (pickle.dumps(e), APIStatus.ERROR)))


async def _response_consumer(
    response_queue: Queue,
    response_buffer: dict,
):
    """Async task that moves responses from queue to buffer."""
    loop = asyncio.get_event_loop()
    while True:
        try:
            result = await loop.run_in_executor(None, response_queue.get, True, 0.1)
            if result is None:
                break
            uid, (response, status) = result
            if uid in response_buffer:
                item = response_buffer[uid]
                if item.response_queue is not None:
                    item.response_queue.append((response, status))
                    item.event.set()
                else:
                    item.response = (response, status)
                    item.event.set()
        except Empty:
            continue
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.debug(f"Error in response consumer: {e}")
            continue


class InferenceServer:
    """Server for hosting InferenceAPI instances with worker processes.

    The server creates worker processes that load the model and process
    inference requests. It exposes FastAPI endpoints for health checks,
    server info, and the main inference endpoint.
    """

    def __init__(
        self,
        api: InferenceAPI,
        accelerator: Literal["cpu", "cuda", "mps", "auto"] = "auto",
        devices: Union[int, Literal["auto"]] = "auto",
        workers_per_device: int = 1,
        timeout: Union[float, bool] = 30,
        model_metadata: Optional[dict] = None,
    ):
        """Initialize the inference server.

        Args:
            api: The InferenceAPI instance defining inference logic
            accelerator: Hardware type ("cpu", "cuda", "mps", "auto")
            devices: Number of devices to use, or "auto" for all available
            workers_per_device: Number of worker processes per device
            timeout: Request timeout in seconds, or False to disable
            model_metadata: Optional metadata about the model
        """
        if workers_per_device < 1:
            raise ValueError("workers_per_device must be >= 1")

        self.api = api
        api.pre_setup()

        self._accelerator = detect_accelerator(accelerator)
        self._devices = detect_device_count(self._accelerator) if devices == "auto" else devices
        self.workers_per_device = workers_per_device
        self.timeout = timeout if timeout is not False else -1
        self.model_metadata = model_metadata

        if timeout not in (False, -1):
            api.request_timeout = timeout

        self.app = FastAPI(lifespan=self._lifespan)
        self.response_buffer: dict[str, ResponseItem] = {}
        self._worker_status: Optional[dict] = None
        self._request_queue: Optional[Queue] = None
        self._response_queue: Optional[Queue] = None
        self._workers: list = []
        self._manager: Optional[mp.Manager] = None
        self._response_task: Optional[asyncio.Task] = None

        self._register_endpoints()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        """Manage server lifecycle."""
        loop = asyncio.get_event_loop()
        self._response_task = loop.create_task(
            _response_consumer(self._response_queue, self.response_buffer)
        )
        try:
            yield
        finally:
            if self._response_task:
                self._response_task.cancel()
                try:
                    await self._response_task
                except asyncio.CancelledError:
                    pass

    def _register_endpoints(self):
        """Register API endpoints."""

        @self.app.get("/")
        async def index():
            return Response(content="aiserver running")

        @self.app.get("/health")
        async def health():
            if not self._worker_status:
                return Response(content="not ready", status_code=503)

            all_ready = all(
                v == WorkerStatus.READY
                for v in self._worker_status.values()
            )
            if all_ready:
                return Response(content="ok", status_code=200)
            return Response(content="not ready", status_code=503)

        @self.app.get("/info")
        async def info():
            return JSONResponse(content={
                "model": self.model_metadata,
                "server": {
                    "accelerator": self._accelerator,
                    "devices": self._devices,
                    "workers_per_device": self.workers_per_device,
                    "timeout": self.timeout,
                    "api_path": self.api.api_path,
                    "max_batch_size": self.api.max_batch_size,
                    "batch_timeout": self.api.batch_timeout,
                    "stream": self.api.stream,
                }
            })

        @self.app.post(self.api.api_path)
        async def predict(request: Request):
            try:
                payload = await request.json()
            except Exception:
                raise HTTPException(400, "Invalid JSON payload")

            uid = str(uuid.uuid4())
            event = asyncio.Event()

            if self.api.stream:
                response_queue = deque()
                self.response_buffer[uid] = ResponseItem(event=event, response_queue=response_queue)
                self._request_queue.put((uid, time.monotonic(), payload))

                async def generate():
                    try:
                        while True:
                            await event.wait()
                            while response_queue:
                                data, status = response_queue.popleft()
                                if status == APIStatus.FINISH_STREAMING:
                                    return
                                if status == APIStatus.ERROR:
                                    return
                                if data is not None:
                                    yield data
                            event.clear()
                    finally:
                        self.response_buffer.pop(uid, None)

                return StreamingResponse(generate())

            self.response_buffer[uid] = ResponseItem(event=event)
            self._request_queue.put((uid, time.monotonic(), payload))

            try:
                await asyncio.wait_for(event.wait(), timeout=self.timeout if self.timeout > 0 else None)
            except asyncio.TimeoutError:
                self.response_buffer.pop(uid, None)
                raise HTTPException(504, "Request timed out")

            item = self.response_buffer.pop(uid)
            response, status = item.response

            if status == APIStatus.ERROR:
                if isinstance(response, HTTPException):
                    raise response
                if isinstance(response, bytes):
                    try:
                        exc = pickle.loads(response)
                        if isinstance(exc, HTTPException):
                            raise exc
                    except Exception:
                        pass
                raise HTTPException(500, "Internal server error")

            return response

    def _start_workers(self):
        """Start inference worker processes."""
        self._manager = mp.Manager()
        self._worker_status = self._manager.dict()
        self._request_queue = self._manager.Queue()
        self._response_queue = self._manager.Queue()

        devices = get_device_identifiers(self._accelerator, self._devices, self.workers_per_device)
        ctx = mp.get_context("spawn")

        for worker_id, device in enumerate(devices):
            self._worker_status[worker_id] = WorkerStatus.STARTING
            api_copy = copy.deepcopy(self.api)
            process = ctx.Process(
                target=_inference_loop,
                args=(
                    api_copy,
                    device,
                    worker_id,
                    self._request_queue,
                    self._response_queue,
                    self._worker_status,
                    self.timeout if self.timeout > 0 else 0,
                ),
                name=f"inference-worker-{worker_id}",
            )
            process.start()
            self._workers.append(process)

    def _wait_for_workers(self):
        """Wait for at least one worker to be ready."""
        while True:
            if any(v == WorkerStatus.READY for v in self._worker_status.values()):
                break
            if any(v == WorkerStatus.ERROR for v in self._worker_status.values()):
                raise RuntimeError("One or more workers failed to start")
            time.sleep(0.05)

    def _stop_workers(self):
        """Stop all worker processes."""
        if self._request_queue:
            for _ in self._workers:
                self._request_queue.put(None)

        for worker in self._workers:
            worker.terminate()
            worker.join(timeout=5)

        if self._manager:
            self._manager.shutdown()

    def run(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        log_level: str = "info",
    ):
        """Start the server.

        Args:
            host: Host address to bind to
            port: Port number to listen on
            log_level: Logging level
        """
        if not 1024 <= port <= 65535:
            raise ValueError(f"port must be between 1024 and 65535, got {port}")

        self._start_workers()

        try:
            self._wait_for_workers()
            logger.info(f"Server starting on {host}:{port}")
            uvicorn.run(self.app, host=host, port=port, log_level=log_level)
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self._stop_workers()
