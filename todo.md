# Todo

## Plan
Implement the serving library in dependency order: Start with the core API class that users extend to define their models, then build the server that runs inference workers. Add batching and streaming capabilities, followed by device/hardware detection. Include health endpoints and callback system for production deployments.

## Tasks
- [x] Task 1: Implement the core inference API class that users extend to define model setup, request decoding, prediction, and response encoding, along with batching support (src/inferenceapi.py + tests/test_api.py)
- [x] Task 2: Implement the inference server that accepts the API class and runs worker processes, handling request queuing via multiprocessing, FastAPI endpoints, and response buffering (src/server.py + tests/test_server.py)
- [x] Task 3: Add request batching with configurable batch size and timeout, allowing multiple requests to be processed together for better throughput (src/batching.py + tests/test_batching.py)
- [x] Task 4: Add streaming response support so inference APIs can yield output incrementally for real-time use cases like LLM token generation (src/streaming.py + tests/test_streaming.py)
- [x] Task 5: Add device/accelerator detection to automatically discover and configure CPU, CUDA GPUs, or Apple MPS devices (src/devices.py + tests/test_devices.py)
- [x] Task 6: Add callback system for server lifecycle events like before/after setup, before/after predict, and server start/stop (src/callbacks.py + tests/test_callbacks.py)
