# Acceptance Criteria

## Task 1: Core Inference API Class (COMPLETE)

### Acceptance Criteria
- [x] Users can subclass InferenceAPI and override setup(device) to initialize their model
- [x] Users can override predict(x) to define inference logic; returns prediction output
- [x] Users can override decode_request(request) to transform HTTP request to model input; defaults to returning request as-is
- [x] Users can override encode_response(output) to transform model output to HTTP response; defaults to returning output as-is
- [x] Users can override batch(inputs) to combine multiple inputs; default stacks torch tensors and numpy arrays
- [x] Users can override unbatch(output) to split batched output; default converts to list
- [x] API supports max_batch_size parameter (default 1) for configuring batching
- [x] API supports batch_timeout parameter (default 0.0) for configuring how long to wait for batch to fill
- [x] API supports stream parameter (default False) for streaming mode
- [x] API supports api_path parameter (default "/predict") for endpoint path
- [x] API validates that max_batch_size > 0 and batch_timeout >= 0
- [x] API validates that api_path starts with "/"
- [x] API has device property that can be get/set
- [x] format_encoded_response() formats dict as JSON string with newline, Pydantic model as JSON, others unchanged

## Task 2: Inference Server (COMPLETE)

### Acceptance Criteria
- [x] Server can be initialized with an InferenceAPI instance
- [x] Server accepts accelerator parameter ("cpu", "cuda", "auto") and devices parameter (int or "auto")
- [x] Server accepts timeout parameter for request timeout (default 30 seconds)
- [x] Server accepts workers_per_device parameter (default 1) for parallel workers
- [x] Server has run() method that starts the server on a specified host and port
- [x] Server creates /health endpoint returning 200 when workers are ready, 503 otherwise
- [x] Server creates /info endpoint returning server configuration as JSON
- [x] Server creates dynamic prediction endpoint based on API's api_path
- [x] POST to prediction endpoint returns the inference result from the API
- [x] Server spawns worker processes that call API's setup and process requests
- [x] Server validates api_path and other configuration at startup
- [x] Server handles request timeouts and returns 504 when exceeded

## Task 3: Request Batching (COMPLETE)

### Acceptance Criteria
- [x] When max_batch_size > 1, workers collect multiple requests before processing
- [x] batch_timeout controls how long to wait for more requests before processing partial batch
- [x] Batch of requests decoded individually then combined via batch() method
- [x] Prediction runs once on batched input
- [x] Results split via unbatch() and sent to respective clients
- [x] If batch_timeout is 0, batch processes immediately with whatever requests are available
- [x] Timed out requests in batch are individually marked as 504 errors
- [x] Batching works correctly with different batch sizes (2, 4, 8)

## Task 4: Streaming Response Support

### Acceptance Criteria
- [ ] When stream=True, predict() can be a generator that yields values
- [ ] encode_response() can also be a generator yielding encoded chunks
- [ ] Server returns StreamingResponse instead of regular JSON response
- [ ] Each yielded value is sent to client as it becomes available
- [ ] Client receives data incrementally without waiting for full response
- [ ] Streaming works with the /info endpoint reporting stream status
- [ ] Non-streaming APIs continue to work normally
- [ ] API validates that predict/encode_response are generators when stream=True
