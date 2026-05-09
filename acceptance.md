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

## Task 2: Inference Server

### Acceptance Criteria
- [ ] Server can be initialized with an InferenceAPI instance
- [ ] Server accepts accelerator parameter ("cpu", "cuda", "auto") and devices parameter (int or "auto")
- [ ] Server accepts timeout parameter for request timeout (default 30 seconds)
- [ ] Server accepts workers_per_device parameter (default 1) for parallel workers
- [ ] Server has run() method that starts the server on a specified host and port
- [ ] Server creates /health endpoint returning 200 when workers are ready, 503 otherwise
- [ ] Server creates /info endpoint returning server configuration as JSON
- [ ] Server creates dynamic prediction endpoint based on API's api_path
- [ ] POST to prediction endpoint returns the inference result from the API
- [ ] Server spawns worker processes that call API's setup and process requests
- [ ] Server validates api_path and other configuration at startup
- [ ] Server handles request timeouts and returns 504 when exceeded
