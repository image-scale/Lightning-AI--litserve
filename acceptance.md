# Acceptance Criteria

## Task 1: Core Inference API Class

### Acceptance Criteria
- [ ] Users can subclass InferenceAPI and override setup(device) to initialize their model
- [ ] Users can override predict(x) to define inference logic; returns prediction output
- [ ] Users can override decode_request(request) to transform HTTP request to model input; defaults to returning request as-is
- [ ] Users can override encode_response(output) to transform model output to HTTP response; defaults to returning output as-is
- [ ] Users can override batch(inputs) to combine multiple inputs; default stacks torch tensors and numpy arrays
- [ ] Users can override unbatch(output) to split batched output; default converts to list
- [ ] API supports max_batch_size parameter (default 1) for configuring batching
- [ ] API supports batch_timeout parameter (default 0.0) for configuring how long to wait for batch to fill
- [ ] API supports stream parameter (default False) for streaming mode
- [ ] API supports api_path parameter (default "/predict") for endpoint path
- [ ] API validates that max_batch_size > 0 and batch_timeout >= 0
- [ ] API validates that api_path starts with "/"
- [ ] API has device property that can be get/set
- [ ] format_encoded_response() formats dict as JSON string with newline, Pydantic model as JSON, others unchanged
