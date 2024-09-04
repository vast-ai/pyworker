This is the base PyWorker for TGI, designed to create PyWorkers that can utilize various LLMs. It offers two primary endpoints:

1. `generate`: Generates the LLM's response to a given prompt in a single request.
2. `generate_stream`: Streams the LLM's response token by token.

Both endpoints use the following API payload format:

```json
{
  "inputs": "PROMPT",
  "parameters": {
    "max_new_tokens": 250
  }
}
```

Note that the max_new_tokens parameter, rather than the prompt size, impacts performance. For example, if an
instance is benchmarked to process 100 tokens per second, a request with max_new_tokens = 200 will take
approximately 2 seconds to complete.
