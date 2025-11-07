# ComfyUI PyWorker

This is the base PyWorker for ComfyUI. It provides a unified interface for running any ComfyUI workflow through a proxy-based architecture.

The cost for each request has a static value of `1`.  ComfyUI does not handle concurrent workloads and there is no current provision to load multiple instances of ComfyUI per worker node.

## Requirements

This worker requires both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [ComfyUI API Wrapper](https://github.com/ai-dock/comfyui-api-wrapper).

A docker image is provided but you may use any if the above requirements are met.

## Benchmarking

### Custom Benchmark Workflows

You can provide a custom ComfyUI workflow for benchmarking by creating `workers/comfyui-json/misc/benchmark.json`. This allows you to test performance using your preferred models and workflow complexity.

**Ways to provide the benchmark file:**
- Fork this repository and add your `benchmark.json` file
- Write the file during worker provisioning (onstart script or setup phase)

An example file is provided in the repository. To ensure varied generations, use the placeholder `__RANDOM_INT__` in place of static seed values - it will be replaced with a random integer for each benchmark run.

### Default Benchmark (Fallback)

If `benchmark.json` is not available, a simple image generation benchmark runs when each worker initializes. This validates GPU performance and helps identify underperforming machines.

The default benchmark uses Stable Diffusion v1.5 with ComfyUI's standard text-to-image workflow. Configure it using these environment variables:

| Environment Variable | Default Value | Description |
| -------------------- | ------------- | ----------- |
| BENCHMARK_TEST_WIDTH | 512 | Image width (pixels) |
| BENCHMARK_TEST_HEIGHT | 512 | Image height (pixels) |
| BENCHMARK_TEST_STEPS | 20 | Number of denoising steps |

Each benchmark run uses a random prompt from `misc/test_prompts.txt` and a random seed to ensure consistent GPU load patterns.

#### Calibrating Fallback Benchmark Duration

To screen for underperforming hardware, set `BENCHMARK_TEST_STEPS` to match your expected production workflow duration. This allows you to identify machines that won't meet performance requirements.

**Example:** If your typical workflow should complete in 90 seconds on acceptable hardware:

```bash
# 1. Measure it/sec on your reference machine
# RTX 4090 typically achieves ~43 it/sec with SD1.5

# 2. Calculate required steps
# 90 seconds × 43 it/sec = 3870 steps

# 3. Configure benchmark
export BENCHMARK_TEST_STEPS=3870

# 4. Machines completing significantly slower than 90s indicate hardware issues
```

**Performance expectations:**
- Benchmark duration should remain consistent across identical GPU models
- Significant variation (>20%) may indicate thermal, power, or configuration issues

## Endpoint

The worker provides a single endpoint:

- `/generate/sync`: Processes ComfyUI workflows using either predefined modifiers or custom workflow JSON

## Request Format

The worker accepts requests in the following format. Choose either modifier mode OR custom workflow mode:

**Modifier Mode:**
```json
{
  "input": {
    "request_id": "uuid-string",    // optional - UUID generated if not provided
    "modifier": "RawWorkflow",
    "modifications": {
      "prompt": "a beautiful landscape",
      "width": 1024,
      "height": 1024,
      "steps": 20,
      "seed": 123456789
    },
    "s3": { ... },       // optional
    "webhook": { ... }   // optional
  }
}
```

**Custom Workflow Mode:**
```json
{
  "input": {
    "request_id": "uuid-string",    // optional - UUID generated if not provided
    "workflow_json": {
      // Complete ComfyUI workflow JSON
    },
    "s3": { ... },       // optional
    "webhook": { ... }   // optional
  }
}
```

## Request Fields

### Required Fields

- **`input`**: Contains the main workflow data
- **`input.request_id`**: Unique identifier for the request

### Workflow Mode (Choose One)

You must provide either `modifier` OR `workflow_json`, but not both:

#### Option 1: Modifier Mode
- **`input.modifier`**: Name of the predefined workflow modifier (e.g., "Text2Image")
- **`input.modifications`**: Parameters to pass to the modifier

#### Option 2: Custom Workflow Mode  
- **`input.workflow_json`**: Complete ComfyUI workflow JSON

### Optional Fields

- **`input.s3`**: S3 configuration for file storage
- **`input.webhook`**: Webhook configuration for notifications

These configurations can be provided in the request JSON or via environment variables. Request-level configuration takes precedence over environment variables.

#### S3 Configuration

**Via Request JSON:**
```json
"s3": {
  "access_key_id": "your-s3-access-key",
  "secret_access_key": "your-s3-secret-access-key", 
  "endpoint_url": "https://my-endpoint.backblaze.com",
  "bucket_name": "your-bucket",
  "region": "us-east-1"
}
```

**Via Environment Variables:**
```bash
S3_ACCESS_KEY_ID=your-key
S3_SECRET_ACCESS_KEY=your-secret
S3_BUCKET_NAME=your-bucket
S3_ENDPOINT_URL=https://s3.amazonaws.com
S3_REGION=us-east-1
```

#### Webhook Configuration

**Via Request JSON:**
```json
"webhook": {
  "url": "your-webhook-url",
  "extra_params": {
    "custom_field": "value"
  }
}
```

**Via Environment Variables:**
```bash
WEBHOOK_URL=https://your-webhook.com  # Default webhook URL
WEBHOOK_TIMEOUT=30                   # Webhook timeout in seconds
```

## Examples

### Basic Text-to-Image (Modifier Mode)

```json
{
  "input": {
    "modifier": "Text2Image",
    "modifications": {
      "prompt": "a cat sitting on a windowsill",
      "width": 512,
      "height": 512,
      "steps": 20,
      "seed": 42
    }
  }
}
```

### Custom Workflow Mode

```json
{
  "input": {
    "request_id": "67890",    // optional - using custom ID for tracking
    "workflow_json": {
      "3": {
        "inputs": {
          "seed": 42,
          "steps": 20,
          "cfg": 8,
          "sampler_name": "euler",
          "scheduler": "normal",
          "denoise": 1,
          "model": ["4", 0],
          "positive": ["6", 0],
          "negative": ["7", 0],
          "latent_image": ["5", 0]
        },
        "class_type": "KSampler"
      }
    }
  }
}
```

## Client Libraries

See the test client examples for implementation details on how to integrate with the ComfyUI worker.

---

See Vast's serverless documentation for more details on how to use ComfyUI with autoscaler.