# ComfyUI PyWorker

This is the base PyWorker for ComfyUI. It provides a unified interface for running any ComfyUI workflow through a proxy-based architecture. See the [Serverless documentation](https://docs.vast.ai/serverless) for guides and how-to's.

The cost for each request has a static value of `100`. ComfyUI does not handle concurrent workloads and there is no current provision to load multiple instances of ComfyUI per worker node.

## Instance Setup

1. Pick a template

- [ComfyUI (Serverless)](https://cloud.vast.ai/?ref_id=62897&creator_id=62897&name=ComfyUI%20(Serverless))

2. Follow the [getting started guide](https://docs.vast.ai/documentation/serverless/quickstart) for help with configuring your serverless setup. For testing, we recommend that you use the default options presented by the web interface.

## Requirements

This worker requires both [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [ComfyUI API Wrapper](https://github.com/ai-dock/comfyui-api-wrapper).

A docker image is provided but you may use any if the above requirements are met.

## Client

The client demonstrates how to use the Vast Serverless SDK to generate images, save them locally, and optionally upload to S3-compatible storage.

### Setup

1. Clone the PyWorker repository to your local machine and install the necessary requirements for running the test client.

```bash
git clone https://github.com/vast-ai/pyworker
cd pyworker
pip install uv
uv venv -p 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Set your API key:

```bash
export VAST_API_KEY=<your_api_key>
```

### Usage

```bash
# Default prompt
python -m workers.comfyui-json.client

# Custom prompt
python -m workers.comfyui-json.client --prompt "a cat sitting on a rainbow"

# With options
python -m workers.comfyui-json.client --prompt "sunset" --width 1024 --height 1024 --steps 30

# Using a custom workflow file
python -m workers.comfyui-json.client --workflow my_workflow.json

# With S3 upload
python -m workers.comfyui-json.client --s3
```

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--endpoint` | `my-comfyui-endpoint` | Vast endpoint name |
| `--prompt` | (default) | Text prompt for image generation |
| `--workflow` | (none) | Path to custom workflow JSON file |
| `--width` | 512 | Image width in pixels |
| `--height` | 512 | Image height in pixels |
| `--steps` | 20 | Number of denoising steps |
| `--seed` | (random) | Random seed for reproducibility |
| `--s3` | (disabled) | Upload generated images to S3 |

### Output

Images are saved to `./generated_images/comfy_{seed}.png`.

### S3 Upload (Optional)

You can optionally upload generated images to an S3-compatible storage service (AWS S3, Cloudflare R2, Backblaze B2, etc.) by using the `--s3` flag.

**1. Set environment variables:**

```bash
export S3_ENDPOINT_URL="https://your-account.r2.cloudflarestorage.com"
export S3_BUCKET_NAME="my-bucket"
export S3_ACCESS_KEY_ID="your-access-key-id"
export S3_SECRET_ACCESS_KEY="your-secret-access-key"
```

**2. Run with S3 upload enabled:**

```bash
python -m workers.comfyui-json.client --prompt "a beautiful landscape" --s3
```

Images will be saved locally AND uploaded to `s3://{bucket}/comfyui/{filename}`.

**Note:** Requires `boto3` (`pip install boto3`).

## Benchmarking

### Custom Benchmark Workflows

You can provide a custom ComfyUI workflow for benchmarking. This allows you to test performance using your preferred models and workflow complexity.

**Ways to provide the benchmark file** (in resolution order — first match wins):

1. **Fork this repository** and commit your workflow to `workers/comfyui-json/misc/benchmark.json`.
2. **Write the file during provisioning** to a path *outside* the pyworker tree (e.g. `/workspace/benchmark.json`) and export `BENCHMARK_JSON_PATH` so the worker can find it. The pyworker repo is cloned by `start_server.sh` *after* provisioning runs, so provisioning cannot write into `misc/` directly — the destination would be clobbered, or the clone would fail.
3. **Run on the vast.ai ComfyUI base image.** Its `convert-workflows.sh` maintains `/opt/comfyui-api-wrapper/workflows/pyworker_benchmark.json` as a symlink to the first provisioned workflow; the worker reads this automatically when neither of the above is set. No env var required.

If `BENCHMARK_JSON_PATH` is set but points at a missing or unreadable file, the worker logs a warning and falls through to the next tier rather than going straight to the SD1.5 fallback.

An example workflow is provided at `workers/comfyui-json/misc/benchmark.json.example`. To ensure varied generations, use the placeholder `__RANDOM_INT__` in place of static seed values — it will be replaced with a random integer for each benchmark run.

### Default Benchmark (Fallback)

If `benchmark.json` is not available, a simple image generation benchmark runs when each worker initializes. This validates GPU performance and helps identify underperforming machines.

The default benchmark uses Stable Diffusion v1.5 with ComfyUI's standard text-to-image workflow. Configure it using these environment variables:

| Environment Variable | Default Value | Description |
| -------------------- | ------------- | ----------- |
| BENCHMARK_JSON_PATH | (unset) | Path to a custom workflow file outside the pyworker tree. Used if `misc/benchmark.json` is absent. Falls through to `/opt/comfyui-api-wrapper/workflows/pyworker_benchmark.json` if set but missing. |
| BENCHMARK_TEST_WIDTH | 512 | Fallback benchmark: image width (pixels) |
| BENCHMARK_TEST_HEIGHT | 512 | Fallback benchmark: image height (pixels) |
| BENCHMARK_TEST_STEPS | 20 | Fallback benchmark: number of denoising steps |

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

See the client example for implementation details on how to integrate with the ComfyUI worker.

---

See Vast's serverless documentation for more details on how to use ComfyUI with autoscaler.
