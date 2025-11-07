# Vast PyWorker

Vast PyWorker is a Python web server designed to run alongside a LLM or image generation models running on vast,
enabling autoscaler integration.
It serves as the primary entry point for API requests, forwarding them to the model's API hosted on the
same instance. Additionally, it monitors performance metrics and estimates current workload based on factors
such as the number of tokens processed for LLMs or image resolution and steps for image generation models,
reporting these metrics to the autoscaler.

## Project Structure

*   `lib/`: Contains the core PyWorker framework code (server logic, data types, metrics).
*   `workers/`: Contains specific implementations (PyWorkers) for different model servers. Each subdirectory represents a worker for a particular model type.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    You may also need `pyright` for type checking:
    ```bash
    sudo npm install -g pyright
    # or use your preferred method to install pyright
    ```

2.  **Configure Environment:** Set any necessary environment variables (e.g., `MODEL_LOG` path, API keys if needed by your worker).

3.  **Run the Server:** Use the provided script. You'll need to specify which worker to run.
    ```bash
    # Example for hello_world worker (assuming MODEL_LOG is set)
    ./start_server.sh workers.hello_world.server
    ```
    Replace `workers.hello_world.server` with the path to the `server.py` module of the worker you want to run.

## How to Use

### Using Existing Workers

If you are using a Vast.ai template that includes PyWorker integration (marked as autoscaler compatible), it should work out of the box. The template will typically start the appropriate PyWorker server automatically. Here's a few:

*   **TGI (Text Generation Inference):** [Vast.ai Template](https://cloud.vast.ai?ref_id=140778&template_id=72d8dcb41ea3a58e06c741e2c725bc00)
*   **ComfyUI:** [Vast.ai Template](https://cloud.vast.ai?ref_id=140778&template_id=ad72c8bf7cf695c3c9ddf0eaf6da0447)

Currently available workers:
*   `hello_world`: A simple example worker for a basic LLM server.
*   `comfyui`: A worker for the ComfyUI image generation backend.
*   `openai`: A worker for OpenAI-compatible inference servers (vLLM, Ollama, TGI, llama.cpp).
*   `tgi`: A worker for the Text Generation Inference backend.

### Implementing a New Worker

To integrate PyWorker with a model server not already supported, you need to create a new worker implementation under the `workers/` directory. Follow these general steps:

1.  **Create Worker Directory:** Add a new directory under `workers/` (e.g., `workers/my_model/`).
2.  **Define Data Types (`data_types.py`):**
    *   Create a class inheriting from `lib.data_types.ApiPayload`.
    *   Implement methods like `for_test`, `generate_payload_json`, `count_workload`, and `from_json_msg` to handle request data, testing, and workload calculation specific to your model's API.
3.  **Implement Endpoint Handlers (`server.py`):**
    *   For each model API endpoint you want PyWorker to proxy, create a class inheriting from `lib.data_types.EndpointHandler`.
    *   Implement methods like `endpoint`, `payload_cls`, `generate_payload_json`, `make_benchmark_payload` (for one handler), and `generate_client_response`.
    *   Instantiate `lib.backend.Backend` with your model server details, log file path, benchmark handler, and log actions.
    *   Define `aiohttp` routes, mapping paths to your handlers using `backend.create_handler()`.
    *   Use `lib.server.start_server` to run the application.
4.  **Add `__init__.py`:** Create an empty `__init__.py` file in your worker directory.
5.  **(Optional) Add Load Testing (`test_load.py`):** Create a script using `lib.test_harness.run` to test your worker against a Vast.ai endpoint group.
6.  **(Optional) Add Client Example (`client.py`):** Provide a script demonstrating how to call your worker's endpoints.

**For a detailed walkthrough, refer to the `hello_world` example:** [workers/hello_world/README.md](workers/hello_world/README.md)


**Type Hinting:** It is strongly recommended to use strict type hinting throughout your implementation. Use `pyright` to check for type errors.

## Testing Your Worker

If you implement a `test_load.py` script for your worker, you can use it to load test a Vast.ai endpoint group running your instance image.

```bash
# Example for hello_world worker
python3 -m workers.hello_world.test_load -n 1000 -rps 0.5 -k "$API_KEY" -e "$ENDPOINT_GROUP_NAME"
```

Replace `workers.hello_world.test_load` with the path to your worker's test script and provide your Vast.ai API Key (`-k`) and the target Endpoint Group Name (`-e`). Adjust the number of requests (`-n`) and requests per second (`-rps`) as needed.

## Community & Support

Join the conversation and get help:

*   **Vast.ai Discord:** [https://discord.gg/Pa9M29FFye](https://discord.gg/Pa9M29FFye)
*   **Vast.ai Subreddit:** [https://reddit.com/r/vastai/](https://reddit.com/r/vastai/)
