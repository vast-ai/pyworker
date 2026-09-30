import logging
import base64
import json
import os
import sys
import subprocess
import argparse
from typing import Any, Dict, List, Optional

from vastai import Serverless
import asyncio

# ---------------------- Logging ----------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

# ---------------------- Prompts ----------------------
COMPLETIONS_PROMPT = "Zebras are primarily grazers and can subsist on lower-quality vegetation. They are preyed on mainly by"
CHAT_PROMPT = "Think step by step: Tell me about the Python programming language."
TOOLS_PROMPT = (
    "Can you list the files in the current working directory and tell me what you see? "
    "What do you think this directory might be for?"
)

ENDPOINT_NAME = "my-vllm-endpoint"       # change this to your vLLM endpoint name
DEFAULT_MODEL = "Qwen/Qwen3-8B"          # must support tool calling
MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.7

# ---------------------- Tooling ----------------------
class ToolManager:
    """Handles tool definitions and execution"""

    @staticmethod
    def list_files() -> str:
        """Execute ls on current directory"""
        try:
            result = subprocess.run(
                ["ls", "-la", "."], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                return result.stdout
            else:
                return f"Error: {result.stderr}"
        except Exception as e:
            return f"Error running ls: {e}"

    @staticmethod
    def get_ls_tool_definition() -> List[Dict[str, Any]]:
        """OpenAI-compatible tool schema"""
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_files",
                    "description": "List files and directories in the cwd",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            }
        ]

    def execute_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Execute a tool call and return the result"""
        function_name = (tool_call.get("function") or {}).get("name")
        if function_name == "list_files":
            return self.list_files()
        raise ValueError(f"Unknown tool function: {function_name}")


# ----- Helpers to handle streamed tool_calls assembly -----
def _merge_tool_call_delta(state: Dict[int, Dict[str, Any]], tc_delta: Dict[str, Any]) -> None:
    """
    OpenAI-style streaming sends partial tool_calls with an index and partial fields.
    We merge into a per-index state dict until the assistant message finishes.
    """
    idx = tc_delta.get("index")
    if idx is None:
        return

    entry = state.setdefault(idx, {"id": None, "function": {"name": None, "arguments": ""}, "type": "function"})

    if tc_delta.get("id"):
        entry["id"] = tc_delta["id"]

    fn_delta = tc_delta.get("function") or {}
    if "name" in fn_delta and fn_delta["name"]:
        entry["function"]["name"] = fn_delta["name"]
    if "arguments" in fn_delta and fn_delta["arguments"]:
        entry["function"]["arguments"] += fn_delta["arguments"]


def _tool_state_to_message_tool_calls(state: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [state[i] for i in sorted(state.keys())]


# ---- OpenAI-compatible calls (non-streaming) ----
async def call_completions(client: Serverless, *, model: str, prompt: str, endpoint_name: str, **kwargs) -> Dict[str, Any]:

    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
        "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
    }
    log.debug("POST /v1/completions %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/v1/completions", payload, cost=payload["max_tokens"])
    return resp["response"]

async def call_chat_completions(client: Serverless, *, model: str, messages: List[Dict[str, Any]], endpoint_name: str, **kwargs) -> Dict[str, Any]:

    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
        "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
        **({"tools": kwargs["tools"]} if "tools" in kwargs else {}),
        **({"tool_choice": kwargs["tool_choice"]} if "tool_choice" in kwargs else {}),
    }
    log.debug("POST /v1/chat/completions %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/v1/chat/completions", payload, cost=payload["max_tokens"])
    return resp["response"]

# ---- Streaming variants ----
async def stream_completions(client: Serverless, *, model: str, prompt: str, endpoint_name: str, **kwargs):

    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
        "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
        "stream": True,
        **({"stop": kwargs["stop"]} if "stop" in kwargs else {}),
    }
    log.debug("STREAM /v1/completions %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/v1/completions", payload, cost=payload["max_tokens"], stream=True)
    return resp["response"]  # async generator

async def stream_chat_completions(client: Serverless, *, model: str, messages: List[Dict[str, Any]], endpoint_name: str, **kwargs):

    endpoint = await client.get_endpoint(name=endpoint_name)

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", MAX_TOKENS),
        "temperature": kwargs.get("temperature", DEFAULT_TEMPERATURE),
        "stream": True,
        **({"tools": kwargs["tools"]} if "tools" in kwargs else {}),
        **({"tool_choice": kwargs["tool_choice"]} if "tool_choice" in kwargs else {}),
    }
    log.debug("STREAM /v1/chat/completions %s", json.dumps(payload)[:500])
    resp = await endpoint.request("/v1/chat/completions", payload, cost=payload["max_tokens"], stream=True)
    return resp["response"]  # async generator


# ---- The other routes: one call each, saving any media they return ----
SPEECH_TEXT = "A quick update on the project: the first round of testing went well."
IMAGE_PROMPT = "A lighthouse on a rocky coast at dusk, oil painting"
EDIT_PROMPT = "Make it a snowy winter scene"
AUDIO_PROMPT = "Rain on a tin roof with distant thunder"
VIDEO_PROMPT = "A red kite flying over a beach, slow camera pan"
BATCH_TEXTS = ["Name a primary colour.", "Name a planet.", "Name an ocean."]
EXTENSIONS = {"audio/wav": "wav", "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/flac": "flac",
              "audio/ogg": "ogg", "audio/opus": "opus", "video/mp4": "mp4", "image/png": "png"}


async def call_route(client: Serverless, route: str, payload: Dict[str, Any], endpoint_name: str):
    """(response, content_type): the response is JSON, text, or bytes for media."""
    endpoint = await client.get_endpoint(name=endpoint_name)
    log.debug("POST %s %s", route, json.dumps(payload)[:500])
    resp = await endpoint.request(route, payload)
    return resp["response"], resp.get("content_type")


def b64_file(path: str) -> Dict[str, str]:
    with open(path, "rb") as f:
        return {"data": base64.b64encode(f.read()).decode(), "name": os.path.basename(path)}


def save(out_dir: str, stem: str, data: bytes, content_type: Optional[str]) -> None:
    ext = EXTENSIONS.get((content_type or "").split(";")[0], "bin")
    path = os.path.join(out_dir, f"{stem}.{ext}")
    with open(path, "wb") as f:
        f.write(data)
    print(f"Saved {path} ({len(data)} bytes, {content_type})")


async def demo_route(client: Serverless, args: argparse.Namespace) -> None:
    """Run the one route mode selected on the command line."""
    model = {"model": args.model} if args.model else {}
    out = args.out
    os.makedirs(out, exist_ok=True)

    if args.embeddings:
        resp, _ = await call_route(client, "/v1/embeddings", {**model, "input": [SPEECH_TEXT]}, args.endpoint)
        vectors = resp.get("data", [])
        print(f"{len(vectors)} vector(s) of {len(vectors[0]['embedding']) if vectors else 0} dimensions")
    elif args.speech:
        resp, ctype = await call_route(client, "/v1/audio/speech",
                                       {**model, "input": SPEECH_TEXT, "response_format": "wav",
                                        "voice": args.voice},
                                       args.endpoint)
        save(out, "speech", resp, ctype)
    elif args.speech_batch:
        resp, _ = await call_route(client, "/v1/audio/speech/batch",
                                   {**model, "items": [{"input": t} for t in BATCH_TEXTS],
                                    "response_format": "wav", "voice": args.voice}, args.endpoint)
        for r in resp.get("results", []):
            if r.get("status") == "success":
                save(out, f"speech-{r['index']}", base64.b64decode(r["audio_data"]), r.get("media_type"))
            else:
                print(f"item {r.get('index')}: {r.get('error')}")
    elif args.transcribe or args.translate:
        route = "/v1/audio/transcriptions" if args.transcribe else "/v1/audio/translations"
        f = b64_file(args.transcribe or args.translate)
        resp, _ = await call_route(client, route, {**model, "file": f["data"], "filename": f["name"]},
                                   args.endpoint)
        print(resp.get("text") if isinstance(resp, dict) else resp)
    elif args.image:
        resp, _ = await call_route(client, "/v1/images/generations",
                                   {**model, "prompt": IMAGE_PROMPT, "size": "1024x1024", "n": 1},
                                   args.endpoint)
        save(out, "image", base64.b64decode(resp["data"][0]["b64_json"]), "image/png")
    elif args.edit:
        f = b64_file(args.edit)
        resp, _ = await call_route(client, "/v1/images/edits",
                                   {**model, "image": f["data"], "filename": f["name"], "prompt": EDIT_PROMPT},
                                   args.endpoint)
        save(out, "edit", base64.b64decode(resp["data"][0]["b64_json"]), "image/png")
    elif args.audio_generate:
        resp, ctype = await call_route(client, "/v1/audio/generate",
                                       {**model, "input": AUDIO_PROMPT, "audio_length": args.seconds},
                                       args.endpoint)
        save(out, "audio", resp, ctype)
    elif args.video:
        payload = {**model, "prompt": VIDEO_PROMPT, "width": 832, "height": 480, "num_frames": 33}
        if args.image_file:
            f = b64_file(args.image_file)
            payload.update(input_reference=f["data"], input_reference_filename=f["name"])
        resp, ctype = await call_route(client, "/v1/videos/sync", payload, args.endpoint)
        save(out, "video-i2v" if args.image_file else "video", resp, ctype)
    elif args.chat_batch:
        resp, _ = await call_route(client, "/v1/chat/completions/batch",
                                   {**model, "messages": [[{"role": "user", "content": t}] for t in BATCH_TEXTS],
                                    "max_tokens": 64}, args.endpoint)
        for text, choice in zip(BATCH_TEXTS, resp.get("choices", [])):
            print(f"{text} -> {choice['message']['content']}")


# ---------------------- Demo Runner ----------------------
class APIDemo:
    """Demo and testing functionality for the API client"""

    def __init__(self, client: Serverless, model: str, endpoint_name: str, tool_manager: Optional[ToolManager] = None):
        self.client = client
        self.model = model
        self.endpoint_name = endpoint_name
        self.tool_manager = tool_manager or ToolManager()

    # ----- Streaming handler -----
    async def handle_streaming_response(self, stream, show_reasoning: bool = True) -> str:
        full_response = ""
        reasoning_content = ""
        printed_reasoning = False
        printed_answer = False
        finish_reason = None

        async for chunk in stream:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})
            
            # Track finish reason
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")

            # reasoning tokens
            rc = delta.get("reasoning_content")
            if rc and show_reasoning:
                if not printed_reasoning:
                    print("\n🧠 Reasoning: ", end="", flush=True)
                    printed_reasoning = True
                print(rc, end="", flush=True)
                reasoning_content += rc

            # content tokens
            content_part = delta.get("content")
            if content_part:
                if not printed_answer:
                    if show_reasoning and printed_reasoning:
                        print("\n💬 Response: ", end="", flush=True)
                    else:
                        print("Assistant: ", end="", flush=True)
                    printed_answer = True
                print(content_part, end="", flush=True)
                full_response += content_part

        print()  # newline
        if show_reasoning:
            if printed_reasoning or printed_answer:
                print("\nStreaming completed.")
            if printed_reasoning:
                print(f"Reasoning tokens: {len(reasoning_content.split())}")
            if printed_answer:
                print(f"Response tokens: {len(full_response.split())}")
            if finish_reason:
                print(f"Finish reason: {finish_reason}")

        return full_response
    
    async def demo_completions(self) -> None:
        print("=" * 60)
        print("COMPLETIONS DEMO")
        print("=" * 60)

        response = await call_completions(
            client=self.client,
            model=self.model,
            prompt=COMPLETIONS_PROMPT,
            endpoint_name=self.endpoint_name,
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        print("\nResponse:")
        print(json.dumps(response, indent=2))

    async def demo_chat(self, use_streaming: bool = True) -> None:
        print("=" * 60)
        print(f"CHAT COMPLETIONS DEMO {'(STREAMING)' if use_streaming else '(NON-STREAMING)'}")
        print("=" * 60)

        messages = [{"role": "user", "content": CHAT_PROMPT}]

        if use_streaming:
            stream = await stream_chat_completions(
                client=self.client,
                model=self.model, 
                messages=messages,
                endpoint_name=self.endpoint_name,
                max_tokens=MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            try:
                await self.handle_streaming_response(stream, show_reasoning=True)
            except Exception as e:
                log.error("\nError during streaming: %s", e, exc_info=True)
        else:
            response = await call_chat_completions(
                client=self.client,
                model=self.model, 
                messages=messages,
                endpoint_name=self.endpoint_name,
                max_tokens=MAX_TOKENS,
                temperature=DEFAULT_TEMPERATURE
            )
            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "")
            reasoning = message.get("reasoning_content", "") or message.get("reasoning", "")
            if reasoning:
                print(f"\n🧠 Reasoning: \033[90m{reasoning}\033[0m")
            print(f"\n💬 Assistant: {content}")
            print(f"\nFull Response:\n{json.dumps(response, indent=2)}")

    async def test_tool_support(self) -> bool:
        """Probe that tool schema is accepted (no actual call)"""
        messages = [{"role": "user", "content": "Hello"}]
        minimal_tool = [
            {
                "type": "function",
                "function": {"name": "test_function", "description": "Test function"},
            }
        ]
        try:
            _ = await call_chat_completions(
                client=self.client,
                model=self.model,
                messages=messages,
                endpoint_name=self.endpoint_name,
                tools=minimal_tool,
                tool_choice="none",
                max_tokens=10
            )
            return True
        except Exception as e:
            log.error("Endpoint does not support tool calling: %s", e)
            return False

    async def demo_ls_tool(self) -> None:
        """Ask to list files using function calling, then provide final analysis"""
        print("=" * 60)
        print("TOOL USE DEMO: List Directory Contents")
        print("=" * 60)

        if not await self.test_tool_support():
            return

        messages: List[Dict[str, Any]] = [{"role": "user", "content": TOOLS_PROMPT}]

        # First pass: let the model decide tools, stream tool_calls and partial content
        stream = await stream_chat_completions(
            client=self.client,
            model=self.model,
            messages=messages,
            endpoint_name=self.endpoint_name,
            tools=self.tool_manager.get_ls_tool_definition(),
            tool_choice="auto",
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        assistant_content_buf: List[str] = []
        tool_calls_state: Dict[int, Dict[str, Any]] = {}
        printed_reasoning = False
        printed_answer = False

        async for chunk in stream:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})

            rc = delta.get("reasoning_content")
            if rc:
                if not printed_reasoning:
                    printed_reasoning = True
                    print("🧠 Reasoning: ", end="", flush=True)
                print(rc, end="", flush=True)

            content_part = delta.get("content")
            if content_part:
                assistant_content_buf.append(content_part)
                if not printed_answer:
                    printed_answer = True
                    print("\n💬 Response: ", end="", flush=True)
                print(content_part, end="", flush=True)

            if "tool_calls" in delta and delta["tool_calls"]:
                for tc_delta in delta["tool_calls"]:
                    _merge_tool_call_delta(tool_calls_state, tc_delta)

        # If no tool calls, we’re done.
        if not tool_calls_state:
            print("\n(No tool calls were made.)")
            return

        # Build assistant message with tool_calls
        assistant_message = {
            "role": "assistant",
            "content": "".join(assistant_content_buf) if assistant_content_buf else None,
            "tool_calls": _tool_state_to_message_tool_calls(tool_calls_state),
        }
        messages.append(assistant_message)

        # Execute tools and feed results back
        for tc in assistant_message["tool_calls"]:
            tool_name = (tc.get("function") or {}).get("name")
            call_id = tc.get("id")
            raw_args = (tc.get("function") or {}).get("arguments") or "{}"

            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
            except Exception as e:
                tool_result = json.dumps({"error": f"Argument parse failed: {str(e)}", "raw_arguments": raw_args})
                messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})
                continue

            try:
                if tool_name == "list_files":
                    tool_result = self.tool_manager.list_files()
                else:
                    tool_result = json.dumps({"error": f"Unknown tool '{tool_name}'"})
            except Exception as e:
                tool_result = json.dumps({"error": f"Tool '{tool_name}' failed: {str(e)}"})

            print("\n[Tool executed]", tool_name)
            print(tool_result[:500] + ("..." if len(tool_result) > 500 else ""))
            messages.append({"role": "tool", "tool_call_id": call_id, "content": tool_result})

        # Second pass: get final streamed answer after tool results
        stream2 = await stream_chat_completions(
            client=self.client,
            model=self.model,
            messages=messages,
            endpoint_name=self.endpoint_name,
            max_tokens=MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        final_buf = []
        printed_reasoning2 = False
        printed_answer2 = False

        async for chunk in stream2:
            choice = (chunk.get("choices") or [{}])[0]
            delta = choice.get("delta", {})

            rc2 = delta.get("reasoning_content")
            if rc2:
                if not printed_reasoning2:
                    printed_reasoning2 = True
                    print("\n🧠 Reasoning (post-tools): ", end="", flush=True)
                print(rc2, end="", flush=True)

            c2 = delta.get("content")
            if c2:
                final_buf.append(c2)
                if not printed_answer2:
                    printed_answer2 = True
                    print("\n💬 Response (final): ", end="", flush=True)
                print(c2, end="", flush=True)

        print("\n" + "=" * 60)
        print("FINAL LLM ANALYSIS:")
        print("=" * 60)
        print("".join(final_buf))
        print("=" * 60)

    async def interactive_chat(self) -> None:
        """Interactive chat session with streaming"""
        print("=" * 60)
        print("INTERACTIVE STREAMING CHAT")
        print("=" * 60)
        print("Type 'quit' to exit, 'clear' to clear history")
        print()

        messages: List[Dict[str, Any]] = []

        while True:
            try:
                user_input = input("You: ").strip()

                if user_input.lower() == "quit":
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == "clear":
                    messages = []
                    print("Chat history cleared")
                    continue
                elif not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})

                print("Assistant: ", end="", flush=True)
                stream = await stream_chat_completions(
                    client=self.client,
                    model=self.model, 
                    messages=messages,
                    endpoint_name=self.endpoint_name,
                    max_tokens=MAX_TOKENS, 
                    temperature=0.7
                )
                assistant_content = await self.handle_streaming_response(stream, show_reasoning=True)

                # Add assistant response to conversation history
                messages.append({"role": "assistant", "content": assistant_content})

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                log.error("\nError: %s", e)
                continue


# ---------------------- CLI ----------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vast vLLM Demo (Serverless SDK)")
    p.add_argument("--model", help=f"Model to use for requests (default: {DEFAULT_MODEL} for "
                   "completions and chat; the worker's own model for the other routes)")
    p.add_argument("--endpoint", default=ENDPOINT_NAME, help=f"Vast endpoint name (default: {ENDPOINT_NAME})")
    p.add_argument("--out", default=".", help="Directory for returned audio, images and video")
    p.add_argument("--voice", default="default", help="Voice for --speech and --speech-batch (default: default)")
    p.add_argument("--seconds", type=float, default=10.0, help="Length for --audio-generate")
    p.add_argument("--image-file", help="Reference image for --video (image to video)")

    modes = p.add_mutually_exclusive_group(required=False)
    modes.add_argument("--completion", action="store_true", help="Test completions endpoint")
    modes.add_argument("--chat", action="store_true", help="Test chat completions endpoint (non-streaming)")
    modes.add_argument("--chat-stream", action="store_true", help="Test chat completions endpoint with streaming")
    modes.add_argument("--tools", action="store_true", help="Test function calling with ls tool (non-streaming+streamed phases)")
    modes.add_argument("--interactive", action="store_true", help="Start interactive streaming chat session")
    modes.add_argument("--chat-batch", action="store_true", help="Test /v1/chat/completions/batch")
    modes.add_argument("--embeddings", action="store_true", help="Test /v1/embeddings")
    modes.add_argument("--speech", action="store_true", help="Test /v1/audio/speech")
    modes.add_argument("--speech-batch", action="store_true", help="Test /v1/audio/speech/batch")
    modes.add_argument("--transcribe", metavar="FILE", help="Test /v1/audio/transcriptions with FILE")
    modes.add_argument("--translate", metavar="FILE", help="Test /v1/audio/translations with FILE")
    modes.add_argument("--image", action="store_true", help="Test /v1/images/generations")
    modes.add_argument("--edit", metavar="FILE", help="Test /v1/images/edits on FILE")
    modes.add_argument("--audio-generate", action="store_true", help="Test /v1/audio/generate")
    modes.add_argument("--video", action="store_true", help="Test /v1/videos/sync")
    return p


ROUTE_MODES = ("chat_batch", "embeddings", "speech", "speech_batch", "transcribe", "translate",
               "image", "edit", "audio_generate", "video")


async def main_async():
    args = build_arg_parser().parse_args()

    route_mode = any(getattr(args, m) for m in ROUTE_MODES)
    selected = sum([args.completion, args.chat, args.chat_stream, args.tools, args.interactive, route_mode])
    if selected == 0:
        print("Please specify exactly one test mode (--help lists them all):")
        print("  --completion    : Test completions endpoint")
        print("  --chat          : Test chat completions endpoint (non-streaming)")
        print("  --chat-stream   : Test chat completions endpoint with streaming")
        print("  --tools         : Test function calling with ls tool")
        print("  --interactive   : Start interactive streaming chat session")
        print(f"\nExample: python {os.path.basename(sys.argv[0])} --model Qwen/Qwen3-8B --chat-stream --endpoint my-vllm-endpoint")
        sys.exit(1)
    elif selected > 1:
        print("Please specify exactly one test mode")
        sys.exit(1)

    if not route_mode:
        args.model = args.model or DEFAULT_MODEL
    print("=" * 60)
    print(f"Using model: {args.model or '(the worker default)'}")
    print(f"Using endpoint: {args.endpoint}")


    try:
        async with Serverless() as client:
            demo = APIDemo(client, args.model, args.endpoint, ToolManager())

            if route_mode:
                await demo_route(client, args)
            elif args.completion:
                await demo.demo_completions()
            elif args.chat:
                await demo.demo_chat(use_streaming=False)
            elif args.chat_stream:
                await demo.demo_chat(use_streaming=True)
            elif args.tools:
                await demo.demo_ls_tool()
            elif args.interactive:
                await demo.interactive_chat()

    except Exception as e:
        log.error("Error during test: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main_async())
