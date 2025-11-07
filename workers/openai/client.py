import logging
import sys
import json
import subprocess
from urllib.parse import urljoin
from typing import Dict, Any, Optional, Iterator, Union, List
import requests
from utils.endpoint_util import Endpoint
from utils.ssl import get_cert_file_path
from .data_types.client import CompletionConfig, ChatCompletionConfig

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s[%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__file__)

COMPLETIONS_PROMPT = "the capital of USA is"
CHAT_PROMPT = "Think step by step: Tell me about the Python programming language."
TOOLS_PROMPT = "Can you list the files in the current working directory and tell me what you see? What do you think this directory might be for?"


class APIClient:
    """Lightweight client focused solely on API communication"""

    # Remove the generic WORKER_ENDPOINT since we're now going direct
    DEFAULT_COST = 100
    DEFAULT_TIMEOUT = 4

    def __init__(
        self,
        endpoint_group_name: str,
        api_key: str,
        server_url: str,
        endpoint_api_key: str,
    ):
        self.endpoint_group_name = endpoint_group_name
        self.api_key = api_key
        self.server_url = server_url
        self.endpoint_api_key = endpoint_api_key

    def _get_worker_url(self, cost: int = DEFAULT_COST) -> Dict[str, Any]:
        """Get worker URL and auth data from routing service"""
        if not self.endpoint_api_key:
            raise ValueError("No valid endpoint API key available")

        route_payload = {
            "endpoint": self.endpoint_group_name,
            "api_key": self.endpoint_api_key,
            "cost": cost,
        }

        response = requests.post(
            urljoin(self.server_url, "/route/"),
            json=route_payload,
            timeout=self.DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _create_auth_data(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Create auth data from routing response"""
        return {
            "signature": message["signature"],
            "cost": message["cost"],
            "endpoint": message["endpoint"],
            "reqnum": message["reqnum"],
            "request_idx": message.get("request_idx",  message["reqnum"]),  # Use reqnum as fallback
            "url": message["url"],
        }

    def _make_request(
        self,
        payload: Dict[str, Any],
        endpoint: str,
        method: str = "POST",
        stream: bool = False,
    ) -> Union[Dict[str, Any], Iterator[str]]:
        """Make request directly to the specific worker endpoint"""
        # Get worker URL and auth data
        cost = payload.get("max_tokens", self.DEFAULT_COST)
        message = self._get_worker_url(cost=cost)
        worker_url = message["url"]
        auth_data = self._create_auth_data(message)

        req_data = {"payload": {"input": payload}, "auth_data": auth_data}

        url = urljoin(worker_url, endpoint)
        log.debug(f"Making direct request to: {url}")
        log.debug(f"Payload: {req_data}")

        # Make the request using the specified method
        if method.upper() == "POST":
            response = requests.post(
                url, json=req_data, stream=stream, verify=get_cert_file_path()
            )
        elif method.upper() == "GET":
            response = requests.get(
                url, params=req_data, stream=stream, verify=get_cert_file_path()
            )
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        response.raise_for_status()

        if stream:
            return self._handle_streaming_response(response)
        else:
            return response.json()

    def _handle_streaming_response(self, response: requests.Response) -> Iterator[str]:
        """Handle streaming response and yield tokens"""
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            yield data  # Yield the full chunk
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            log.error(f"Error handling streaming response: {e}")
            raise

    def call_completions(
        self, config: CompletionConfig
    ) -> Union[Dict[str, Any], Iterator[str]]:
        payload = config.to_dict()

        return self._make_request(
            payload=payload, endpoint="/v1/completions", stream=config.stream
        )

    def call_chat_completions(
        self, config: ChatCompletionConfig
    ) -> Union[Dict[str, Any], Iterator[str]]:
        payload = config.to_dict()

        return self._make_request(
            payload=payload, endpoint="/v1/chat/completions", stream=config.stream
        )


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
        """Get the ls tool definition"""
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
        function_name = tool_call["function"]["name"]

        if function_name == "list_files":
            return self.list_files()
        else:
            raise ValueError(f"Unknown tool function: {function_name}")


class APIDemo:
    """Demo and testing functionality for the API client"""

    def __init__(
        self, client: APIClient, model: str, tool_manager: Optional[ToolManager] = None
    ):
        self.client = client
        self.model = model
        self.tool_manager = tool_manager or ToolManager()

    def handle_streaming_response(
        self, response_stream, show_reasoning: bool = True
    ) -> str:
        """
        Handle streaming chat response and display all output.
        """

        full_response = ""
        reasoning_content = ""
        reasoning_started = False
        content_started = False

        for chunk in response_stream:
            # Normalize the chunk
            if isinstance(chunk, str):
                chunk = chunk.strip()
                if chunk.startswith("data: "):
                    chunk = chunk[6:].strip()
                if chunk in ["[DONE]", ""]:
                    continue
                try:
                    parsed_chunk = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
            elif isinstance(chunk, dict):
                parsed_chunk = chunk
            else:
                continue

            # Parse delta from the chunk
            choices = parsed_chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            reasoning_token = delta.get("reasoning_content", "")
            content_token = delta.get("content", "")

            # Print reasoning token if applicable
            if show_reasoning and reasoning_token:
                if not reasoning_started:
                    print("\n🧠 Reasoning: ", end="", flush=True)
                    reasoning_started = True
                print(f"\033[90m{reasoning_token}\033[0m", end="", flush=True)
                reasoning_content += reasoning_token

            # Print content token
            if content_token:
                if not content_started:
                    if show_reasoning and reasoning_started:
                        print(f"\n💬 Response: ", end="", flush=True)
                    else:
                        print("Assistant: ", end="", flush=True)
                    content_started = True
                print(content_token, end="", flush=True)
                full_response += content_token

        print()  # Ensure newline after response

        if show_reasoning:
            if reasoning_started or content_started:
                print("\nStreaming completed.")
            if reasoning_started:
                print(f"Reasoning tokens: {len(reasoning_content.split())}")
            if content_started:
                print(f"Response tokens: {len(full_response.split())}")

        return full_response

    def test_tool_support(self) -> bool:
        """Test if the endpoint supports function calling"""
        log.debug("Testing endpoint tool calling support...")

        # Try a simple request with minimal tools to test support
        messages = [{"role": "user", "content": "Hello"}]
        minimal_tool = [
            {
                "type": "function",
                "function": {"name": "test_function", "description": "Test function"},
            }
        ]

        config = ChatCompletionConfig(
            model=self.model,
            messages=messages,
            max_tokens=10,
            tools=minimal_tool,
            tool_choice="none",  # Don't actually call the tool
        )

        try:
            response = self.client.call_chat_completions(config)
            return True
        except Exception as e:
            log.error(f"Error: Endpoint does not support tool calling: {e}")
            return False

    def demo_completions(self) -> None:
        """Demo: test basic completions endpoint"""
        print("=" * 60)
        print("COMPLETIONS DEMO")
        print("=" * 60)

        config = CompletionConfig(
            model=self.model, prompt=COMPLETIONS_PROMPT, stream=False
        )

        log.info(
            f"Testing completions with model '{self.model}' and prompt: '{config.prompt}'"
        )
        response = self.client.call_completions(config)

        if isinstance(response, dict):
            print("\nResponse:")
            print(json.dumps(response, indent=2))
        else:
            log.error("Unexpected response format")

    def demo_chat(self, use_streaming: bool = True) -> None:
        """
        Demo: test chat completions endpoint with optional streaming
        """
        print("=" * 60)
        print(
            f"CHAT COMPLETIONS DEMO {'(STREAMING)' if use_streaming else '(NON-STREAMING)'}"
        )
        print("=" * 60)

        config = ChatCompletionConfig(
            model=self.model,
            messages=[{"role": "user", "content": CHAT_PROMPT}],
            stream=use_streaming,
        )

        log.info(f"Testing chat completions with model '{self.model}'...")
        response = self.client.call_chat_completions(config)

        if use_streaming:
            try:
                self.handle_streaming_response(response, show_reasoning=True)
            except Exception as e:
                log.error(f"\nError during streaming: {e}")
                import traceback

                traceback.print_exc()
                return

        else:
            if isinstance(response, dict):
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content", "")
                reasoning = message.get("reasoning_content", "") or message.get(
                    "reasoning", ""
                )

                if reasoning:
                    print(f"\n🧠 Reasoning: \033[90m{reasoning}\033[0m")

                print(f"\n💬 Assistant: {content}")
                print(f"\nFull Response:")
                print(json.dumps(response, indent=2))
            else:
                log.error("Unexpected response format")

    def demo_ls_tool(self) -> None:
        """Demo: ask LLM to list files in the current directory and describe what it sees"""
        print("=" * 60)
        print("TOOL USE DEMO: List Directory Contents")
        print("=" * 60)

        # Test if tools are supported first
        if not self.test_tool_support():
            return

        # Request with tool available
        messages = [{"role": "user", "content": TOOLS_PROMPT}]

        config = ChatCompletionConfig(
            model=self.model,
            messages=messages,
            tools=self.tool_manager.get_ls_tool_definition(),
            tool_choice="auto",
        )

        log.info(f"Making initial request with tool using model '{self.model}'...")
        response = self.client.call_chat_completions(config)

        if not isinstance(response, dict):
            raise ValueError("Expected dict response for tool use")

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        print(f"Assistant response: {message.get('content', 'No content')}")

        # Check for tool calls
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            raise ValueError(
                "No tool calls made - model may not support function calling"
            )

        print(f"Tool calls detected: {len(tool_calls)}")

        # Execute the tool call
        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            print(f"Executing tool: {function_name}")

            tool_result = self.tool_manager.execute_tool_call(tool_call)
            print(f"Tool result:\n{tool_result}")

            # Add tool result and continue conversation
            messages.append(message)  # Add assistant's message with tool call
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

            # Get final response
            final_config = ChatCompletionConfig(
                model=self.model,
                messages=messages,
                tools=self.tool_manager.get_ls_tool_definition(),
            )

            print("Getting final response...")
            final_response = self.client.call_chat_completions(final_config)

            if isinstance(final_response, dict):
                final_choice = final_response.get("choices", [{}])[0]
                final_message = final_choice.get("message", {})
                final_content = final_message.get("content", "")

                print("\n" + "=" * 60)
                print("FINAL LLM ANALYSIS:")
                print("=" * 60)
                print(final_content)
                print("=" * 60)

    def interactive_chat(self) -> None:
        """Interactive chat session with streaming"""
        print("=" * 60)
        print("INTERACTIVE STREAMING CHAT")
        print("=" * 60)
        print(f"Using model: {self.model}")
        print("Type 'quit' to exit, 'clear' to clear history")
        print()

        messages = []

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

                config = ChatCompletionConfig(
                    model=self.model, messages=messages, stream=True, temperature=0.7
                )

                print("Assistant: ", end="", flush=True)

                response = self.client.call_chat_completions(config)
                assistant_content = self.handle_streaming_response(
                    response, show_reasoning=True
                )

                # Add assistant response to conversation history
                messages.append({"role": "assistant", "content": assistant_content})

            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                log.error(f"\nError: {e}")
                continue


def main():
    """Main function with CLI switches for different tests"""
    from lib.test_utils import test_args

    # Add mandatory model argument
    test_args.add_argument(
        "--model", required=True, help="Model to use for requests (required)"
    )

    # Add test mode arguments
    test_args.add_argument(
        "--completion", action="store_true", help="Test completions endpoint"
    )
    test_args.add_argument(
        "--chat",
        action="store_true",
        help="Test chat completions endpoint (non-streaming)",
    )
    test_args.add_argument(
        "--chat-stream",
        action="store_true",
        help="Test chat completions endpoint with streaming",
    )
    test_args.add_argument(
        "--tools",
        action="store_true",
        help="Test function calling with ls tool (non-streaming)",
    )
    test_args.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive streaming chat session",
    )

    args = test_args.parse_args()

    # Check that only one test mode is selected
    test_modes = [
        args.completion,
        args.chat,
        args.chat_stream,
        args.tools,
        args.interactive,
    ]
    selected_count = sum(test_modes)

    if selected_count == 0:
        print("Please specify exactly one test mode:")
        print("  --completion    : Test completions endpoint")
        print("  --chat          : Test chat completions endpoint (non-streaming)")
        print("  --chat-stream   : Test chat completions endpoint with streaming")
        print("  --tools         : Test function calling with ls tool (non-streaming)")
        print("  --interactive   : Start interactive streaming chat session")
        print(
            f"\nExample: python {sys.argv[0]} --model Qwen/Qwen3-8B --chat-stream -k YOUR_KEY -e YOUR_ENDPOINT"
        )
        sys.exit(1)
    elif selected_count > 1:
        print("Please specify exactly one test mode")
        sys.exit(1)

    try:
        endpoint_api_key = Endpoint.get_endpoint_api_key(
            endpoint_name=args.endpoint_group_name,
            account_api_key=args.api_key,
            instance=args.instance,
        )

        if not endpoint_api_key:
            log.error(
                f"Could not retrieve API key for endpoint '{args.endpoint_group_name}'. Exiting."
            )
            sys.exit(1)

        # Create the core API client
        client = APIClient(
            endpoint_group_name=args.endpoint_group_name,
            api_key=args.api_key,
            server_url=Endpoint.get_autoscaler_server_url(args.instance),
            endpoint_api_key=endpoint_api_key,
        )

        # Create tool manager and demo (passing the model parameter)
        tool_manager = ToolManager()
        demo = APIDemo(client, args.model, tool_manager)

        print(f"Using model: {args.model}")
        print("=" * 60)

        # Run the selected test
        if args.completion:
            demo.demo_completions()
        elif args.chat:
            demo.demo_chat(use_streaming=False)
        elif args.chat_stream:
            demo.demo_chat(use_streaming=True)
        elif args.tools:
            demo.demo_ls_tool()
        elif args.interactive:
            demo.interactive_chat()

    except Exception as e:
        log.error(f"Error during test: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
