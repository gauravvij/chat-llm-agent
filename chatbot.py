"""
CLI-based chatbot application supporting multiple LLM backends:
- Remote: Qwen 3.5-Flash via OpenRouter API
- Local: LLM inference via llama.cpp HTTP server

Auto Mode: Autonomous multi-turn execution with tool use (read/write/edit/execute).
"""

import sys
import os
import json
import logging
import re
import subprocess
import tempfile
import requests
import datetime
from datetime import timezone
from typing import Optional

# Configure logging to stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "qwen/qwen3.5-flash-02-23"
OPENROUTER_CONFIG_PATH = os.path.expanduser("~/.config/openrouter/config")

LLAMA_CPP_DEFAULT_URL = "http://localhost:8080"
LLAMA_CPP_ENDPOINT = "/v1/chat/completions"

AUTO_MODE_MAX_ITERATIONS = 5  # Pause for feedback after this many steps

SYSTEM_PROMPT = (
    "You are a helpful, concise, and friendly AI assistant. "
    "Use the conversation history to provide contextually aware responses."
)

AUTO_MODE_SYSTEM_PROMPT = """You are an autonomous AI agent capable of completing tasks by using tools.
You have access to the following tools. To use a tool, output a JSON block wrapped in <tool_call> tags.

Available tools:
1. read_file   - Read the contents of a file
   {"tool": "read_file", "path": "<file_path>"}

2. write_file  - Write content to a file (creates or overwrites)
   {"tool": "write_file", "path": "<file_path>", "content": "<file_content>"}

3. edit_file   - Replace a specific string in a file with new content
   {"tool": "edit_file", "path": "<file_path>", "search": "<text_to_find>", "replace": "<replacement_text>"}

4. execute_code - Execute a shell command or Python script and return stdout/stderr
   {"tool": "execute_code", "command": "<shell_command>"}

Rules:
- Use ONE tool per response, then wait for the result before proceeding.
- After receiving a tool result, analyze it and decide the next step.
- When the task is fully complete, output <task_complete> followed by a summary of what was accomplished.
- If you are blocked and cannot proceed, output <task_blocked reason="<explanation>">.
- Do NOT output multiple tool calls in one response.
- Always wrap tool calls in <tool_call>...</tool_call> tags with valid JSON inside.

Example tool call:
<tool_call>
{"tool": "write_file", "path": "/tmp/hello.py", "content": "print('Hello, World!')"}
</tool_call>

Example completion:
<task_complete>
Successfully wrote and executed the script. The factorial of 5 is 120.
</task_complete>
"""

COMMANDS = {
    "/exit": "Exit the chat session",
    "/quit": "Exit the chat session",
    "/history": "View conversation history",
    "/reset": "Reset conversation history",
    "/backend": "Show or switch backend (usage: /backend [openrouter|llamacpp])",
    "/help": "Show available commands",
    "/auto <task>": "Run autonomous agent mode for a given task",
}


# ─────────────────────────────────────────────
# Conversation History Manager
# ─────────────────────────────────────────────
class ConversationHistory:
    """Manages in-memory conversation history with timestamps."""

    def __init__(self):
        """Initialize an empty conversation history."""
        self._history: list[dict] = []

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Either 'user', 'assistant', or 'system'
            content: The message text
        """
        self._history.append(
            {
                "timestamp": datetime.datetime.now(timezone.utc).isoformat(),
                "role": role,
                "content": content,
            }
        )

    def get_messages_for_api(self) -> list[dict]:
        """
        Return messages in the format expected by OpenAI-compatible APIs.

        Returns:
            List of dicts with 'role' and 'content' keys
        """
        return [{"role": m["role"], "content": m["content"]} for m in self._history]

    def get_full_history(self) -> list[dict]:
        """
        Return the full history including timestamps.

        Returns:
            List of message dicts with timestamp, role, and content
        """
        return list(self._history)

    def reset(self) -> None:
        """Clear all conversation history."""
        self._history.clear()

    def __len__(self) -> int:
        """Return the number of messages in history."""
        return len(self._history)


# ─────────────────────────────────────────────
# OpenRouter Backend
# ─────────────────────────────────────────────
class OpenRouterBackend:
    """Backend for Qwen 3.5-Flash via OpenRouter API."""

    def __init__(self, api_key: str, model: str = OPENROUTER_DEFAULT_MODEL):
        """
        Initialize the OpenRouter backend.

        Args:
            api_key: OpenRouter API key
            model: Model identifier to use (default: qwen/qwen3.5-flash-02-23)
        """
        self.api_key = api_key
        self.model = model
        self.base_url = OPENROUTER_BASE_URL
        self.name = f"OpenRouter ({model})"

    @classmethod
    def from_config(cls, model: str = OPENROUTER_DEFAULT_MODEL) -> Optional["OpenRouterBackend"]:
        """
        Load API key from the OpenRouter config file and return an instance.

        Returns:
            OpenRouterBackend instance or None if config is unavailable
        """
        try:
            with open(OPENROUTER_CONFIG_PATH, "r") as f:
                config = json.load(f)
            api_key = config.get("api_key", "").strip()
            if not api_key:
                logger.warning("OpenRouter config found but api_key is empty.")
                return None
            return cls(api_key=api_key, model=model)
        except FileNotFoundError:
            logger.warning(f"OpenRouter config not found at {OPENROUTER_CONFIG_PATH}.")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenRouter config: {e}")
            return None

    def is_available(self) -> bool:
        """
        Check if the OpenRouter API is reachable.

        Returns:
            True if the API responds, False otherwise
        """
        try:
            resp = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=5,
            )
            return resp.status_code == 200
        except requests.RequestException:
            return False

    def chat(self, messages: list[dict], system_prompt: str = SYSTEM_PROMPT,
             max_tokens: int = 2048) -> str:
        """
        Send messages to the OpenRouter API and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt to use (defaults to SYSTEM_PROMPT)
            max_tokens: Maximum tokens in the response

        Returns:
            The assistant's response text

        Raises:
            RuntimeError: If the API call fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/cli-chatbot",
            "X-Title": "CLI Chatbot",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                raise RuntimeError(
                    "Rate limit exceeded. Please wait a moment before sending another message."
                )
            if resp.status_code == 401:
                raise RuntimeError(
                    "Authentication failed. Check your OpenRouter API key."
                )
            if resp.status_code == 402:
                raise RuntimeError(
                    "Insufficient credits on OpenRouter. Please top up your account."
                )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.Timeout:
            raise RuntimeError(
                "Request timed out. The model may be busy — please try again."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                "Cannot connect to OpenRouter. Check your internet connection."
            )
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected API response format: {e}")


# ─────────────────────────────────────────────
# llama.cpp Backend
# ─────────────────────────────────────────────
class LlamaCppBackend:
    """Backend for local LLM inference via llama.cpp HTTP server."""

    def __init__(self, server_url: str = LLAMA_CPP_DEFAULT_URL):
        """
        Initialize the llama.cpp backend.

        Args:
            server_url: Base URL of the llama.cpp server (default: http://localhost:8080)
        """
        self.server_url = server_url.rstrip("/")
        self.endpoint = f"{self.server_url}{LLAMA_CPP_ENDPOINT}"
        self.name = f"llama.cpp ({self.server_url})"

    def is_available(self) -> bool:
        """
        Check if the llama.cpp server is running and reachable.

        Returns:
            True if the server responds, False otherwise
        """
        try:
            resp = requests.get(f"{self.server_url}/health", timeout=3)
            return resp.status_code == 200
        except requests.RequestException:
            try:
                resp = requests.get(f"{self.server_url}/v1/models", timeout=3)
                return resp.status_code == 200
            except requests.RequestException:
                return False

    def chat(self, messages: list[dict], system_prompt: str = SYSTEM_PROMPT,
             max_tokens: int = 2048) -> str:
        """
        Send messages to the llama.cpp server and return the response text.

        Args:
            messages: List of message dicts with 'role' and 'content'
            system_prompt: System prompt to use
            max_tokens: Maximum tokens in the response

        Returns:
            The assistant's response text

        Raises:
            RuntimeError: If the server call fails
        """
        payload = {
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.Timeout:
            raise RuntimeError(
                "llama.cpp server timed out. The model may be processing a long request."
            )
        except requests.ConnectionError:
            raise RuntimeError(
                f"Cannot connect to llama.cpp server at {self.server_url}. "
                "Ensure the server is running: `llama-server -m model.gguf --port 8080`"
            )
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected llama.cpp response format: {e}")


# ─────────────────────────────────────────────
# Backend Manager
# ─────────────────────────────────────────────
class BackendManager:
    """Manages multiple LLM backends and handles selection and fallback logic."""

    def __init__(self, openrouter_model: str = OPENROUTER_DEFAULT_MODEL):
        """
        Initialize available backends and select the active one.

        Args:
            openrouter_model: Model identifier to use for the OpenRouter backend
        """
        self.backends: dict[str, object] = {}
        self.active_backend_name: Optional[str] = None
        self._openrouter_model = openrouter_model
        self._initialize_backends()

    def _initialize_backends(self) -> None:
        """Detect and register available backends."""
        or_backend = OpenRouterBackend.from_config(model=self._openrouter_model)
        if or_backend:
            self.backends["openrouter"] = or_backend
            logger.info("OpenRouter backend registered.")

        llama_url = os.environ.get("LLAMA_CPP_URL", LLAMA_CPP_DEFAULT_URL)
        llama_backend = LlamaCppBackend(server_url=llama_url)
        self.backends["llamacpp"] = llama_backend

        if "openrouter" in self.backends:
            self.active_backend_name = "openrouter"
        else:
            self.active_backend_name = "llamacpp"

    def get_active_backend(self):
        """
        Return the currently active backend instance.

        Returns:
            The active backend object

        Raises:
            RuntimeError: If no backend is configured
        """
        if not self.active_backend_name:
            raise RuntimeError("No backend configured.")
        return self.backends[self.active_backend_name]

    def switch_backend(self, name: str) -> str:
        """
        Switch the active backend by name.

        Args:
            name: Backend name ('openrouter' or 'llamacpp')

        Returns:
            Status message string
        """
        name = name.lower().strip()
        if name not in self.backends:
            available = ", ".join(self.backends.keys())
            return f"Unknown backend '{name}'. Available: {available}"
        self.active_backend_name = name
        backend = self.backends[name]
        return f"Switched to backend: {backend.name}"

    def check_active_availability(self) -> tuple[bool, str]:
        """
        Check if the active backend is currently reachable.

        Returns:
            Tuple of (is_available: bool, message: str)
        """
        backend = self.get_active_backend()
        available = backend.is_available()
        if available:
            return True, f"✅ {backend.name} is reachable."
        else:
            return False, f"❌ {backend.name} is not reachable."

    def status_string(self) -> str:
        """
        Return a human-readable status of all backends.

        Returns:
            Multi-line status string
        """
        lines = ["Backend Status:"]
        for name, backend in self.backends.items():
            marker = "▶" if name == self.active_backend_name else " "
            lines.append(f"  {marker} [{name}] {backend.name}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────

def tool_read_file(path: str) -> dict:
    """
    Read the contents of a file from disk.

    Args:
        path: Absolute or relative path to the file

    Returns:
        Dict with 'success', 'output', and optional 'error' keys
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"success": True, "output": content}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied reading: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_write_file(path: str, content: str) -> dict:
    """
    Write content to a file, creating parent directories as needed.

    Args:
        path: Absolute or relative path to the file
        content: Text content to write

    Returns:
        Dict with 'success', 'output', and optional 'error' keys
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"success": True, "output": f"File written successfully: {path}"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied writing: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_edit_file(path: str, search: str, replace: str) -> dict:
    """
    Replace the first occurrence of a search string in a file.

    Args:
        path: Path to the file to edit
        search: Exact text to find
        replace: Replacement text

    Returns:
        Dict with 'success', 'output', and optional 'error' keys
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        if search not in content:
            return {"success": False, "error": f"Search string not found in {path}"}
        new_content = content.replace(search, replace, 1)
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return {"success": True, "output": f"File edited successfully: {path}"}
    except FileNotFoundError:
        return {"success": False, "error": f"File not found: {path}"}
    except PermissionError:
        return {"success": False, "error": f"Permission denied: {path}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def tool_execute_code(command: str, timeout: int = 30) -> dict:
    """
    Execute a shell command and return stdout, stderr, and exit code.

    Args:
        command: Shell command string to execute
        timeout: Maximum seconds to wait (default 30)

    Returns:
        Dict with 'success', 'output', 'exit_code', and optional 'error' keys
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        combined = ""
        if result.stdout:
            combined += result.stdout
        if result.stderr:
            combined += ("\n" if combined else "") + f"[stderr]: {result.stderr}"
        success = result.returncode == 0
        return {
            "success": success,
            "output": combined.strip() if combined.strip() else "(no output)",
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Command timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def dispatch_tool(tool_call: dict) -> str:
    """
    Dispatch a parsed tool call dict to the appropriate tool function.

    Args:
        tool_call: Dict containing 'tool' key and tool-specific parameters

    Returns:
        Formatted string result to feed back to the model
    """
    tool_name = tool_call.get("tool", "")

    if tool_name == "read_file":
        path = tool_call.get("path", "")
        if not path:
            return "ERROR: 'path' is required for read_file"
        result = tool_read_file(path)

    elif tool_name == "write_file":
        path = tool_call.get("path", "")
        content = tool_call.get("content", "")
        if not path:
            return "ERROR: 'path' is required for write_file"
        result = tool_write_file(path, content)

    elif tool_name == "edit_file":
        path = tool_call.get("path", "")
        search = tool_call.get("search", "")
        replace = tool_call.get("replace", "")
        if not path or not search:
            return "ERROR: 'path' and 'search' are required for edit_file"
        result = tool_edit_file(path, search, replace)

    elif tool_name == "execute_code":
        command = tool_call.get("command", "")
        if not command:
            return "ERROR: 'command' is required for execute_code"
        result = tool_execute_code(command)

    else:
        return f"ERROR: Unknown tool '{tool_name}'. Available: read_file, write_file, edit_file, execute_code"

    if result["success"]:
        return f"TOOL RESULT [{tool_name}]:\n{result['output']}"
    else:
        return f"TOOL ERROR [{tool_name}]: {result.get('error', 'Unknown error')}"


# ─────────────────────────────────────────────
# Auto Mode Parser
# ─────────────────────────────────────────────

def parse_auto_response(response: str) -> dict:
    """
    Parse the model's response in auto mode to extract tool calls or completion signals.

    Looks for:
      - <tool_call>...</tool_call> blocks containing JSON
      - <task_complete>...</task_complete> blocks
      - <task_blocked reason="..."> tags

    Args:
        response: Raw response text from the model

    Returns:
        Dict with keys:
          - 'type': 'tool_call' | 'complete' | 'blocked' | 'text'
          - 'tool_call': parsed dict (if type == 'tool_call')
          - 'summary': completion/blocked message (if type in ['complete', 'blocked'])
          - 'raw': original response text
    """
    # Check for task_complete
    complete_match = re.search(
        r"<task_complete>(.*?)</task_complete>", response, re.DOTALL | re.IGNORECASE
    )
    if complete_match:
        return {
            "type": "complete",
            "summary": complete_match.group(1).strip(),
            "raw": response,
        }

    # Check for task_blocked
    blocked_match = re.search(
        r'<task_blocked\s+reason=["\']?(.*?)["\']?\s*/?>',
        response,
        re.DOTALL | re.IGNORECASE,
    )
    if blocked_match:
        return {
            "type": "blocked",
            "summary": blocked_match.group(1).strip(),
            "raw": response,
        }

    # Check for tool_call
    tool_match = re.search(
        r"<tool_call>(.*?)</tool_call>", response, re.DOTALL | re.IGNORECASE
    )
    if tool_match:
        raw_json = tool_match.group(1).strip()
        tool_call, parse_error = _parse_tool_json(raw_json)
        if tool_call is not None:
            return {
                "type": "tool_call",
                "tool_call": tool_call,
                "raw": response,
            }
        else:
            return {
                "type": "text",
                "raw": response,
                "parse_error": parse_error,
            }

    # Plain text response (no structured output)
    return {"type": "text", "raw": response}


def _parse_tool_json(raw_json: str) -> tuple:
    """
    Robustly parse a JSON string extracted from a tool_call block.

    Attempts multiple strategies to handle common LLM output issues:
      1. Direct json.loads (happy path)
      2. Extract first complete JSON object via brace-counting (handles trailing data)
      3. Strip markdown code fences and retry
      4. Attempt to fix common escape issues and retry

    Args:
        raw_json: The raw string content from inside <tool_call>...</tool_call>

    Returns:
        Tuple of (parsed_dict_or_None, error_message_or_None).
        On success: (dict, None). On failure: (None, error_str).
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(raw_json), None
    except json.JSONDecodeError as e1:
        first_error = str(e1)

    # Strategy 2: Extract first complete JSON object by brace counting
    # Handles "Extra data" errors where the model appended text after the JSON
    try:
        extracted = _extract_first_json_object(raw_json)
        if extracted is not None:
            return json.loads(extracted), None
    except json.JSONDecodeError:
        pass

    # Strategy 3: Strip markdown code fences (```json ... ``` or ``` ... ```)
    try:
        stripped = re.sub(r"^```(?:json)?\s*", "", raw_json.strip(), flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped.strip())
        if stripped != raw_json:
            return json.loads(stripped), None
    except json.JSONDecodeError:
        pass

    # Strategy 4: Fix unescaped newlines/tabs inside string values
    try:
        fixed = re.sub(r'(?<!\\)\n', r'\\n', raw_json)
        fixed = re.sub(r'(?<!\\)\t', r'\\t', fixed)
        return json.loads(fixed), None
    except json.JSONDecodeError:
        pass

    error_msg = f"Invalid JSON format: {first_error}\nRaw content: {raw_json[:300]}"
    return None, error_msg


def _extract_first_json_object(text: str) -> str | None:
    """
    Extract the first complete JSON object from a string using brace counting.

    Handles cases where the LLM appends extra text after a valid JSON object,
    which causes json.JSONDecodeError with 'Extra data'.

    Args:
        text: String potentially containing a JSON object followed by extra content

    Returns:
        The first complete JSON object as a string, or None if not found
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start: i + 1]
    return None


# ─────────────────────────────────────────────
# Auto Mode Runner
# ─────────────────────────────────────────────

AUTO_PREFIX = "\033[1;35m🤖 [AUTO]\033[0m"
TOOL_PREFIX = "\033[1;33m🔧 [TOOL]\033[0m"
COMPLETE_PREFIX = "\033[1;32m✅ [DONE]\033[0m"
BLOCKED_PREFIX = "\033[1;31m⚠️  [BLOCKED]\033[0m"
PAUSE_PREFIX = "\033[1;33m⏸  [PAUSE]\033[0m"


def run_auto_mode(task: str, backend_manager: BackendManager) -> None:
    """
    Run the autonomous agent loop for a given task.

    The loop:
      1. Sends the task + conversation history to the model with the auto-mode system prompt
      2. Parses the response for tool calls or completion signals
      3. Executes any tool calls and feeds results back as user observations
      4. Repeats until: task_complete, task_blocked, or max iterations reached
      5. At max iterations, pauses and asks the user whether to continue

    Args:
        task: The task description provided by the user
        backend_manager: The active BackendManager instance
    """
    print(f"\n{'═' * 60}")
    print(f"  🤖  AUTO MODE — Autonomous Agent")
    print(f"{'═' * 60}")
    print(f"  Task: {task}")
    print(f"  Max steps before pause: {AUTO_MODE_MAX_ITERATIONS}")
    print(f"{'─' * 60}\n")

    # Build the auto-mode conversation history (separate from main chat history)
    auto_history: list[dict] = [
        {"role": "user", "content": f"Task: {task}"}
    ]

    backend = backend_manager.get_active_backend()
    iteration = 0
    total_iterations = 0

    while True:
        iteration += 1
        total_iterations += 1

        print(f"\n{AUTO_PREFIX} Step {total_iterations} — Thinking...")

        # Get model response
        try:
            response = backend.chat(
                messages=auto_history,
                system_prompt=AUTO_MODE_SYSTEM_PROMPT,
                max_tokens=2048,
            )
        except RuntimeError as e:
            print(f"\033[1;31m❌ Backend error: {e}\033[0m")
            print(f"{BLOCKED_PREFIX} Auto mode aborted due to backend error.")
            return

        # Display the model's response (strip tool_call blocks for cleaner output)
        display_response = re.sub(
            r"<tool_call>.*?</tool_call>", "[tool call issued]", response,
            flags=re.DOTALL | re.IGNORECASE
        )
        display_response = re.sub(
            r"<task_complete>.*?</task_complete>", "", display_response,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()
        if display_response:
            print(f"\n{AUTO_PREFIX}")
            print(f"{display_response}\n")

        # Add assistant response to history
        auto_history.append({"role": "assistant", "content": response})

        # Parse the response
        parsed = parse_auto_response(response)

        if parsed["type"] == "complete":
            print(f"\n{COMPLETE_PREFIX} Task completed!")
            print(f"{'─' * 60}")
            print(parsed["summary"])
            print(f"{'─' * 60}\n")
            break

        elif parsed["type"] == "blocked":
            print(f"\n{BLOCKED_PREFIX} Agent is blocked: {parsed['summary']}")
            print("Please provide guidance to continue or type 'exit' to abort.")
            try:
                feedback = input("\033[1;33mYour guidance:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAuto mode aborted.")
                return
            if feedback.lower() in ("exit", "quit", "abort"):
                print_info("Auto mode aborted by user.")
                return
            # Inject user feedback and continue
            auto_history.append({"role": "user", "content": f"User guidance: {feedback}"})
            iteration = 0  # Reset iteration counter after feedback
            continue

        elif parsed["type"] == "tool_call":
            tool_call = parsed["tool_call"]
            tool_name = tool_call.get("tool", "unknown")
            print(f"\n{TOOL_PREFIX} Executing: {tool_name}")

            # Show tool parameters (sanitized for display)
            display_params = {k: v for k, v in tool_call.items() if k != "tool"}
            if "content" in display_params and len(str(display_params["content"])) > 100:
                display_params["content"] = str(display_params["content"])[:100] + "..."
            print(f"  Parameters: {json.dumps(display_params, indent=2)}")

            # Execute the tool
            tool_result = dispatch_tool(tool_call)
            print(f"\n  Result:")
            # Indent result for readability
            for line in tool_result.split("\n"):
                print(f"    {line}")

            # Feed result back as a user observation
            observation = (
                f"Tool execution result:\n{tool_result}\n\n"
                "Continue with the next step to complete the task."
            )
            auto_history.append({"role": "user", "content": observation})

        elif parsed["type"] == "text":
            # Model gave a plain text response without structured output
            if "parse_error" in parsed:
                # Display the parse warning to the user
                print(f"\033[90m  [Parse warning: {parsed['parse_error']}]\033[0m")
                # Inject the exact error into history so the LLM knows it failed
                # and must retry with a corrected, valid JSON tool call
                error_feedback = (
                    f"SYSTEM: Your previous tool call could not be executed because "
                    f"the JSON was invalid.\n"
                    f"Error: {parsed['parse_error']}\n\n"
                    f"Please retry by outputting a corrected <tool_call> block with "
                    f"valid JSON. Ensure:\n"
                    f"  - The JSON is a single object with no trailing data\n"
                    f"  - All string values use \\n for newlines (not literal newlines)\n"
                    f"  - All backslashes inside strings are escaped as \\\\\n"
                    f"  - No markdown code fences (``` or ```json) inside the <tool_call> tags"
                )
                auto_history.append({"role": "user", "content": error_feedback})
            else:
                # Plain text with no tool call and no parse error — nudge the model
                nudge = (
                    "Please use a tool to make progress on the task, or output "
                    "<task_complete> if the task is done, or <task_blocked reason=\"...\"> "
                    "if you cannot proceed."
                )
                auto_history.append({"role": "user", "content": nudge})

        # Check if we've hit the max iterations threshold
        if iteration >= AUTO_MODE_MAX_ITERATIONS:
            print(f"\n{PAUSE_PREFIX} Reached {AUTO_MODE_MAX_ITERATIONS} steps without completion.")
            print(f"{'─' * 60}")
            print("Options:")
            print("  [Enter]       — Continue for another batch of steps")
            print("  [feedback]    — Type guidance to steer the agent")
            print("  [exit/abort]  — Stop auto mode")
            print(f"{'─' * 60}")
            try:
                feedback = input(f"\033[1;33mContinue? (Enter/feedback/exit):\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAuto mode aborted.")
                return

            if feedback.lower() in ("exit", "quit", "abort"):
                print_info("Auto mode stopped by user.")
                return
            elif feedback:
                # User provided guidance — inject it
                auto_history.append({
                    "role": "user",
                    "content": f"User feedback after {total_iterations} steps: {feedback}"
                })
                print_info(f"Feedback received. Continuing with guidance...")
            else:
                print_info(f"Continuing for another {AUTO_MODE_MAX_ITERATIONS} steps...")

            # Reset per-batch iteration counter
            iteration = 0


# ─────────────────────────────────────────────
# CLI Display Helpers
# ─────────────────────────────────────────────
SEPARATOR = "─" * 60
BOT_PREFIX = "\033[1;36m🤖 Assistant:\033[0m"
USER_PREFIX = "\033[1;32mYou:\033[0m"
INFO_PREFIX = "\033[1;33mℹ️  Info:\033[0m"
ERROR_PREFIX = "\033[1;31m❌ Error:\033[0m"


def print_separator() -> None:
    """Print a visual separator line."""
    print(f"\033[90m{SEPARATOR}\033[0m")


def print_bot_response(text: str) -> None:
    """
    Print the bot's response with formatting.

    Args:
        text: The response text to display
    """
    print(f"\n{BOT_PREFIX}")
    print(f"{text}\n")
    print_separator()


def print_info(text: str) -> None:
    """
    Print an informational message.

    Args:
        text: The info text to display
    """
    print(f"{INFO_PREFIX} {text}")


def print_error(text: str) -> None:
    """
    Print an error message.

    Args:
        text: The error text to display
    """
    print(f"{ERROR_PREFIX} {text}")


def print_welcome(backend_name: str) -> None:
    """
    Print the welcome banner.

    Args:
        backend_name: Name of the active backend
    """
    print("\n" + "═" * 60)
    print("  🤖  CLI Chatbot — Powered by LLM")
    print("═" * 60)
    print(f"  Backend : {backend_name}")
    print(f"  Time    : {datetime.datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("─" * 60)
    print("  Type your message and press Enter to chat.")
    print("  Type /help to see available commands.")
    print("  Type /auto <task> to run autonomous agent mode.")
    print("═" * 60 + "\n")


def print_help() -> None:
    """Print the help message listing all available commands."""
    print("\n\033[1mAvailable Commands:\033[0m")
    for cmd, desc in COMMANDS.items():
        print(f"  \033[1;36m{cmd:<20}\033[0m {desc}")
    print()
    print("\033[1mAuto Mode Tools:\033[0m")
    print("  read_file     — Read a file from disk")
    print("  write_file    — Write content to a file")
    print("  edit_file     — Find-and-replace in a file")
    print("  execute_code  — Run a shell command")
    print()


def display_history(history: list[dict]) -> None:
    """
    Display the full conversation history.

    Args:
        history: List of message dicts with timestamp, role, and content
    """
    if not history:
        print_info("No conversation history yet.")
        return
    print(f"\n\033[1mConversation History ({len(history)} messages):\033[0m")
    print_separator()
    for msg in history:
        ts = msg.get("timestamp", "")[:19].replace("T", " ")
        role = msg["role"].capitalize()
        color = "\033[1;32m" if msg["role"] == "user" else "\033[1;36m"
        print(f"{color}[{ts}] {role}:\033[0m")
        print(f"  {msg['content']}\n")
    print_separator()


# ─────────────────────────────────────────────
# Command Handler
# ─────────────────────────────────────────────
def handle_command(
    command: str,
    history: ConversationHistory,
    backend_manager: BackendManager,
) -> bool:
    """
    Handle a slash command from the user.

    Args:
        command: The raw command string (e.g., '/help', '/backend openrouter')
        history: The active ConversationHistory instance
        backend_manager: The active BackendManager instance

    Returns:
        True if the session should exit, False otherwise
    """
    parts = command.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/exit", "/quit"):
        print_info("Goodbye! Chat session ended.")
        return True

    elif cmd == "/help":
        print_help()

    elif cmd == "/history":
        display_history(history.get_full_history())

    elif cmd == "/reset":
        history.reset()
        print_info("Conversation history has been reset.")

    elif cmd == "/backend":
        if not arg:
            print(backend_manager.status_string())
        else:
            result = backend_manager.switch_backend(arg)
            print_info(result)

    elif cmd == "/auto":
        if not arg:
            print_error("Usage: /auto <task description>")
            print_info("Example: /auto write a Python script that prints Hello World and execute it")
        else:
            run_auto_mode(arg, backend_manager)

    else:
        print_error(f"Unknown command '{cmd}'. Type /help for available commands.")

    return False


# ─────────────────────────────────────────────
# Input Validation
# ─────────────────────────────────────────────
def validate_input(text: str) -> tuple[bool, str]:
    """
    Validate and sanitize user input.

    Args:
        text: Raw user input string

    Returns:
        Tuple of (is_valid: bool, sanitized_text_or_error: str)
    """
    text = text.strip()
    if not text:
        return False, "Input cannot be empty."
    if len(text) > 8000:
        return False, "Input is too long (max 8000 characters)."
    return True, text


# ─────────────────────────────────────────────
# Main Chat Loop
# ─────────────────────────────────────────────
def run_chat(backend_manager: BackendManager) -> None:
    """
    Run the main interactive chat loop.

    Args:
        backend_manager: Configured BackendManager instance
    """
    history = ConversationHistory()
    active = backend_manager.get_active_backend()
    print_welcome(active.name)

    while True:
        try:
            user_input = input(f"{USER_PREFIX} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print_info("Session interrupted. Goodbye!")
            break

        # Handle slash commands
        if user_input.startswith("/"):
            should_exit = handle_command(user_input, history, backend_manager)
            if should_exit:
                break
            continue

        # Validate input
        valid, sanitized = validate_input(user_input)
        if not valid:
            print_error(sanitized)
            continue

        # Add user message to history
        history.add_message("user", sanitized)

        # Get response from active backend
        backend = backend_manager.get_active_backend()
        try:
            print(f"\n\033[90mThinking...\033[0m", end="\r")
            response = backend.chat(history.get_messages_for_api())
            print(" " * 20, end="\r")
            history.add_message("assistant", response)
            print_bot_response(response)

        except RuntimeError as e:
            print(" " * 20, end="\r")
            print_error(str(e))

            # Attempt fallback to the other backend
            other_backends = [
                (name, b)
                for name, b in backend_manager.backends.items()
                if name != backend_manager.active_backend_name
            ]
            for fallback_name, fallback_backend in other_backends:
                if fallback_backend.is_available():
                    print_info(
                        f"Attempting fallback to [{fallback_name}] {fallback_backend.name}..."
                    )
                    try:
                        response = fallback_backend.chat(history.get_messages_for_api())
                        history.add_message("assistant", response)
                        print_bot_response(response)
                        print_info(
                            f"Note: Response served by fallback backend [{fallback_name}]. "
                            f"Use /backend {fallback_name} to switch permanently."
                        )
                        break
                    except RuntimeError as fe:
                        print_error(f"Fallback also failed: {fe}")
            else:
                if history.get_full_history() and history.get_full_history()[-1]["role"] == "user":
                    history._history.pop()
                print_error(
                    "No backends are available. Check your connection or start llama.cpp server."
                )


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────
def main() -> None:
    """Main entry point for the CLI chatbot application."""
    import argparse

    parser = argparse.ArgumentParser(
        description="CLI Chatbot — Qwen 3.5-Flash (OpenRouter) + llama.cpp"
    )
    parser.add_argument(
        "--backend",
        choices=["openrouter", "llamacpp"],
        default=None,
        help="Force a specific backend (default: auto-select)",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL_ID",
        help=(
            f"Model identifier to use with the selected backend. "
            f"Defaults to '{OPENROUTER_DEFAULT_MODEL}' for OpenRouter. "
            f"Example: --model qwen/qwen3.5-flash-02-23"
        ),
    )
    parser.add_argument(
        "--llama-url",
        default=os.environ.get("LLAMA_CPP_URL", LLAMA_CPP_DEFAULT_URL),
        help=f"llama.cpp server URL (default: {LLAMA_CPP_DEFAULT_URL})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check backend availability and exit",
    )
    parser.add_argument(
        "--auto",
        metavar="TASK",
        default=None,
        help="Run auto mode directly with the given task and exit",
    )
    args = parser.parse_args()

    os.environ["LLAMA_CPP_URL"] = args.llama_url

    # Determine the model to use for OpenRouter
    openrouter_model = args.model if args.model else OPENROUTER_DEFAULT_MODEL

    backend_manager = BackendManager(openrouter_model=openrouter_model)

    if args.backend:
        result = backend_manager.switch_backend(args.backend)
        print_info(result)

    if args.check:
        print(backend_manager.status_string())
        for name, backend in backend_manager.backends.items():
            avail = backend.is_available()
            status = "✅ reachable" if avail else "❌ unreachable"
            print(f"  [{name}] {status}")
        return

    if not backend_manager.backends:
        print_error("No backends configured. Exiting.")
        sys.exit(1)

    # Non-interactive auto mode via CLI flag
    if args.auto:
        run_auto_mode(args.auto, backend_manager)
        return

    run_chat(backend_manager)


if __name__ == "__main__":
    main()
