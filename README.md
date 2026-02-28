# CLI Chatbot with LLM Backends

A command-line chatbot application that supports multi-turn conversations with persistent history, powered by multiple LLM backends: **Qwen 3.5-Flash via OpenRouter API** (remote) and **llama.cpp** (local/offline).

---

## Features

- 🤖 **Multi-turn conversation** with full in-memory history for contextually aware responses
- 🌐 **OpenRouter backend** — Qwen 3.5-Flash (`qwen/qwen3.5-flash-02-23`) via the OpenRouter API
- 🖥️ **llama.cpp backend** — local offline inference via a llama.cpp HTTP server
- 🔄 **Auto mode** — autonomous agent that can read, write, edit files and execute shell commands to complete tasks
- 🔀 **Seamless backend switching** at runtime with `/backend` command
- 📜 **Session commands** — view history, reset session, switch backends, get help
- 🛡️ **Robust error handling** — rate limits, auth failures, network errors, and server unavailability all produce clear messages

---

## Requirements

- Python 3.8+
- `requests` library

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. OpenRouter (Qwen 3.5-Flash — default remote backend)

Create the config file at `~/.config/openrouter/config`:

```json
{
  "api_key": "YOUR_OPENROUTER_API_KEY"
}
```

Get your API key at [https://openrouter.ai](https://openrouter.ai).

### 2. llama.cpp (local backend — optional)

Start a llama.cpp HTTP server with a GGUF model:

```bash
llama-server -m /path/to/model.gguf --port 8080
```

The chatbot connects to `http://localhost:8080` by default. Override with the environment variable:

```bash
export LLAMA_CPP_URL=http://localhost:8080
```

---

## Usage

### Start the chatbot

```bash
python chatbot.py
```

By default, the chatbot uses the **OpenRouter backend** (Qwen 3.5-Flash) if an API key is configured, otherwise it falls back to **llama.cpp**.

### Specify a backend explicitly

```bash
# Use OpenRouter (Qwen 3.5-Flash default)
python chatbot.py --backend openrouter

# Use a custom model on OpenRouter
python chatbot.py --backend openrouter --model mistralai/mistral-7b-instruct

# Use local llama.cpp server
python chatbot.py --backend llamacpp

# Use llama.cpp at a custom URL
python chatbot.py --backend llamacpp --llama-url http://localhost:11434
```

### Auto mode (autonomous agent)

```bash
python chatbot.py --auto "Write a Python script that prints the Fibonacci sequence"
```

Or trigger auto mode from within the chat session:

```
You: /auto Refactor the file /tmp/script.py to use list comprehensions
```

---

## Session Commands

| Command | Description |
|---|---|
| `/help` | Show all available commands |
| `/history` | View the full conversation history with timestamps |
| `/reset` | Clear conversation history and start fresh |
| `/backend` | Show current backend and available backends |
| `/backend openrouter` | Switch to OpenRouter (Qwen 3.5-Flash) |
| `/backend llamacpp` | Switch to local llama.cpp backend |
| `/auto <task>` | Run autonomous agent mode for a given task |
| `/exit` or `/quit` | Exit the chat session |

---

## Backends

### OpenRouter — Qwen 3.5-Flash

| Property | Value |
|---|---|
| Default model | `qwen/qwen3.5-flash-02-23` |
| API endpoint | `https://openrouter.ai/api/v1/chat/completions` |
| Auth | Bearer token from `~/.config/openrouter/config` |
| Timeout | 60 seconds |
| Max tokens | 2048 (default) |

Handled errors:
- `401` — Invalid API key
- `402` — Insufficient credits
- `429` — Rate limit exceeded (with retry guidance)
- Network timeout / connection failure

### llama.cpp (Local)

| Property | Value |
|---|---|
| Default URL | `http://localhost:8080` |
| Endpoint | `/v1/chat/completions` |
| Override | `LLAMA_CPP_URL` environment variable |
| Timeout | 120 seconds |
| Model format | GGUF |

The chatbot checks `/health` and `/v1/models` to detect server availability.

---

## Auto Mode (Autonomous Agent)

In auto mode, the chatbot acts as an autonomous agent that can complete multi-step tasks using built-in tools:

| Tool | Description |
|---|---|
| `read_file` | Read the contents of a file |
| `write_file` | Write or overwrite a file |
| `edit_file` | Replace a specific string in a file |
| `execute_code` | Run a shell command and capture output |

The agent iterates up to **5 steps** before pausing for user feedback, preventing runaway execution.

---

## Project Structure

```
.
├── chatbot.py          # Main chatbot application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLAMA_CPP_URL` | `http://localhost:8080` | llama.cpp server URL |

---

## Example Session

```
$ python chatbot.py

╔══════════════════════════════════════════╗
║         CLI Chatbot — LLM Assistant      ║
╚══════════════════════════════════════════╝
Backend: OpenRouter (qwen/qwen3.5-flash-02-23)
Type /help for commands, /exit to quit.

You: What is the capital of France?
Assistant: The capital of France is Paris.

You: And what is it known for?
Assistant: Paris is known for the Eiffel Tower, world-class cuisine, fashion, art museums like the Louvre, and its romantic atmosphere.

You: /history
[1] 2026-02-28T15:30:00Z | user      | What is the capital of France?
[2] 2026-02-28T15:30:01Z | assistant | The capital of France is Paris.
[3] 2026-02-28T15:30:10Z | user      | And what is it known for?
[4] 2026-02-28T15:30:11Z | assistant | Paris is known for the Eiffel Tower...

You: /exit
Goodbye!
```

---

## License

MIT

---

<div align="center">

[![Made by Neo](https://img.shields.io/badge/Made%20by-Neo-6C63FF?style=for-the-badge&logo=sparkles&logoColor=white)](https://heyneo.com/)

</div>
