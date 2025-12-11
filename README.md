# OpenWebUI API Client

A Python client for interacting with OpenWebUI API services, specifically designed for Johannes Gutenberg University's (JGU) AI chat models hosted at [ki-chat.uni-mainz.de](https://ki-chat.uni-mainz.de).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gree.svg)](https://opensource.org/licenses/MIT)

## Features

- 🤖 **Easy API Access**: Simple interface for interacting with JGU's OpenWebUI models
- 💬 **Multiple Interfaces**: Support for simple `invoke()` and advanced `chat_completion()` methods
- 🌊 **Streaming Support**: Real-time streaming responses with `invoke_stream()` and `chat_completion_stream()`
- 📚 **Chat History**: Built-in conversation history management with file persistence
- 🔢 **Text Embeddings**: Generate embeddings with bge-m3 for semantic search and RAG applications
- 🖼️ **Multimodal Vision**: Analyze images with Qwen3 235B VL (base64, file path, or URL input)
- 💻 **Fill-in-the-Middle**: Code completion with Qwen3 Coder 30B for intelligent code generation
- 🔧 **Tool Calling**: Function calling support with all chat models
- 🌐 **Web Search**: Built-in web search via `use_web=True` parameter (optional dependency)
- ⚙️ **Auto Configuration**: Interactive setup wizard for easy configuration
- 🔐 **Secure**: API keys stored in local config files (not in code)
- ⏱️ **Rate Limiting**: Built-in rate limiting to prevent API throttling
- 🎛️ **Customizable**: Control temperature, top_p, seed, and other parameters

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/k-kammler/openwebui-client.git
cd openwebui-client
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the client:**
```bash
python openwebui_client.py
```

On first run, the interactive setup wizard will guide you through configuration.

## Quick Start

### Getting Your API Key

1. Visit [ki-chat.uni-mainz.de](https://ki-chat.uni-mainz.de)
2. Go to **Settings** (Einstellungen) → **Account** (Konto) → **API Key** (API-Schlüssel)
3. Click **Show** at the right side of the API Key field and copy the key

### Basic Usage

```python
from openwebui_client import OpenWebUIClient

# Initialize client (uses config.ini or prompts for setup)
client = OpenWebUIClient()

# Simple single-turn conversation
response = client.invoke("What is the capital of Germany?")
print(response)

# With specific model
response = client.invoke("Explain quantum computing", model="Qwen3 235B Thinking")
print(response)
```

### Advanced Usage with Message Formatting

```python
# Using chat_completion with system prompts
messages = [
    {"role": "system", "content": "You are a helpful math tutor. Be concise."},
    {"role": "user", "content": "What is 2 + 2?"}
]
response = client.chat_completion(messages, model="GPT OSS 120B")
print(response)
```

### Web Search (use_web)

```python
# Enable web search for up-to-date information
# Requires: pip install ddgs

# Simple web-enabled query
response = client.invoke("What are the latest AI news?", use_web=True)
print(response)

# With chat_completion
messages = [{"role": "user", "content": "What is the current weather in Berlin?"}]
response = client.chat_completion(messages, use_web=True)
print(response)

# With history
response = client.chat_with_history("Search for Python 3.13 new features", use_web=True)
```

### Streaming Responses

```python
# Simple streaming
print("Response: ", end="", flush=True)
for chunk in client.invoke_stream("Tell me a short story"):
    print(chunk, end="", flush=True)
print()

# Advanced streaming with message formatting
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Count from 1 to 5"}
]
for chunk in client.chat_completion_stream(messages):
    print(chunk, end="", flush=True)
```

### Chat with History

```python
# Create client with history enabled
client = OpenWebUIClient(
    save_history=True, 
    history_file='my_conversation.json'
)

# Have a multi-turn conversation
response1 = client.chat_with_history("My favorite color is blue. Remember this!")
print(response1)

response2 = client.chat_with_history("What's my favorite color?")
print(response2)  # Will remember: blue

# View conversation summary
summary = client.get_history_summary()
print(f"Messages: {summary['message_count']}")
```

### Text Embeddings

```python
# Generate embeddings for semantic search or RAG
from openwebui_client import OpenWebUIClient

client = OpenWebUIClient()

# Single text
result = client.create_embeddings("Machine learning is fascinating")
embedding = result['data'][0]['embedding']  # Vector of 1024 dimensions

# Batch processing (more efficient)
texts = ["First text", "Second text", "Third text"]
result = client.create_embeddings(texts)
embeddings = [item['embedding'] for item in result['data']]

# Use for semantic search, clustering, or RAG
```

### Multimodal Vision

```python
# Analyze images with vision models
client = OpenWebUIClient()

# From file path
response = client.chat_completion_multimodal(
    text_prompt="What do you see in this image?",
    image_path="photo.jpg"
)
print(response)

# From base64 string
response = client.chat_completion_multimodal(
    text_prompt="Describe this image",
    image_base64="iVBORw0KGgoAAAANS..."
)

# From URL
response = client.chat_completion_multimodal(
    text_prompt="What's in this picture?",
    image_url="https://example.com/image.jpg"
)

# Streaming also supported
for chunk in client.chat_completion_multimodal(
    text_prompt="Describe this image",
    image_path="photo.jpg",
    stream=True
):
    print(chunk, end="", flush=True)
```

### Fill-in-the-Middle (Code Completion)

```python
# Intelligent code completion
client = OpenWebUIClient()

# Complete a function body
completion = client.fill_in_the_middle(
    prefix="def calculate_fibonacci(n):\n    ",
    suffix="\n    return result",
    max_tokens=200
)
print(completion)

# Streaming code generation
for chunk in client.fill_in_the_middle(
    prefix="class Calculator:\n    def add(self, a, b):\n        ",
    suffix="\n\n    def subtract(self, a, b):",
    stream=True
):
    print(chunk, end="", flush=True)
```

### Tool Calling / Function Calling

```python
# Define tools for the model to use
client = OpenWebUIClient()

# Define a calculator tool
calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Math expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    }
}

# Implement the actual function
def calculator(expression: str) -> str:
    import math
    result = eval(expression)  # Simplified; use safe eval in production
    return str(result)

# Let the model decide when to use tools
response = client.invoke_with_tools(
    "What is the square root of 144 plus 5 to the power of 3?",
    tools=[calculator_tool],
    tool_functions={"calculator": calculator}
)
print(response)

# Multi-turn conversation with tools
messages = [{"role": "user", "content": "Calculate 15% tip on $85"}]
response = client.chat_with_tools(
    messages,
    tools=[calculator_tool],
    tool_functions={"calculator": calculator}
)
```

### Rate Limiting

```python
# Set rate limit (e.g., 1 second between requests)
client = OpenWebUIClient(request_delay=1.0)

# Or update it later
client.set_rate_limit(2.0)  # 2 seconds between requests
```

## Model Capabilities

Different models support different features:

| Model | Chat | Embeddings | Vision | FIM | Tool Calling |
|-------|------|------------|--------|-----|--------------|
| **Qwen3 235B Thinking** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Qwen3 235B VL** | ✅ | ❌ | ✅ | ❌ | ✅ |
| **GPT OSS 120B** | ✅ | ❌ | ❌ | ❌ | ✅ |
| **Qwen3 Coder 30B** | ✅ | ❌ | ❌ | ✅ | ✅ |
| **bge-m3** | ❌ | ✅ | ❌ | ❌ | ❌ |

The client automatically validates model capabilities and provides helpful error messages.

## Using with Other OpenWebUI Instances

While this client is designed for JGU's OpenWebUI instance, it can be used with **any OpenWebUI API-compatible service**. Simply configure the `base_url` to point to your OpenWebUI instance:

```python
# Using with a different OpenWebUI instance
client = OpenWebUIClient(
    api_key="your-api-key",
    base_url="https://your-openwebui-instance.com"
)

# Or update config.ini with your instance URL
```

The client is compatible with any OpenWebUI deployment that follows the standard OpenAI-compatible API format.

## Configuration

### Automatic Configuration

On first run, the client will prompt you for:
- API key
- Base URL (defaults to `https://ki-chat.uni-mainz.de`)

Configuration is saved to `config.ini` in the current directory.

### Manual Configuration

Create a `config.ini` file:

```ini
# OpenWebUI Configuration File
# Keep this file secure and don't share it!

[openwebui]
api_key = your-api-key-here
base_url = https://ki-chat.uni-mainz.de
```

### Programmatic Configuration

```python
# Override config file settings
client = OpenWebUIClient(
    api_key="your-api-key",
    base_url="https://ki-chat.uni-mainz.de",
    default_model="Qwen3 235B Thinking",
    default_temperature=0.7,
    default_top_p=0.95,
    logging_level=logging.INFO
)
```

## API Reference

### Module Constants

The following constants are available for import:

```python
from openwebui_client import (
    EMBEDDING_MODELS,        # ['bge-m3']
    MULTIMODAL_MODELS,       # ['Qwen3 235B VL']
    FIM_MODELS,              # ['Qwen3 Coder 30B']
    TOOL_CALLING_MODELS,     # ['GPT OSS 120B', 'Qwen3 235B Thinking', 'Qwen3 235B VL', 'Qwen3 Coder 30B']
    MAX_TOOL_ROUNDS,         # 10 (default max rounds for tool calling)
    WEB_SEARCH_AVAILABLE     # True/False (whether ddgs package is installed)
)
```

### OpenWebUIClient

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | str | None | API key (from config if None) |
| `base_url` | str | None | Base URL (from config if None) |
| `default_model` | str | "Qwen3 235B Thinking" | Default model to use |
| `default_temperature` | float | 0.7 | Sampling temperature (0.0-2.0) |
| `default_top_p` | float | 0.95 | Nucleus sampling (0.0-1.0) |
| `default_seed` | int | None | Random seed for reproducibility |
| `logging_level` | int | logging.WARNING | Logging level |
| `save_history` | bool | False | Enable chat history |
| `history_file` | str | 'chat_history.json' | History file path |
| `request_delay` | float | 0.0 | Seconds between requests |

#### Main Methods

**`invoke(prompt, model=None, use_web=False, **kwargs)`**
- Simple single-turn conversation
- Args: `use_web` enables built-in web search for current information
- Returns: `str` - Assistant's response

**`invoke_stream(prompt, model=None, use_web=False, **kwargs)`**
- Streaming single-turn conversation
- Args: `use_web` enables web search (disables streaming if true)
- Yields: `str` - Response chunks

**`chat_completion(messages, model=None, temperature=None, top_p=None, seed=None, stream=False, use_web=False, **kwargs)`**
- Advanced chat with message formatting
- Args: `use_web` enables built-in web search for current information
- Returns: `str` - Assistant's response

**`chat_completion_stream(messages, model=None, temperature=None, top_p=None, seed=None, **kwargs)`**
- Streaming chat with message formatting
- Yields: `str` - Response chunks

**`chat_with_history(user_message, model=None, include_system_prompt=None, use_web=False, **kwargs)`**
- Chat with automatic history management
- Args: `use_web` enables built-in web search for current information
- Returns: `str` - Assistant's response

**`get_models()`**
- Get list of available models
- Returns: `List[Dict]` - Model information

**`create_embeddings(input, model=None)`**
- Generate text embeddings for semantic search or RAG
- Args: `input` (str or List[str]), `model` (defaults to 'bge-m3')
- Returns: `Dict` - Embeddings data with vectors

**`chat_completion_multimodal(text_prompt, image_path=None, image_base64=None, image_url=None, model=None, stream=False, **kwargs)`**
- Send image with text prompt to vision model
- Args: `text_prompt` (str), one of the image parameters, `model` (defaults to 'Qwen3 235B VL')
- Returns: `str` or generator - Model's response

**`fill_in_the_middle(prefix, suffix, model=None, max_tokens=512, stream=False, **kwargs)`**
- Complete code between prefix and suffix
- Args: `prefix` (str), `suffix` (str), `model` (defaults to 'Qwen3 Coder 30B')
- Returns: `str` or generator - Generated code

**`chat_with_tools(messages, tools, tool_functions, model=None, max_tool_rounds=None, **kwargs)`**
- Chat with automatic tool execution
- Args: `messages` (list), `tools` (list of tool definitions), `tool_functions` (dict mapping names to callables), `max_tool_rounds` (defaults to MAX_TOOL_ROUNDS=10)
- Returns: `str` - Final response after tool execution

**`invoke_with_tools(prompt, tools, tool_functions, model=None, system_prompt=None, **kwargs)`**
- Simple tool-calling interface
- Args: `prompt` (str), `tools` (list), `tool_functions` (dict)
- Returns: `str` - Response after tool execution

**`get_history_summary()`**
- Get chat session statistics
- Returns: `Dict` - Message counts and session info

**`clear_history()`**
- Clear conversation history

**`set_rate_limit(delay_seconds)`**
- Update rate limiting delay

## Examples

The `examples/` directory contains comprehensive demonstrations of all features:

### Running Examples

```bash
# Run from project root
python examples/basic_usage.py
python examples/chat_history.py
python examples/embeddings_example.py
python examples/multimodal_example.py
python examples/fim_example.py
python examples/tool_calling_example.py
```

### Available Examples

**`basic_usage.py`** - Getting started
- Listing available models
- Simple invoke() usage
- Chat completion with system prompts
- Streaming responses

**`chat_history.py`** - Multi-turn conversations
- Maintaining conversation context
- Saving/loading history
- History management

**`embeddings_example.py`** - Text embeddings and RAG
- Single and batch embeddings
- Semantic similarity calculation
- PDF document processing
- Map-reduce summarization
- RAG (Retrieval-Augmented Generation) demo
- Demonstrates how embeddings solve context window limits

**`multimodal_example.py`** - Vision capabilities
- Image analysis with Qwen3 235B VL
- Different image input methods (file, base64, URL)
- Image description and Q&A
- Streaming vision responses
- Uses `jgu_entrance.jpg` as example

**`fim_example.py`** - Code completion
- Fill-in-the-middle with Qwen3 Coder 30B
- Function body generation
- Class method completion
- Various code patterns
- Streaming code generation

**`tool_calling_example.py`** - Function calling
- Define custom tools for the LLM
- Implement web search, calculator, weather tools
- Model intelligently decides which tool to use (and when NOT to use them)
- Demonstrates that web search is only used for current/recent information
- Multi-step tool calling workflows
- Using `use_web=True` parameter for simplified web search

### Quick Test

The main file includes a simple connectivity test:

```bash
python openwebui_client.py
```

This verifies your configuration and API connectivity.

## Security Notes

⚠️ **Important**: 
- Never commit `config.ini` to version control (it's in `.gitignore`)
- Keep your API key secure
- Don't share your API key in code or screenshots

## Requirements

- Python 3.8+
- requests >= 2.31.0
- PyPDF2 >= 3.0.0 (for PDF examples)

**Optional:**
- ddgs >= 0.3.0 (for `use_web=True` web search feature)

Install core dependencies:
```bash
pip install -r requirements.txt
```

Install with web search support:
```bash
pip install ddgs
```

## Author

**Kevin Kammler**  
Johannes Gutenberg University Mainz

## Version

1.2.0 - December 11, 2025

### Changelog

**v1.2.0** (December 11, 2025)
- Added `use_web` parameter for built-in web search across all chat methods
- Added `MAX_TOOL_ROUNDS` constant (default: 10)
- Extended tool calling support to all chat models (Qwen3 235B VL, Qwen3 Coder 30B)
- Switched to `ddgs` package for web search (from deprecated `duckduckgo-search`)
- Improved tool calling with intelligent web search usage (only for current/recent info)
- Made web search package optional
- Added tool_calling_example.py for function calling

**v1.1.0** (December 10, 2025)
- Added embeddings support with bge-m3 model
- Added multimodal vision capabilities with Qwen3 235B VL
- Added fill-in-the-middle (FIM) code completion with Qwen3 Coder 30B
- Created comprehensive examples directory with separate example files
- Added embeddings_example.py with RAG demonstration
- Added multimodal_example.py for vision tasks
- Added fim_example.py for code completion

**v1.0.0** (Initial release)
- Core API client functionality (invoke, chat_completion, streaming)
- Chat history management
- Configuration system with interactive setup

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/k-kammler/OpenWebUIClient).

## Acknowledgments

Built for use with Johannes Gutenberg University's OpenWebUI instance at [ki-chat.uni-mainz.de](https://ki-chat.uni-mainz.de).
