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
- ⚙️ **Auto Configuration**: Interactive setup wizard for easy configuration
- 🔐 **Secure**: API keys stored in local config files (not in code)
- ⏱️ **Rate Limiting**: Built-in rate limiting to prevent API throttling
- 🎛️ **Customizable**: Control temperature, top_p, seed, and other parameters

## Installation

1. **Clone the repository:**
```bash
git clone https://github.com/k-kammler/OpenWebUIClient.git
cd OpenWebUIClient
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

### Rate Limiting

```python
# Set rate limit (e.g., 1 second between requests)
client = OpenWebUIClient(request_delay=1.0)

# Or update it later
client.set_rate_limit(2.0)  # 2 seconds between requests
```

## Model Capabilities

Different models support different features:

| Model | Chat | Embeddings | Vision | Fill-in-the-Middle |
|-------|------|------------|--------|--------------------|
| **Qwen3 235B Thinking** | ✅ | ❌ | ❌ | ❌ |
| **Qwen3 235B VL** | ✅ | ❌ | ✅ | ❌ |
| **GPT OSS 120B** | ✅ | ❌ | ❌ | ❌ |
| **Qwen3 Coder 30B** | ✅ | ❌ | ❌ | ✅ |
| **bge-m3** | ❌ | ✅ | ❌ | ❌ |

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

**`invoke(prompt, model=None, **kwargs)`**
- Simple single-turn conversation
- Returns: `str` - Assistant's response

**`invoke_stream(prompt, model=None, **kwargs)`**
- Streaming single-turn conversation
- Yields: `str` - Response chunks

**`chat_completion(messages, model=None, temperature=None, top_p=None, seed=None, stream=False, **kwargs)`**
- Advanced chat with message formatting
- Returns: `str` - Assistant's response

**`chat_completion_stream(messages, model=None, temperature=None, top_p=None, seed=None, **kwargs)`**
- Streaming chat with message formatting
- Yields: `str` - Response chunks

**`chat_with_history(user_message, model=None, include_system_prompt=None, **kwargs)`**
- Chat with automatic history management
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

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Author

**Kevin Kammler**  
Johannes Gutenberg University Mainz

## Version

1.0.0 - September 9, 2025

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or contributions, please open an issue on the [GitHub repository](https://github.com/k-kammler/OpenWebUIClient).

## Acknowledgments

Built for use with Johannes Gutenberg University's OpenWebUI instance at [ki-chat.uni-mainz.de](https://ki-chat.uni-mainz.de).
