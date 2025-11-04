# OpenWebUI API Client

A Python client for interacting with OpenWebUI API services, specifically designed for Johannes Gutenberg University's (JGU) AI chat models hosted at [ki-chat.uni-mainz.de](https://ki-chat.uni-mainz.de).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-gree.svg)](https://opensource.org/licenses/MIT)

## Features

- 🤖 **Easy API Access**: Simple interface for interacting with JGU's OpenWebUI models
- 💬 **Multiple Interfaces**: Support for simple `invoke()` and advanced `chat_completion()` methods
- 🌊 **Streaming Support**: Real-time streaming responses with `invoke_stream()` and `chat_completion_stream()`
- 📚 **Chat History**: Built-in conversation history management with file persistence
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

### Rate Limiting

```python
# Set rate limit (e.g., 1 second between requests)
client = OpenWebUIClient(request_delay=1.0)

# Or update it later
client.set_rate_limit(2.0)  # 2 seconds between requests
```

## Available Models

The JGU OpenWebUI instance provides several models:

- **Auto**: Automatically selects an appropriate model
- **Qwen3 235B Thinking**: Large reasoning model
- **Qwen3 235B VL**: Vision-enabled model
- **GPT OSS 120B**: General-purpose assistant with reasoning
- **Qwen3 Coder 30B**: Specialized coding assistant
- **bge-m3**: Embedding model for text processing

```python
# List all available models
client = OpenWebUIClient()
models = client.get_models()
for model in models:
    print(f"- {model['id']}: {model.get('name', 'N/A')}")
```

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

**`get_history_summary()`**
- Get chat session statistics
- Returns: `Dict` - Message counts and session info

**`clear_history()`**
- Clear conversation history

**`set_rate_limit(delay_seconds)`**
- Update rate limiting delay

## Examples

See the `if __name__ == "__main__":` section in `openwebui_client.py` for comprehensive examples including:
- Model listing
- Simple invoke usage
- Advanced chat completion with system prompts
- Multi-turn conversations with history
- Streaming responses (simple and advanced)

Run the examples:
```bash
python openwebui_client.py
```

## Security Notes

⚠️ **Important**: 
- Never commit `config.ini` to version control (it's in `.gitignore`)
- Keep your API key secure
- Don't share your API key in code or screenshots

## Requirements

- Python 3.8+
- requests >= 2.31.0

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
