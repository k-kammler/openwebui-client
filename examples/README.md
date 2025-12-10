# OpenWebUI Client Examples

This directory contains example scripts demonstrating the various features of the OpenWebUI Client.

## Prerequisites

1. Ensure you have configured your API key in `config.ini` (in the parent directory)
2. Install requirements: `pip install -r ../requirements.txt`

## Example Scripts

### 1. Basic Usage (`basic_usage.py`)
Demonstrates fundamental client operations:
- Initializing the client
- Listing available models
- Simple invoke (single-turn conversations)
- Chat completion with system prompts
- Streaming responses

```bash
cd examples
python basic_usage.py
```

### 2. Chat History (`chat_history.py`)
Shows multi-turn conversation handling:
- Maintaining conversation context
- Saving/loading chat history to file
- History summary and management

```bash
cd examples
python chat_history.py
```

### 3. Embeddings (`embeddings_example.py`)
Demonstrates text embedding generation:
- Single and batch text embeddings
- Semantic similarity calculation
- Document retrieval simulation

**Model used:** `bge-m3`

```bash
cd examples
python embeddings_example.py
```

### 4. Multimodal/Vision (`multimodal_example.py`)
Shows image analysis capabilities:
- Sending images to vision models
- Various analysis prompts
- Streaming responses with images

**Model used:** `Qwen3 235B VL`  
**Requires:** `jgu_entrance.jpg` (included)

```bash
cd examples
python multimodal_example.py
```

### 5. Fill-in-the-Middle (`fim_example.py`)
Demonstrates code completion:
- Function body generation
- Code infilling between prefix and suffix
- Various code patterns (functions, classes, algorithms)

**Model used:** `Qwen3 Coder 30B`

```bash
cd examples
python fim_example.py
```

## Included Files

- `jgu_entrance.jpg` - Example image for multimodal demonstrations
- `jgu_introduction.pdf` - Example document for embedding demonstrations

## Running All Examples

To run all examples sequentially:

```bash
cd examples
python basic_usage.py
python chat_history.py
python embeddings_example.py
python multimodal_example.py
python fim_example.py
```

## Notes

- All examples use the API key from `../config.ini`
- Examples automatically select appropriate models based on capability
- Streaming examples show real-time output
- Error handling demonstrates proper exception management
