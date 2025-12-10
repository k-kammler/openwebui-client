"""
Basic Usage Examples for OpenWebUI Client

Demonstrates:
- Initializing the client
- Listing available models
- Simple invoke (single-turn)
- Chat completion with system prompt
- Streaming responses
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient


def main():
    print("🧪 Basic Usage Examples")
    print("=" * 50)
    
    # Initialize client
    print("\n🔧 Initializing OpenWebUI client...")
    client = OpenWebUIClient()
    
    print(f"📡 Connected to: {client.base_url}")
    print(f"🔑 API key: {'***' + client.api_key[-4:] if len(client.api_key) > 4 else '***'}")
    
    # Get available models
    print("\n📋 Available Models:")
    models = client.get_models()
    for i, model in enumerate(models):
        print(f"   {i}. {model.get('id', 'Unknown')}")
    
    # Use first model for examples
    test_model = models[0]['id'] if models else "default"
    print(f"\n🎯 Using model: {test_model}")
    
    # Example 1: Simple invoke
    print("\n" + "=" * 50)
    print("💬 Example 1: Simple Invoke (single-turn)")
    print("-" * 50)
    
    prompt1 = "What is the capital of Germany? Answer in one sentence."
    print(f"Prompt: {prompt1}")
    
    response = client.invoke(prompt1, model=test_model)
    print(f"Response: {response}")
    
    # Example 2: Chat completion with system prompt
    print("\n" + "=" * 50)
    print("💬 Example 2: Chat Completion with System Prompt")
    print("-" * 50)
    
    messages = [
        {"role": "system", "content": "You are a helpful math tutor. Be concise."},
        {"role": "user", "content": "What is 2 + 2?"}
    ]
    print(f"System: {messages[0]['content']}")
    print(f"User: {messages[1]['content']}")
    
    response = client.chat_completion(messages, model=test_model)
    print(f"Response: {response}")
    
    # Example 3: Streaming response
    print("\n" + "=" * 50)
    print("🌊 Example 3: Streaming Response")
    print("-" * 50)
    
    prompt3 = "Count from 1 to 5, one number per line."
    print(f"Prompt: {prompt3}")
    print("Response: ", end="", flush=True)
    
    for chunk in client.invoke_stream(prompt3, model=test_model):
        print(chunk, end="", flush=True)
    print()  # Newline after streaming
    
    # Cleanup
    client.close()
    print("\n✅ Basic examples completed!")


if __name__ == "__main__":
    main()
