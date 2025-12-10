"""
Chat History Example for OpenWebUI Client

Demonstrates:
- Multi-turn conversations with history
- Saving and loading chat history
- History summary
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient


def main():
    print("🧪 Chat History Example")
    print("=" * 50)
    
    # Initialize client with history enabled
    print("\n🔧 Initializing client with history...")
    client = OpenWebUIClient(
        save_history=True,
        history_file='example_chat_history.json'
    )
    
    # Get first available model
    models = client.get_models()
    test_model = models[0]['id'] if models else "default"
    print(f"🎯 Using model: {test_model}")
    
    # Clear any existing history
    client.clear_history()
    print("\n📚 Starting new conversation...")
    
    # First turn - share information
    print("\n" + "-" * 50)
    user_msg1 = "My favorite color is blue and my favorite number is 42. Remember this!"
    print(f"👤 User: {user_msg1}")
    
    response1 = client.chat_with_history(user_msg1, model=test_model)
    print(f"🤖 Assistant: {response1}")
    
    # Second turn - test memory
    print("\n" + "-" * 50)
    user_msg2 = "What's my favorite color?"
    print(f"👤 User: {user_msg2}")
    
    response2 = client.chat_with_history(user_msg2, model=test_model)
    print(f"🤖 Assistant: {response2}")
    
    # Third turn - test memory again
    print("\n" + "-" * 50)
    user_msg3 = "And what's my favorite number?"
    print(f"👤 User: {user_msg3}")
    
    response3 = client.chat_with_history(user_msg3, model=test_model)
    print(f"🤖 Assistant: {response3}")
    
    # Show history summary
    print("\n" + "=" * 50)
    print("📊 Session Summary:")
    summary = client.get_history_summary()
    print(f"   Total messages: {summary['message_count']}")
    print(f"   User messages: {summary['user_messages']}")
    print(f"   Assistant messages: {summary['assistant_messages']}")
    print(f"   History file: {summary['history_file']}")
    
    # Show raw history
    print("\n📜 Raw History:")
    for msg in client.get_history():
        role = "👤" if msg['role'] == 'user' else "🤖"
        content_preview = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']
        print(f"   {role} [{msg['role']}]: {content_preview}")
    
    # Cleanup
    client.close()
    print("\n✅ Chat history example completed!")


if __name__ == "__main__":
    main()
