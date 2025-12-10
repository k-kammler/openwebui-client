"""
Multimodal (Vision) Example for OpenWebUI Client

Demonstrates:
- Sending images to vision-capable models
- Image analysis with Qwen3 235B VL
- Different image input methods (file path, base64, URL)

Uses: jgu_entrance.jpg (image of JGU entrance)
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient, MULTIMODAL_MODELS


def main():
    print("🧪 Multimodal (Vision) Example")
    print("=" * 50)
    
    # Show available multimodal models
    print(f"\n📋 Available multimodal models: {MULTIMODAL_MODELS}")
    
    # Initialize client
    print("\n🔧 Initializing OpenWebUI client...")
    client = OpenWebUIClient()
    
    # Path to example image (relative to this script's location)
    script_dir = Path(__file__).parent
    image_path = script_dir / "jgu_entrance.jpg"
    
    # Check if image exists
    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path}")
        print("   Please ensure jgu_entrance.jpg is in the examples/ directory")
        return
    
    # Convert to string for the API
    image_path = str(image_path)
    
    print(f"📷 Using image: {image_path}")
    
    # Example 1: Basic image description
    print("\n" + "=" * 50)
    print("🖼️  Example 1: Basic Image Description")
    print("-" * 50)
    
    prompt1 = "What do you see in this image? Describe it briefly."
    print(f"Prompt: {prompt1}")
    
    response = client.chat_completion_multimodal(
        text_prompt=prompt1,
        image_path=image_path
    )
    print(f"Response: {response}")
    
    # Example 2: Specific question about the image
    print("\n" + "=" * 50)
    print("🖼️  Example 2: Specific Question")
    print("-" * 50)
    
    prompt2 = "Is this an educational institution? What details suggest this?"
    print(f"Prompt: {prompt2}")
    
    response = client.chat_completion_multimodal(
        text_prompt=prompt2,
        image_path=image_path
    )
    print(f"Response: {response}")
    
    # Example 3: Detailed analysis
    print("\n" + "=" * 50)
    print("🖼️  Example 3: Detailed Analysis")
    print("-" * 50)
    
    prompt3 = """Analyze this image and provide:
1. The type of building or location
2. Architectural style
3. Any visible text or signage
4. The overall atmosphere or mood"""
    print(f"Prompt: {prompt3}")
    
    response = client.chat_completion_multimodal(
        text_prompt=prompt3,
        image_path=image_path,
        temperature=0.3  # Lower temperature for more focused analysis
    )
    print(f"Response: {response}")
    
    # Example 4: Streaming response with image
    print("\n" + "=" * 50)
    print("🖼️  Example 4: Streaming Response with Image")
    print("-" * 50)
    
    prompt4 = "Write a short poem (4 lines) inspired by this image."
    print(f"Prompt: {prompt4}")
    print("Response: ", end="", flush=True)
    
    for chunk in client.chat_completion_multimodal(
        text_prompt=prompt4,
        image_path=image_path,
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()  # Newline after streaming
    
    # Cleanup
    client.close()
    print("\n✅ Multimodal example completed!")


if __name__ == "__main__":
    main()
