"""
Fill-in-the-Middle (FIM) Example for OpenWebUI Client

Demonstrates:
- Code completion using Qwen3 Coder 30B
- Fill-in-the-middle for various code patterns
- Practical code generation scenarios
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient, FIM_MODELS


def main():
    print("🧪 Fill-in-the-Middle (FIM) Example")
    print("=" * 50)
    
    # Show available FIM models
    print(f"\n📋 Available FIM models: {FIM_MODELS}")
    
    # Initialize client
    print("\n🔧 Initializing OpenWebUI client...")
    client = OpenWebUIClient()
    
    # Example 1: Simple function body
    print("\n" + "=" * 50)
    print("💻 Example 1: Hello World Function")
    print("-" * 50)
    
    prefix1 = '''def hello_world():
    """Print a greeting message."""
'''
    suffix1 = '''

# Call the function
hello_world()'''
    
    print("Prefix:")
    print(prefix1)
    print("Suffix:")
    print(suffix1)
    print("\n🔄 Generating fill-in...")
    
    completion1 = client.fill_in_the_middle(
        prefix=prefix1,
        suffix=suffix1,
        max_tokens=50
    )
    print(f"\n✅ Generated code:\n{completion1}")
    
    # Example 2: Mathematical function
    print("\n" + "=" * 50)
    print("💻 Example 2: Calculate Circle Area")
    print("-" * 50)
    
    prefix2 = '''import math

def calculate_circle_area(radius: float) -> float:
    """Calculate the area of a circle given its radius."""
'''
    suffix2 = '''
    return area

# Test
print(calculate_circle_area(5))  # Expected: ~78.54'''
    
    print("Prefix:")
    print(prefix2)
    print("Suffix:")
    print(suffix2)
    print("\n🔄 Generating fill-in...")
    
    completion2 = client.fill_in_the_middle(
        prefix=prefix2,
        suffix=suffix2,
        max_tokens=100
    )
    print(f"\n✅ Generated code:\n{completion2}")
    
    # Example 3: List processing
    print("\n" + "=" * 50)
    print("💻 Example 3: Filter Even Numbers")
    print("-" * 50)
    
    prefix3 = '''def filter_even_numbers(numbers: list) -> list:
    """Return only the even numbers from the input list."""
'''
    suffix3 = '''
    return result

# Test
print(filter_even_numbers([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))'''
    
    print("Prefix:")
    print(prefix3)
    print("Suffix:")
    print(suffix3)
    print("\n🔄 Generating fill-in...")
    
    completion3 = client.fill_in_the_middle(
        prefix=prefix3,
        suffix=suffix3,
        max_tokens=100
    )
    print(f"\n✅ Generated code:\n{completion3}")
    
    # Example 4: Class method
    print("\n" + "=" * 50)
    print("💻 Example 4: Class Method")
    print("-" * 50)
    
    prefix4 = '''class Rectangle:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
    
    def calculate_perimeter(self) -> float:
        """Calculate the perimeter of the rectangle."""
'''
    suffix4 = '''
    
    def calculate_area(self) -> float:
        """Calculate the area of the rectangle."""
        return self.width * self.height

# Test
rect = Rectangle(5, 3)
print(f"Perimeter: {rect.calculate_perimeter()}")  # Expected: 16'''
    
    print("Prefix:")
    print(prefix4)
    print("Suffix:")
    print(suffix4)
    print("\n🔄 Generating fill-in...")
    
    completion4 = client.fill_in_the_middle(
        prefix=prefix4,
        suffix=suffix4,
        max_tokens=100
    )
    print(f"\n✅ Generated code:\n{completion4}")
    
    # Example 5: Streaming FIM
    print("\n" + "=" * 50)
    print("💻 Example 5: Streaming FIM - Fibonacci")
    print("-" * 50)
    
    prefix5 = '''def fibonacci(n: int) -> list:
    """Generate the first n Fibonacci numbers."""
'''
    suffix5 = '''
    return fib_sequence

# Generate first 10 Fibonacci numbers
print(fibonacci(10))'''
    
    print("Prefix:")
    print(prefix5)
    print("Suffix:")
    print(suffix5)
    print("\n🔄 Generating fill-in (streaming)...")
    print("\n✅ Generated code:")
    
    for chunk in client.fill_in_the_middle(
        prefix=prefix5,
        suffix=suffix5,
        max_tokens=200,
        stream=True
    ):
        print(chunk, end="", flush=True)
    print()  # Newline after streaming
    
    # Cleanup
    client.close()
    print("\n✅ FIM example completed!")


if __name__ == "__main__":
    main()
