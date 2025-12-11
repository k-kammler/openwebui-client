"""
Tool Calling / Function Calling Example for OpenWebUI Client

Demonstrates:
- Defining custom tools for the LLM
- Implementing a web search tool
- The model deciding when and how to use tools
- Multi-step tool calling workflows

The LLM doesn't execute tools - it decides WHICH tool to call and with WHAT arguments.
Your code executes the actual functions and returns results to the LLM.
"""

import sys
from pathlib import Path

# Add parent directory to Python path so we can import openwebui_client
# This allows the example to run from any directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from openwebui_client import OpenWebUIClient, TOOL_CALLING_MODELS

# Try to import web search library
try:
    from ddgs import DDGS
    SEARCH_AVAILABLE = True
except ImportError:
    SEARCH_AVAILABLE = False


# =============================================================================
# TOOL DEFINITIONS (JSON Schema format - tells the LLM what tools are available)
# =============================================================================

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for CURRENT or RECENT information only. Use ONLY when the user explicitly asks about: (1) recent news or events from 2024-2025, (2) current prices, stock values, or statistics that change frequently, (3) information about very recent releases, updates, or announcements, (4) real-time data like weather or sports scores. Do NOT use for general knowledge questions, historical facts, definitions, explanations, or anything you already know from training.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return (default: 3, max: 10)"
                }
            },
            "required": ["query"]
        }
    }
}

CALCULATOR_TOOL = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Perform mathematical calculations. Use this for any math operations.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g., '2 + 2' or 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    }
}

GET_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather in a given location (simulated for demo)",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and country, e.g., 'Berlin, Germany'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit (default: celsius)"
                }
            },
            "required": ["location"]
        }
    }
}


# =============================================================================
# TOOL IMPLEMENTATIONS (Actual Python functions that do the work)
# =============================================================================

def web_search(query: str, num_results: int = 3) -> str:
    """
    Search the web using DuckDuckGo.
    
    Args:
        query: Search query
        num_results: Number of results to return
        
    Returns:
        Formatted search results
    """
    if not SEARCH_AVAILABLE:
        return "Web search is not available. Install with: pip install ddgs"
    
    num_results = min(num_results, 10)  # Cap at 10
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=num_results))
        
        if not results:
            return f"No results found for: {query}"
        
        # Format results
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            body = result.get('body', 'No description')
            href = result.get('href', '')
            formatted.append(f"{i}. {title}\n   {body}\n   URL: {href}")
        
        return "\n\n".join(formatted)
    
    except Exception as e:
        return f"Search error: {str(e)}"


def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    
    Args:
        expression: Math expression to evaluate
        
    Returns:
        Result or error message
    """
    import math
    
    # Safe evaluation with limited functions
    allowed_names = {
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'pow': pow,
        'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos,
        'tan': math.tan, 'log': math.log, 'log10': math.log10,
        'exp': math.exp, 'pi': math.pi, 'e': math.e
    }
    
    try:
        # Replace common symbols
        expression = expression.replace('^', '**')
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Calculation error: {str(e)}"


def get_weather(location: str, unit: str = "celsius") -> str:
    """
    Get weather for a location (simulated for demo).
    In a real app, you'd call a weather API like OpenWeatherMap.
    
    Args:
        location: City and country
        unit: Temperature unit
        
    Returns:
        Weather information
    """
    # Simulated weather data for demo
    import random
    
    temp = random.randint(10, 30) if unit == "celsius" else random.randint(50, 86)
    unit_symbol = "°C" if unit == "celsius" else "°F"
    conditions = random.choice(["sunny", "partly cloudy", "cloudy", "light rain"])
    humidity = random.randint(40, 80)
    
    return f"""Weather in {location}:
Temperature: {temp}{unit_symbol}
Conditions: {conditions}
Humidity: {humidity}%
(Note: This is simulated data for demonstration)"""


# =============================================================================
# MAIN EXAMPLE
# =============================================================================

def main():
    print("🧪 Tool Calling / Function Calling Example")
    print("=" * 60)
    
    # Show available tool-calling models
    print(f"\n📋 Models supporting tool calling: {TOOL_CALLING_MODELS}")
    
    # Check web search availability
    if SEARCH_AVAILABLE:
        print("✅ Web search available (ddgs installed)")
    else:
        print("⚠️  Web search unavailable. Install with: pip install ddgs")
    
    # Initialize client
    print("\n🔧 Initializing OpenWebUI client...")
    client = OpenWebUIClient()
    
    # Define all tools and their implementations
    all_tools = [WEB_SEARCH_TOOL, CALCULATOR_TOOL, GET_WEATHER_TOOL]
    tool_functions = {
        "web_search": web_search,
        "calculator": calculator,
        "get_weather": get_weather
    }
    
    # Example 1: Calculator (always works)
    print("\n" + "=" * 60)
    print("🔢 Example 1: Calculator Tool")
    print("-" * 60)
    
    prompt1 = "What is the square root of 144 plus 5 to the power of 3?"
    print(f"Prompt: {prompt1}")
    print("\n🔄 Model is deciding which tool to use...")
    
    response1 = client.invoke_with_tools(
        prompt1,
        tools=[CALCULATOR_TOOL],
        tool_functions={"calculator": calculator}
    )
    print(f"\n✅ Response:\n{response1}")
    
    # Example 2: Weather (simulated)
    print("\n" + "=" * 60)
    print("🌤️ Example 2: Weather Tool (Simulated)")
    print("-" * 60)
    
    prompt2 = "What's the weather like in Tokyo, Japan?"
    print(f"Prompt: {prompt2}")
    print("\n🔄 Model is deciding which tool to use...")
    
    response2 = client.invoke_with_tools(
        prompt2,
        tools=[GET_WEATHER_TOOL],
        tool_functions={"get_weather": get_weather}
    )
    print(f"\n✅ Response:\n{response2}")
    
    # Example 3: Web Search - queries that SHOULD trigger search
    if SEARCH_AVAILABLE:
        print("\n" + "=" * 60)
        print("🔍 Example 3: Web Search - Recent News (SHOULD search)")
        print("-" * 60)
        
        # This should trigger web search - asking for recent/current news
        prompt3 = "What are the latest developments in AI from December 2025?"
        print(f"Prompt: {prompt3}")
        print("\n🔄 Model should use web search for current news...")
        
        response3 = client.invoke_with_tools(
            prompt3,
            tools=[WEB_SEARCH_TOOL],
            tool_functions={"web_search": web_search}
        )
        print(f"\n✅ Response:\n{response3[:500]}..." if len(response3) > 500 else f"\n✅ Response:\n{response3}")
        
        print("\n" + "=" * 60)
        print("🔍 Example 4: General Knowledge (should NOT search)")
        print("-" * 60)
        
        # This should NOT trigger web search - general knowledge
        prompt4 = "What is photosynthesis?"
        print(f"Prompt: {prompt4}")
        print("\n🔄 Model should answer directly without searching...")
        
        response4 = client.invoke_with_tools(
            prompt4,
            tools=[WEB_SEARCH_TOOL],
            tool_functions={"web_search": web_search}
        )
        print(f"\n✅ Response:\n{response4[:500]}..." if len(response4) > 500 else f"\n✅ Response:\n{response4}")
    
    # Example 5: Using use_web parameter (simpler API)
    print("\n" + "=" * 60)
    print("🌐 Example 5: Using use_web=True Parameter")
    print("-" * 60)
    print("The use_web parameter provides a simpler way to enable web search.")
    print("It automatically adds the web search tool when needed.\n")
    
    # Query that benefits from web search
    prompt5a = "What are the current trending topics on GitHub in December 2025?"
    print(f"Prompt (with use_web=True): {prompt5a}")
    print("🔄 Processing with web search enabled...")
    
    response5a = client.invoke(prompt5a, use_web=True)
    print(f"\n✅ Response:\n{response5a[:400]}..." if len(response5a) > 400 else f"\n✅ Response:\n{response5a}")
    
    # Query that doesn't need web search
    print("\n" + "-" * 40)
    prompt5b = "Explain the concept of recursion in programming."
    print(f"Prompt (with use_web=True): {prompt5b}")
    print("🔄 Model should answer without searching (general knowledge)...")
    
    response5b = client.invoke(prompt5b, use_web=True)
    print(f"\n✅ Response:\n{response5b[:400]}..." if len(response5b) > 400 else f"\n✅ Response:\n{response5b}")
    
    # Example 6: Multiple tools - model chooses the right one
    print("\n" + "=" * 60)
    print("🎯 Example 6: Multiple Tools Available")
    print("-" * 60)
    print("When multiple tools are available, the model chooses the best one.\n")
    
    prompt6 = "If I invest $10,000 at 7.5% annual interest, how much will I have after 5 years with compound interest?"
    print(f"Prompt: {prompt6}")
    print(f"Available tools: calculator, weather, web_search")
    print("\n🔄 Model should choose calculator (not web search)...")
    
    response6 = client.invoke_with_tools(
        prompt6,
        tools=all_tools,
        tool_functions=tool_functions
    )
    print(f"\n✅ Response:\n{response6}")
    
    # Cleanup
    client.close()
    print("\n" + "=" * 60)
    print("✅ Tool calling examples completed!")
    print("\n💡 Key Concepts Demonstrated:")
    print("   - Tools are defined as JSON schemas")
    print("   - The LLM decides WHEN to call tools (not always!)")
    print("   - Web search is only used for current/recent information")
    print("   - General knowledge questions are answered directly")
    print("   - use_web=True provides simple web search integration")


if __name__ == "__main__":
    main()
