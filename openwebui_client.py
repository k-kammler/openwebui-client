"""
OpenWebUI API Client

A Python client for interacting with OpenWebUI API services.
Specifically designed for JGU's OpenWebUI instance at ki-chat.uni-mainz.de.
Provides functionality for model management and chat completions. 

Author: Kevin Kammler
Date: September 9, 2025
Version: 1.0.0
"""

import json
import logging
import os
import configparser
from pathlib import Path
from time import sleep, time
from typing import Dict, List, Optional, Union, Any, Iterator
import requests


# Configuration Constants
DEFAULT_BASE_URL = 'https://ki-chat.uni-mainz.de'
DEFAULT_API_KEY_PLACEHOLDER = 'your-api-key-here'
DEFAULT_MODEL = 'Qwen3 235B Thinking'

# API Configuration
# To find/create your API key:
# 1. Visit https://ki-chat.uni-mainz.de, 
# 2. Go to Settings/Einstellungen → Account/Konto → API Key/API-Schlüssel 
# 3. Click show at the right side of the API Key field and copy the key.

def create_initial_config():
    """Create initial config file by asking user for all values."""
    config_file = Path('config.ini')
    
    print("\n🔧 Initial Setup - OpenWebUI Configuration")
    print("="*50)
    print("No configuration file found. Let's set up your OpenWebUI client!")
    
    # Get API key
    print("\n📝 Steps to get your API key:")
    print(f"1. Visit: {DEFAULT_BASE_URL}")
    print("2. Go to Settings → Account → API Key")
    print("3. Click 'Show' and copy the key")
    api_key = input("\n🔐 Enter your API key: ").strip()
    
    # Get base URL
    print(f"\n🌐 Base URL Setup")
    print(f"Default: {DEFAULT_BASE_URL}")
    base_url_input = input(f"\n🔗 Enter base URL (or press Enter for default): ").strip()
    base_url = base_url_input if base_url_input else DEFAULT_BASE_URL
    
    # Save all values to config
    if api_key:  # Only save if user provided an API key
        try:
            config = configparser.ConfigParser()
            config.add_section('openwebui')
            config.set('openwebui', 'api_key', api_key)
            config.set('openwebui', 'base_url', base_url)
            
            with open(config_file, 'w') as f:
                f.write("# OpenWebUI Configuration File\n")
                f.write("# Keep this file secure and don't share it!\n\n")
                config.write(f)
            
            print(f"\n✅ Configuration saved to {config_file}")
            print("💡 All values saved! You won't need to enter them again.")
            
            return {
                'api_key': api_key,
                'base_url': base_url
            }
        except Exception as e:
            print(f"⚠️  Couldn't save config file: {e}")
            print("💡 You can create config.ini manually")
            return {
                'api_key': api_key,
                'base_url': base_url
            }
    else:
        print("❌ No API key provided. Setup cancelled.")
        return None

def get_config_value(key: str, default_value: str = None, prompt_text: str = None):
    """
    Get configuration value from multiple sources.
    
    Args:
        key: The config key to look for ('api_key', 'base_url', or 'port')
        default_value: Default value if not found
        prompt_text: Text to show when prompting user
    
    Returns:
        The configuration value
    """
    config_file = Path('config.ini')
    
    # If config file doesn't exist, create it with all values
    if not config_file.exists():
        print(f"🔍 Looking for {key} but no config file exists...")
        config_values = create_initial_config()
        if config_values:
            return config_values.get(key, default_value)
        else:
            return default_value
    
    # Config file exists, try to read the specific value
    try:
        config = configparser.ConfigParser()
        config.read(config_file)
        value = config.get('openwebui', key, fallback=None)
        
        if value and (key != 'api_key' or value != DEFAULT_API_KEY_PLACEHOLDER):
            # print(f"✅ Using {key} from config.ini")
            return value
        else:
            print(f"⚠️  config.ini exists but {key} not configured properly")
            # Ask user to update this specific value
            if key == 'api_key':
                print("\n📝 Steps to get your API key:")
                print(f"1. Visit: {DEFAULT_BASE_URL}")
                print("2. Go to Settings → Account → API Key")
                print("3. Click 'Show' and copy the key")
                new_value = input("\n🔐 Enter your API key: ").strip()
            elif key == 'base_url':
                print(f"\n🌐 Base URL Setup")
                print(f"Default: {default_value}")
                user_input = input(f"\n🔗 Enter base URL (or press Enter for default): ").strip()
                new_value = user_input if user_input else default_value
            else:
                new_value = input(f"\n📝 Enter {key}: ").strip() if prompt_text is None else input(f"\n📝 {prompt_text}: ").strip()
            
            # Update config file with new value
            if new_value:
                try:
                    config.set('openwebui', key, new_value)
                    with open(config_file, 'w') as f:
                        f.write("# OpenWebUI Configuration File\n")
                        f.write("# Keep this file secure and don't share it!\n\n")
                        config.write(f)
                    print(f"✅ {key} updated in {config_file}")
                    return new_value
                except Exception as e:
                    print(f"⚠️  Couldn't update config file: {e}")
                    return new_value
            
            return default_value
            
    except Exception as e:
        print(f"⚠️  Error reading config.ini: {e}")
        return default_value

def get_api_key():
    """Get API key from config file or prompt user."""
    return get_config_value('api_key')

def get_base_url():
    """Get base URL from config file or use default."""
    return get_config_value('base_url', DEFAULT_BASE_URL)

# Allow setting all programmatically
get_api_key.override_key = None
get_base_url.override_url = None 

class OpenWebUIAPIError(Exception):
    """Custom exception for OpenWebUI API errors."""
    pass


class OpenWebUIClient:
    """
    A client for interacting with OpenWebUI API services.

    This client provides methods for:
    - Retrieving available models
    - Sending chat completion requests
    - Uploading files (when available)
    - Rate limiting to prevent API timeouts and limits

    Attributes:
        base_url (str): The base URL for the API
        api_key (str): The API authentication key
        session (requests.Session): HTTP session for connection pooling
        request_delay (float): Minimum seconds to wait between API requests (rate limiting)
    """
    
    def __init__(self, api_key: str = None, base_url: str = None, default_model: str = DEFAULT_MODEL,
                 default_temperature: float = 0.7, default_top_p: float = 0.95, default_seed: Optional[int] = None,
                 logging_level: int = logging.WARNING, save_history: bool = False, history_file: str = None,
                 request_delay: float = 0.0):
        """
        Initialize the OpenWebUI API client.

        Args:
            api_key (str): Your API key. If None, will try to get from config
            base_url (str): The base URL for the API service. If None, will try to get from config
            default_model (str): Default model to use when no model is specified in method calls
            default_temperature (float): Default temperature for sampling (0.0 to 2.0)
            default_top_p (float): Default nucleus sampling parameter (0.0 to 1.0)
            default_seed (Optional[int]): Default random seed for reproducible outputs
            logging_level (int): The logging level for the client
            save_history (bool): Whether to save/load chat history to/from file
            history_file (str): Path to history file. If None, uses 'chat_history.json'
            request_delay (float): Minimum seconds to wait between API requests (rate limiting). Default: 0.0 (no delay)

        Raises:
            ValueError: If no API key can be found from any source
        """
        # Get API key from override, parameter, or config
        if hasattr(get_api_key, 'override_key') and get_api_key.override_key:
            self.api_key = get_api_key.override_key
        else:
            self.api_key = api_key or get_api_key()
        
        # Get base URL from override, parameter, or config
        if hasattr(get_base_url, 'override_url') and get_base_url.override_url:
            self.base_url = get_base_url.override_url
        else:
            self.base_url = base_url or get_base_url()
        
        if not self.api_key:
            raise ValueError("API key cannot be found. Please configure in config.ini or provide api_key parameter.")
            
        self.base_url = self.base_url.rstrip('/')
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_seed = default_seed

        # Make these available as instance attributes for LLMInterface compatibility
        self.temperature = default_temperature
        self.top_p = default_top_p
        self.seed = default_seed

        self.session = requests.Session()
        self._setup_logging(logging_level)

        # Rate limiting (delay between requests)
        self.request_delay = request_delay
        self._last_request_time = 0.0

        # Chat history management
        self.save_history = save_history
        self.history_file = history_file or 'chat_history.json'
        self.chat_history: List[Dict[str, str]] = []

        # Load existing history if enabled
        if self.save_history:
            self._load_history()

    def _setup_logging(self, logging_level: int) -> None:
        """Set up logging for the client."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging_level)

    def _wait_for_rate_limit(self) -> None:
        """Enforce rate limiting by waiting if necessary."""
        if self.request_delay <= 0:
            return  # No rate limiting

        current_time = time()
        time_since_last_request = current_time - self._last_request_time

        if time_since_last_request < self.request_delay:
            wait_time = self.request_delay - time_since_last_request
            self.logger.debug(f"Rate limiting: waiting {wait_time:.2f} seconds")
            sleep(wait_time)

        self._last_request_time = time()

    def _make_request(self, endpoint: str, method: str = 'GET', data: Optional[Dict] = None,
                     headers: Optional[Dict] = None, **kwargs) -> requests.Response:
        """
        Make a request to the API with error handling and rate limiting.

        Args:
            endpoint (str): The API endpoint
            method (str): HTTP method
            data (Optional[Dict]): Request data
            headers (Optional[Dict]): Additional headers
            **kwargs: Additional arguments for requests

        Returns:
            requests.Response: The response object

        Raises:
            OpenWebUIAPIError: If the request fails
        """
        # Apply rate limiting
        self._wait_for_rate_limit()

        url = f"{self.base_url}{endpoint}"

        # Set up headers
        if headers is None:
            headers = {}
        headers.update({'Authorization': f'Bearer {self.api_key}'})
        headers.setdefault('Content-Type', 'application/json')

        self.logger.debug(f"Making {method} request to {url}")

        try:
            if method.upper() == 'GET':
                response = self.session.get(url, headers=headers, **kwargs)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data, headers=headers, **kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            self.logger.debug(f"Response status code: {response.status_code}")
            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if 'error' in error_data:
                        raise OpenWebUIAPIError(f"API Error: {error_data['error']}")
                except (json.JSONDecodeError, KeyError):
                    pass
            raise OpenWebUIAPIError(f"Request failed: {e}")

    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get the list of available models.
        
        Returns:
            List[Dict[str, Any]]: List of available models
            
        Raises:
            OpenWebUIAPIError: If the request fails
        """
        try:
            response = self._make_request('/api/models')
            response_data = response.json()
            self.logger.info(f"Retrieved {len(response_data['data'])} models")
            return response_data['data']
        except json.JSONDecodeError:
            raise OpenWebUIAPIError("Invalid JSON response from API")

    def _make_streaming_request(self, endpoint: str, data: Dict, headers: Optional[Dict] = None) -> requests.Response:
        """
        Make a streaming request to the API with rate limiting.

        Args:
            endpoint (str): The API endpoint
            data (Dict): Request data
            headers (Optional[Dict]): Additional header

        Returns:
            requests.Response: The streaming response object

        Raises:
            OpenWebUIAPIError: If the request fails
        """
        # Apply rate limiting
        self._wait_for_rate_limit()

        url = f"{self.base_url}{endpoint}"

        if headers is None:
            headers = {}
        headers.update({'Authorization': f'Bearer {self.api_key}'})
        headers.setdefault('Content-Type', 'application/json')

        self.logger.debug(f"Making streaming POST request to {url}")

        try:
            response = self.session.post(
                url,
                json=data,
                headers=headers,
                stream=True,
                timeout=30
            )
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Streaming request failed: {e}")
            raise OpenWebUIAPIError(f"Streaming error: {e}")

    def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        top_p: float = None,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Send a chat completion request with streaming response.
        
        Args:
            messages (List[Dict[str, str]]): List of conversation messages
            model (str): Model to use for completion
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            seed (Optional[int]): Random seed for reproducible outputs
            **kwargs: Additional parameters
            
        Yields:
            str: Response chunks as they arrive
            
        Raises:
            OpenWebUIAPIError: If the request fails
        """
        # Use default parameters if none specified
        if model is None:
            model = self.default_model
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if seed is None:
            seed = self.seed

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
            **kwargs
        }
        
        if seed is not None:
            data["seed"] = seed
        
        if seed is not None:
            data["seed"] = seed
        
        self.logger.info(f"Starting streaming chat completion with model: {model}")
        
        response = self._make_streaming_request('/api/chat/completions', data)
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith('data: '):
                    chunk_data = line[6:]  # Remove 'data: ' prefix
                    
                    if chunk_data.strip() == '[DONE]':
                        self.logger.debug("Stream completed with [DONE] marker")
                        break
                    
                    try:
                        chunk = json.loads(chunk_data)
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        # Skip invalid JSON chunks
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error processing streaming response: {e}")
            raise OpenWebUIAPIError(f"Streaming processing error: {e}")
        finally:
            response.close()

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        temperature: float = None,
        top_p: float = None,
        seed: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, Any]:
        """
        Send a chat completion request.
        
        Args:
            messages (List[Dict[str, str]]): List of conversation messages
            model (str): Model to use for completion
            temperature (float): Sampling temperature (0.0 to 2.0)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            seed (Optional[int]): Random seed for reproducible outputs
            stream (bool): Whether to stream the response
            **kwargs: Additional parameters
            
        Returns:
            Union[str, Any]: The completion response or generator for streaming
            
        Raises:
            OpenWebUIAPIError: If the request fails
        """
        # Use default parameters if none specified
        if model is None:
            model = self.default_model
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if seed is None:
            seed = self.seed

        if stream:
            return self.chat_completion_stream(messages, model, temperature, top_p, seed, **kwargs)

        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "stream": False,
            **kwargs
        }
        
        if seed is not None:
            data["seed"] = seed
        
        self.logger.info(f"Sending chat completion request with model: {model}")
        
        try:
            response = self._make_request('/api/chat/completions', 'POST', data)
            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                self.logger.info("Chat completion successful")
                return content
            else:
                raise OpenWebUIAPIError("No choices returned in response")
                
        except json.JSONDecodeError:
            raise OpenWebUIAPIError("Invalid JSON response from API")

    def upload_file(self, file_path: str, purpose: str = "assistants") -> str:
        """
        Upload a file to the API.
        
        Args:
            file_path (str): Path to the file to upload
            purpose (str): Purpose of the file upload
            
        Returns:
            str: The file ID
            
        Raises:
            OpenWebUIAPIError: If the upload fails
        """
        try:
            with open(file_path, 'rb') as file:
                files = {
                    'file': file,
                    'purpose': (None, purpose)
                }
                
                headers = {'Authorization': f'Bearer {self.api_key}'}
                
                response = self.session.post(
                    f"{self.base_url}/api/files",
                    files=files,
                    headers=headers
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    if 'id' in response_data:
                        self.logger.info(f"File uploaded successfully: {response_data['id']}")
                        return response_data['id']
                    else:
                        raise OpenWebUIAPIError("No file ID returned from upload")
                else:
                    raise OpenWebUIAPIError("File upload is not available via API")
                    
        except FileNotFoundError:
            raise OpenWebUIAPIError(f"Failed to read file: {e}")

    def invoke(self, prompt: str, model: str = None, **kwargs) -> str:
        """
        Simple interface for single-turn conversations, including for MCP tool usage.

        Args:
            prompt: The user prompt
            model: The model to use (defaults to self.default_model if not specified)
            **kwargs: Additional parameters for chat_completion (temperature, etc.)

        Returns:
            The assistant's response as a string
        """
        messages = [{"role": "user", "content": prompt}]
        # Use provided model or default (chat_completion will handle None)
        return self.chat_completion(messages, model=model, **kwargs)

    def invoke_stream(self, prompt: str, model: str = None, **kwargs):
        """
        Simple interface for single-turn conversations with streaming response.

        Args:
            prompt: The user prompt
            model: The model to use (defaults to self.default_model if not specified)
            **kwargs: Additional parameters for chat_completion_stream (temperature, etc.)

        Yields:
            str: Response chunks as they arrive
        """
        messages = [{"role": "user", "content": prompt}]
        # Use provided model or default (chat_completion_stream will handle None)
        return self.chat_completion_stream(messages, model=model, **kwargs)

    def set_rate_limit(self, delay_seconds: float) -> None:
        """
        Set or update the rate limiting delay.

        Args:
            delay_seconds (float): Minimum seconds to wait between API requests.
                                 Set to 0.0 to disable rate limiting.
        """
        self.request_delay = delay_seconds
        self.logger.info(f"Rate limit updated to {delay_seconds} seconds between requests")

    def close(self):
        """Close the session."""
        self.session.close()
        self.logger.info("API client session closed")

    # Chat History Management Methods
    
    def _load_history(self) -> None:
        """Load chat history from file if it exists."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.chat_history = json.load(f)
                self.logger.info(f"Loaded {len(self.chat_history)} messages from {self.history_file}")
            else:
                self.logger.info(f"No existing history file found at {self.history_file}")
        except Exception as e:
            self.logger.error(f"Error loading chat history: {e}")
            self.chat_history = []

    def _save_history(self) -> None:
        """Save current chat history to file."""
        if not self.save_history:
            return
            
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.chat_history, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved {len(self.chat_history)} messages to {self.history_file}")
        except Exception as e:
            self.logger.error(f"Error saving chat history: {e}")

    def add_to_history(self, role: str, content: str) -> None:
        """
        Add a message to chat history.
        
        Args:
            role (str): The role of the message sender ('user', 'assistant', 'system')
            content (str): The content of the message
        """
        message = {"role": role, "content": content}
        self.chat_history.append(message)
        self._save_history()

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the current chat history.
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries with 'role' and 'content'
        """
        return self.chat_history.copy()

    def clear_history(self) -> None:
        """Clear the chat history and optionally save empty history to file."""
        self.chat_history = []
        self._save_history()
        self.logger.info("Chat history cleared")

    def chat_with_history(self, user_message: str, model: str = None,
                         include_system_prompt: str = None, **kwargs) -> str:
        """
        Send a chat message while maintaining conversation history.
        
        Args:
            user_message (str): The user's message
            model (str): Model to use for completion
            include_system_prompt (str): Optional system prompt to include
            **kwargs: Additional parameters for chat_completion
            
        Returns:
            str: The assistant's response
        """
        # Add user message to history
        self.add_to_history("user", user_message)
        
        # Prepare messages for API call
        messages = []
        
        # Add system prompt if provided
        if include_system_prompt:
            messages.append({"role": "system", "content": include_system_prompt})
        
        # Add conversation history
        messages.extend(self.chat_history)
        
        # Get response from API (chat_completion will handle None model)
        response = self.chat_completion(messages, model=model, **kwargs)
        
        # Add assistant response to history
        self.add_to_history("assistant", response)
        
        return response

    def get_history_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current chat session.
        
        Returns:
            Dict containing message counts and session info
        """
        if not self.chat_history:
            return {"message_count": 0, "session_active": False}
            
        user_messages = sum(1 for msg in self.chat_history if msg["role"] == "user")
        assistant_messages = sum(1 for msg in self.chat_history if msg["role"] == "assistant")
        system_messages = sum(1 for msg in self.chat_history if msg["role"] == "system")
        
        return {
            "message_count": len(self.chat_history),
            "user_messages": user_messages,
            "assistant_messages": assistant_messages,
            "system_messages": system_messages,
            "session_active": True,
            "history_file": self.history_file if self.save_history else None
        }

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Example usage and testing
if __name__ == "__main__":
    # This will only run if you execute this file directly
    print("🧪 Testing OpenWebUI Client")
    print("="*40)
    
    try:
        # Initialize client (will automatically handle config setup if needed)
        print("🔧 Initializing OpenWebUI client...")
        client = OpenWebUIClient()   

        print(f"📡 Connected to: {client.base_url}")
        print(f"🔑 API key configured: {'***' + client.api_key[-4:] if len(client.api_key) > 4 else '***'}")
        print(f"⏱️  Rate limiting: {client.request_delay} seconds between requests")

        # Test getting models
        print("\n📋 Testing model access...")
        models = client.get_models()
        if models:
            print(f"✅ Found {len(models)} available models:")
            for i, model in enumerate(models):  # Show first 5 models
                print(f"   {i}. {model.get('id', 'Unknown')}")
        else:
            print("⚠️  No models found")

        # Test a simple chat  
        first_model = models[0]['id'] if models else "default"

        # First request with first model using invoke method
        print("\n💬 Testing invoke (simple single-turn)...")
        print("   Making first request...")
        response1 = client.invoke("What model are you and where are you hosted?", model=first_model)
        print(f"   ✅ First response: {response1}\n")

        # Second request using chat_completion with messages
        print("\n💬 Testing chat_completion (with message formatting)...")
        print("   Making request with system prompt...")
        messages = [
            {"role": "system", "content": "You are a helpful math tutor. Be concise."},
            {"role": "user", "content": "What is 2 + 2?"}
        ]
        response2 = client.chat_completion(messages, model=first_model)
        print(f"   ✅ Second response: {response2}")

        # Test chat with history
        print("\n📚 Testing chat with history (multi-turn conversation)...")
        history_client = OpenWebUIClient(save_history=True, history_file='test_chat_history.json')
        
        # Clear any existing history
        history_client.clear_history()
        print("   Starting new conversation...")
        
        # First turn
        hist_response1 = history_client.chat_with_history(
            "My favorite color is blue. Remember this!",
            model=first_model
        )
        print(f"   👤 User: My favorite color is blue. Remember this!")
        print(f"   🤖 Assistant: {hist_response1}")
        
        # Second turn - testing if it remembers
        hist_response2 = history_client.chat_with_history(
            "What's my favorite color?",
            model=first_model
        )
        print(f"   👤 User: What's my favorite color?")
        print(f"   🤖 Assistant: {hist_response2}")
        
        # Show history summary
        summary = history_client.get_history_summary()
        print(f"\n   📊 Session summary: {summary['message_count']} messages")
        print(f"      ({summary['user_messages']} user, {summary['assistant_messages']} assistant)")
        print(f"   💾 History saved to: {summary['history_file']}")
        
        history_client.close()

        # Test streaming with invoke_stream (simple interface)
        print("\n🌊 Testing streaming with invoke_stream (simple interface)...")
        print("   Streaming response in real-time:")
        print("   🤖 Assistant: ", end="", flush=True)
        
        for chunk in client.invoke_stream("Count from 1 to 10 slowly, with a brief pause between each number.", model=first_model):
            print(chunk, end="", flush=True)
        print()  # New line after streaming completes
        
        # Test streaming with chat_completion_stream (advanced interface)
        print("\n🌊 Testing streaming with chat_completion_stream (advanced interface)...")
        print("   Streaming response with custom messages:")
        print("   🤖 Assistant: ", end="", flush=True)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Be very detailed and answer thoroughly."},
            {"role": "user", "content": "Tell me a joke about programming."}
        ]
        for chunk in client.chat_completion_stream(messages, model=first_model):
            print(chunk, end="", flush=True)
        print()  # New line after streaming completes
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure your API key is correct in config.ini (file should be automatically created in first run)")
        print("2. Check that then OpenWebUI server is accessible and that the base URL in config.ini is correct")
        print("3. Verify your internet connection and firewall settings")


