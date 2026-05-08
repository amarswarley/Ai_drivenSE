"""
Model Interface: Abstraction layer for API-based and open-access models
Supports: OpenAI, Anthropic, Ollama (local), HuggingFace Inference API
"""

import os
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import json

logger = logging.getLogger(__name__)


class BaseModelInterface(ABC):
    """Abstract base class for model interfaces"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        """
        Generate completion
        
        Returns:
            {
                'text': generated text,
                'prompt_tokens': number of prompt tokens,
                'completion_tokens': number of completion tokens
            }
        """
        pass


class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models (GPT-4o, GPT-4o-mini)"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
        
        logger.info(f"Initialized OpenAI interface for {model}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        return {
            'text': response.choices[0].message.content,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }


class AnthropicInterface(BaseModelInterface):
    """Interface for Anthropic models (Claude Sonnet, Opus, Haiku)"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not self.api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")
        
        logger.info(f"Initialized Anthropic interface for {model}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'text': response.content[0].text,
            'prompt_tokens': response.usage.input_tokens,
            'completion_tokens': response.usage.output_tokens
        }


class OllamaInterface(BaseModelInterface):
    """Interface for local Ollama models"""
    
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("Please install requests: pip install requests")
        
        # Verify connection
        try:
            response = self.requests.get(f"{base_url}/api/tags")
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {base_url}")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to Ollama at {base_url}. "
                f"Make sure Ollama is running with: ollama serve"
            ) from e
        
        logger.info(f"Initialized Ollama interface for {model}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        response = self.requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "num_predict": max_tokens
            }
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Ollama doesn't return token counts, estimate them
        prompt_tokens = len(prompt.split())  # Rough estimate
        completion_tokens = len(data.get('response', '').split())
        
        return {
            'text': data['response'],
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }


class HuggingFaceInterface(BaseModelInterface):
    """Interface for HuggingFace Inference API models"""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("HUGGINGFACE_API_KEY")
        
        if not self.api_key:
            raise ValueError("HuggingFace API key not found. Set HUGGINGFACE_API_KEY environment variable.")
        
        try:
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(api_key=self.api_key)
        except ImportError:
            raise ImportError("Please install huggingface-hub: pip install huggingface-hub")
        
        logger.info(f"Initialized HuggingFace interface for {model}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        response = self.client.text_generation(
            self.model,
            prompt,
            max_new_tokens=max_tokens,
            temperature=0.7
        )
        
        # Estimate token counts
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response.split())
        
        return {
            'text': response,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens
        }


class LocalTransformerInterface(BaseModelInterface):
    """Interface for local transformer models (using transformers library)"""
    
    def __init__(self, model: str):
        self.model = model
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model_obj = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto"
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        except ImportError:
            raise ImportError(
                "Please install transformers and torch: "
                "pip install transformers torch"
            )
        
        logger.info(f"Initialized local transformer interface for {model}")
        logger.info(f"Using device: {self.device}")
    
    def generate(self, prompt: str, max_tokens: int = 2048) -> Dict[str, Any]:
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model_obj.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from generated text
        text = generated_text[len(prompt):].strip()
        
        return {
            'text': text,
            'prompt_tokens': len(inputs['input_ids'][0]),
            'completion_tokens': len(outputs[0]) - len(inputs['input_ids'][0])
        }


class ModelInterface:
    """Factory class for creating model interfaces"""
    
    # Map of model patterns to interface classes
    _MODEL_MAPPING = {
        # OpenAI models
        'gpt-4': OpenAIInterface,
        'gpt-4o': OpenAIInterface,
        'gpt-3.5': OpenAIInterface,
        
        # Anthropic models
        'claude': AnthropicInterface,
        'claude-opus': AnthropicInterface,
        'claude-sonnet': AnthropicInterface,
        'claude-sonnet-4': AnthropicInterface,
        'claude-haiku': AnthropicInterface,
        
        # Open-access / local models
        'ollama:': OllamaInterface,
        'llama': LocalTransformerInterface,
        'mistral': LocalTransformerInterface,
        'qwen': LocalTransformerInterface,
        'phi': LocalTransformerInterface,
        'deepseek': LocalTransformerInterface,
        'openchat': LocalTransformerInterface,
        'neural-chat': LocalTransformerInterface,
        
        # HuggingFace Inference API
        'hf:': HuggingFaceInterface,
        'huggingface:': HuggingFaceInterface,
    }
    
    @classmethod
    def get_interface(cls, model: str) -> BaseModelInterface:
        """
        Get the appropriate interface for a model identifier
        
        Args:
            model: Model identifier, e.g.:
                - 'gpt-4o-mini' -> OpenAI
                - 'claude-sonnet-4.5' -> Anthropic
                - 'ollama:llama2' -> Local Ollama
                - 'meta-llama/Llama-2-7b-hf' -> Local transformer
                - 'hf:mistralai/Mistral-7B' -> HuggingFace Inference API
        
        Returns:
            Initialized model interface
        
        Raises:
            ValueError: If model type is unknown
        """
        
        logger.info(f"Creating interface for model: {model}")
        
        # Special handling for Ollama models
        if model.startswith('ollama:'):
            model_name = model.split(':', 1)[1]
            return OllamaInterface(model_name)
        
        # Special handling for HuggingFace Inference API
        if model.startswith('hf:') or model.startswith('huggingface:'):
            model_name = model.split(':', 1)[1]
            return HuggingFaceInterface(model_name)
        
        # Check against patterns
        for pattern, interface_class in cls._MODEL_MAPPING.items():
            if pattern in model.lower():
                return interface_class(model)
        
        # Default: try as local transformer model
        logger.info(f"Model '{model}' not recognized in mapping, attempting local transformer load")
        return LocalTransformerInterface(model)


def test_model_interface():
    """Test function to verify model interfaces work"""
    
    test_prompt = "Write a Python function that computes GC content of a DNA sequence."
    
    # Example configurations (uncomment to test)
    models_to_test = [
        # 'gpt-4o-mini',  # Requires OPENAI_API_KEY
        # 'claude-sonnet-4.5',  # Requires ANTHROPIC_API_KEY
        # 'ollama:llama2',  # Requires local Ollama
        # 'meta-llama/Llama-2-7b-hf',  # Requires transformers + model downloaded
    ]
    
    for model in models_to_test:
        try:
            interface = ModelInterface.get_interface(model)
            logger.info(f"Testing {model}...")
            
            response = interface.generate(test_prompt, max_tokens=200)
            
            logger.info(f"✓ {model} response tokens: {response['prompt_tokens']} prompt, {response['completion_tokens']} completion")
            
        except Exception as e:
            logger.error(f"✗ {model} failed: {str(e)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_model_interface()
