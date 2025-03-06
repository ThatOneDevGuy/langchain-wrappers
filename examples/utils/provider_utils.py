"""
Provider initialization utility module for langchain-wrappers examples.

This module provides functions to initialize different LLM providers and create 
wrappers that can be used for testing or example scripts. It handles
provider-specific imports and configuration, making it easy to switch between
providers when running example scripts or benchmarks.
"""

import argparse
from typing import Optional

from langchain_wrappers import wrapper_from_chatmodel

# Import providers with error handling
from langchain_openai import ChatOpenAI

# For cerebras support
try:
    from langchain_cerebras import ChatCerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False

# For Groq support
try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


def get_default_model(provider: str) -> str:
    """Get the default model for the specified provider.
    
    Args:
        provider (str): The name of the provider (openai, cerebras, groq)
        
    Returns:
        str: The default model name for the provider
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    if provider == "openai":
        return "gpt-4o-mini"
    elif provider == "cerebras":
        return "llama-3.3-70b"
    elif provider == "groq":
        return "llama-3.3-70b-versatile"  # Groq's Llama 3.1 8B model
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_llm_wrapper(provider: str, model: Optional[str] = None, **provider_args):
    """Create an LLM wrapper based on provider and model.
    
    Args:
        provider (str): The name of the provider (openai, cerebras, groq)
        model (str, optional): The model name to use. If None, uses the default model.
        **provider_args: Additional arguments to pass to the provider's constructor
        
    Returns:
        The LLM wrapper for the specified provider and model
        
    Raises:
        ValueError: If the provider is not supported
        ImportError: If the provider's module is not installed
    """
    provider = provider.lower()
    
    # If model is not specified, use the default for the provider
    if model is None:
        model = get_default_model(provider)
    
    if provider == "openai":
        chat_model = ChatOpenAI(model=model, **provider_args)
    elif provider == "cerebras":
        if not CEREBRAS_AVAILABLE:
            raise ImportError("langchain_cerebras is not installed. Install it with: pip install langchain-cerebras")
        chat_model = ChatCerebras(model=model, **provider_args)
    elif provider == "groq":
        if not GROQ_AVAILABLE:
            raise ImportError("langchain_groq is not installed. Install it with: pip install langchain-groq")
        chat_model = ChatGroq(model=model, **provider_args)
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers are: openai, cerebras, groq")
    
    return wrapper_from_chatmodel(chat_model)


def get_available_providers():
    """Get a list of available providers.
    
    Returns:
        list: List of available provider names
    """
    providers = ["openai"]
    
    if CEREBRAS_AVAILABLE:
        providers.append("cerebras")
    
    if GROQ_AVAILABLE:
        providers.append("groq")
    
    return providers


def add_provider_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add provider and model arguments to an ArgumentParser.
    
    Args:
        parser (argparse.ArgumentParser): The parser to add arguments to
        
    Returns:
        argparse.ArgumentParser: The parser with the arguments added
    """
    parser.add_argument(
        "--provider", 
        type=str, 
        default="openai", 
        choices=get_available_providers(),
        help="LLM provider to use (default: openai)"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None, 
        help="Model identifier to use (defaults to provider-specific default model)"
    )
    return parser