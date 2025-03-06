"""Example of using LLM Facades to create a consistent personality/style.

This example demonstrates creating an LLM Facade that always responds in simplified
language suitable for a young audience (ELI5 - "Explain Like I'm 5"). The ELI5 facade:

- Takes any input query and first gets a normal response
- Then automatically simplifies that response into child-friendly language
- Maintains this simplified style across all interface methods (streaming, structured, etc.)

This pattern is useful for:
- Creating consistent "personalities" or response styles
- Automatically adjusting technical content for different audiences
- Maintaining a specific tone or complexity level across an entire application
- Converting complex explanations into simpler language

The facade pattern makes it easy to add this ELI5 capability to any LLM while
maintaining a simple interface - the simplification happens automatically without
the caller needing to handle it explicitly.
"""

import argparse
import asyncio
from typing import AsyncGenerator

from langchain_wrappers import LLMDecorator
from examples.utils.provider_utils import (
    create_llm_wrapper,
    add_provider_arguments
)


class ELI5(LLMDecorator):
    async def hook_query(self, prompt_args: dict[str, str], api_args: dict[str, str]) -> AsyncGenerator[tuple[dict[str, str], dict[str, str]], str]:
        initial_response = await self.underlying_llm.query_response(**prompt_args, **api_args)
        response = yield {
            "CONTENT": initial_response,
            "TASK": "ELI5 the CONTENT. In other words, rephrase the CONTENT in a way that is easy to understand for a 5 year old.",
            **api_args
        }


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example of using LLM Facades to create a consistent personality/style")
    add_provider_arguments(parser)
    parser.add_argument(
        "--question", 
        type=str, 
        default="How does modern photolithography exposure work?",
        help="The question to ask and explain like you're 5 (default: photolithography)"
    )
    args = parser.parse_args()
    
    # Create LLM wrapper with specified provider and model
    llm = create_llm_wrapper(args.provider, args.model)
    eli5 = ELI5(underlying_llm=llm)

    # Normal langchain calls work
    response = eli5.invoke(args.question)
    print(response.content)


if __name__ == "__main__":
    asyncio.run(main())