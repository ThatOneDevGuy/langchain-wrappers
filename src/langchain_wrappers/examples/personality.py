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


from typing import AsyncGenerator
from ..llm_wrappers import wrapper_from_chatmodel, LLMDecorator
from langchain_openai import ChatOpenAI

class ELI5(LLMDecorator):
    async def hook_query(self, prompt_args: dict[str, str], api_args: dict[str, str]) -> AsyncGenerator[tuple[dict[str, str], dict[str, str]], str]:
        initial_response = await self.underlying_llm.query_response(**prompt_args, **api_args)
        response = yield {
            "CONTENT": initial_response,
            "TASK": "ELI5 the CONTENT. In other words, rephrase the CONTENT in a way that is easy to understand for a 5 year old.",
            **api_args
        }

async def main():
    gpt4omini = wrapper_from_chatmodel(ChatOpenAI(model="gpt-4o-mini"))
    eli5 = ELI5(underlying_llm=gpt4omini)

    # Normal langchain calls work
    response = eli5.invoke("How does modern photolithography exposure work?")
    print(response.content)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())