"""Example of using LLM Facades for debugging and input/output capture.

This example demonstrates creating a debugging-focused LLM Facade that records all inputs 
and outputs while transparently passing them through to an underlying LLM. The CapturingLLM
facade maintains a history of all interactions that can be inspected later.

This pattern is useful for:
- Debugging complex LLM workflows by inspecting the exact inputs/outputs
- Logging LLM interactions for analysis
- Testing LLM-based applications by verifying the interaction history
- Auditing LLM usage and behavior

It's also easy to modify this LLM Facade to record additional information, such as timestamps.

The facade follows the core LLM Facade pattern of providing a simple interface that hides
complexity - in this case the complexity of capturing and storing interaction history while
maintaining the standard LLM interface.
"""

import argparse
import asyncio
from typing import Any, AsyncGenerator

from langchain_wrappers import ChatWrapper
from examples.utils.provider_utils import (
    create_llm_wrapper,
    add_provider_arguments
)
from examples.workflow import WorkflowQA


class CapturingLLM(ChatWrapper):
    underlying_llm: ChatWrapper = None
    history: list[dict[str, Any]] = []

    def __init__(self, underlying_llm: ChatWrapper, **kwargs):
        super().__init__(**kwargs)
        self.underlying_llm = underlying_llm
        self.history = []
    
    async def query(self, **kwargs) -> AsyncGenerator[str, None]:
        record = {
            "input": kwargs,
            "output": []
        }

        async for chunk in self.underlying_llm.query(**kwargs):
            record["output"].append(chunk)
            yield chunk
        
        self.history.append(record)


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example of using LLM Facades for debugging and input/output capture")
    add_provider_arguments(parser)
    args = parser.parse_args()
    
    # Create LLM wrapper with specified provider and model
    llm = create_llm_wrapper(args.provider, args.model)
    capturing_llm = CapturingLLM(underlying_llm=llm)
    qa = WorkflowQA(underlying_llm=capturing_llm)

    await qa.query_block(
        "md", # Markdown output
        QUESTION="How are computer chips made?"
    )

    for record in capturing_llm.history:
        print("-"*80)
        print("Input arguments:", record["input"])
        print('')
        print("Output:", "".join(record["output"]))


if __name__ == "__main__":
    asyncio.run(main())