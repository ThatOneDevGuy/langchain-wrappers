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

from typing import AsyncGenerator
from ..llm_wrappers import wrapper_from_chatmodel
from langchain_openai import ChatOpenAI
from typing import Any, AsyncGenerator
from .workflow import WorkflowQA



from ..llm_wrappers import ChatWrapper

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
    gpt4omini = wrapper_from_chatmodel(ChatOpenAI(model="gpt-4o-mini"))
    capturing_llm = CapturingLLM(underlying_llm=gpt4omini)
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
    import asyncio
    asyncio.run(main())