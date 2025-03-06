"""Example of using LLM Facades for multi-step workflows with parallel tasks.

This example demonstrates creating a workflow-focused LLM Facade that breaks down
question answering into multiple coordinated steps. The WorkflowQA facade:

1. Analyzes the user's knowledge level (task1)
2. Identifies key points to cover (task2) 
3. Uses the gathered context to generate a comprehensive response

Tasks 1 and 2 run in parallel to optimize performance. The facade then combines
their outputs to inform the final response generation.

This pattern is useful for:
- Breaking complex LLM tasks into coordinated subtasks
- Gathering multiple pieces of context in parallel
- Ensuring responses are properly scoped and targeted
- Maintaining consistent response quality through structured workflows

The facade pattern makes it easy to encapsulate this multi-step workflow while
maintaining a simple interface - the complexity of task coordination happens
automatically without the caller needing to handle it explicitly.
"""

import argparse
import asyncio
from typing import AsyncGenerator

from langchain_wrappers import LLMDecorator
from examples.utils.provider_utils import (
    create_llm_wrapper,
    add_provider_arguments
)


class WorkflowQA(LLMDecorator):
    async def hook_query(self, prompt_args: dict[str, str], api_args: dict[str, str]) -> AsyncGenerator[tuple[dict[str, str], dict[str, str]], str]:
        task1 = self.underlying_llm.query_block(
            "text",
            USER_ARGS=prompt_args,
            TASK=(
                "Analyze the question posed by the user in the USER_ARGS. "
                "Infer the user's knowledge level based on the request, and provide a statement of that level."
            )
        )

        task2 = self.underlying_llm.query_block(
            "text",
            USER_ARGS=prompt_args,
            TASK=(
                "Analyze the question posed by the user in the USER_ARGS. "
                "Identify the key points that need to be covered to answer the question, and provide a list of those key points."
            )
        )

        knowledge_level, key_points = await asyncio.gather(task1, task2)

        response = yield {
            "KNOWLEDGE_LEVEL": knowledge_level,
            "KEY_POINTS": key_points,
            "USER_ARGS": prompt_args,
            "TASK": (
                "Analyze the question posed by the user in the USER_ARGS. "
                "Provide a comprehensive response to question posed by the user in USER_ARGS. "
                "The response should be tailored to the KNOWLEDGE_LEVEL of the user. "
                "The response should cover the KEY_POINTS that are relevant to the question. "
            ),
            **api_args
        }


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Example of using LLM Facades for multi-step workflows")
    add_provider_arguments(parser)
    parser.add_argument(
        "--question", 
        type=str, 
        default="How are computer chips made?",
        help="The question to answer using the workflow process (default: How are computer chips made?)"
    )
    args = parser.parse_args()
    
    # Create LLM wrapper with specified provider and model
    llm = create_llm_wrapper(args.provider, args.model)
    qa = WorkflowQA(underlying_llm=llm)

    response = await qa.query_block(
        "md", # Markdown output
        QUESTION=args.question
    )

    print(response)


if __name__ == "__main__":
    asyncio.run(main())