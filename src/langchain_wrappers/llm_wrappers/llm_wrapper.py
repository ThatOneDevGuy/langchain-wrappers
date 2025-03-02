from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, AsyncGenerator, Generator, Optional, Type, TypeVar, Union

from pydantic import BaseModel

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, AIMessageChunk
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult, ChatGeneration, ChatGenerationChunk


T = TypeVar("T", bound=BaseModel)
S = TypeVar("S", bound=Union[str, T])


class LLMWrapper(BaseChatModel, ABC):
    """
    Base class for asynchronous Language Learning Model (LLM) engines.

    Provides a standardized interface for:
    - Making raw queries to LLM APIs
    - Getting responses as structured objects
    - Getting responses in specific block formats
    - Rate limiting and usage tracking

    Subclasses must implement the abstract query methods for specific LLM APIs.
    """


    @abstractmethod
    async def query_response(self, **kwargs: Any) -> tuple[str, int]:
        """
        Send a query to the LLM and get the complete response.

        Args:
            **kwargs: API-specific arguments (e.g. max_tokens, temperature)

        Returns:
            tuple[str, int]: Response text and tokens consumed
        """
        ...

    @abstractmethod
    async def query_stream(self, **kwargs: Any) -> AsyncGenerator[str, None]:
        """
        Send a query to the LLM and stream the response chunks.

        Args:
            **kwargs: API-specific arguments

        Yields:
            str: Response text chunks as they arrive
        """
        ...

    @abstractmethod
    async def query_object(self, response_model: Type[T], **kwargs: Any) -> T:
        """
        Query the LLM and parse the response into a structured object.

        Args:
            response_model: Pydantic model class to parse response into
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            T: Response parsed into response_model instance
        """
        ...

    @abstractmethod
    async def query_block(self, block_type: str, **kwargs: Any) -> str:
        """
        Query the LLM for a specific block type response.

        Args:
            block_type: Type of block to request (e.g. "python", "json")
            **kwargs: Prompt args (UPPERCASE) and API args (lowercase)

        Returns:
            str: Response formatted as requested block type
        """
        ...

    def _llm_type(self) -> str:
        return type(self).__module__ + "." + self.__class__.__name__

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            return new_loop.run_until_complete(
                self._agenerate(messages, stop, run_manager)
            )
        
        with ThreadPoolExecutor() as executor:
            future = executor.submit(run_in_thread)
            return future.result()

            
    async def _agenerate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> ChatResult:
        compiled_messages = []
        for message in messages:
            if isinstance(message.content, str):
                compiled_messages.append({"role": "user", "content": message.content})
            elif isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, str):
                        compiled_messages.append({"role": "user", "content": item})
                    elif isinstance(item, dict):
                        compiled_messages.append(item)

        result = await self.query_response(messages=compiled_messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=result))])
    

    def _stream(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> Generator[ChatGenerationChunk, None, None]:
        from queue import Queue
        queue = Queue()
        completion_obj = object()
        
        async def process_stream():
            async for chunk in self._astream(messages, stop, run_manager):
                queue.put(chunk)
            queue.put(completion_obj)  # Signal completion
            
        def run_in_thread():
            new_loop = asyncio.new_event_loop()
            return new_loop.run_until_complete(process_stream())
        
        thread = ThreadPoolExecutor().submit(run_in_thread)
        
        while True:
            item = queue.get()
            if item is completion_obj:
                break
            yield item
                

    async def _astream(self, messages: list[BaseMessage], stop: Optional[list[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> AsyncGenerator[str, None]:
        async for chunk in self.query_stream(messages=messages):
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
