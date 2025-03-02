from langchain_core.language_models import BaseChatModel
from typing import AsyncGenerator

from pydantic import Field


from .chat_wrapper import ChatWrapper

class LangchainChatWrapper(ChatWrapper):
    llm: BaseChatModel = Field(default=None)

    async def query(self, **kwargs) -> AsyncGenerator[str, None]:
        if kwargs.get("stream", False):
            async for chunk in self.llm.astream(kwargs["messages"]):
                yield chunk.content
        else:
            yield self.llm.invoke(kwargs["messages"]).content

def wrapper_from_chatmodel(llm: BaseChatModel) -> LangchainChatWrapper:
    return LangchainChatWrapper(llm=llm)
