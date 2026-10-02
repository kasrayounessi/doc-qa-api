from langchain_openai import ChatOpenAI

from app.core.config import settings


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.generation_model,
        openai_api_key=settings.openai_api_key,
        temperature=0.0,
        max_retries=2,
    )
