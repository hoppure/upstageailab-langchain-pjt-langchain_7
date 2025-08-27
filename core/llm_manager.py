from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_upstage import UpstageEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatAnthropic  # Claude
from langchain_upstage import ChatUpstage   # Upstage
from langchain_community.chat_models import ChatOllama    # 로컬 LLM 예시

from config.models import AVAILABLE_MODELS, AVAILABLE_MODELS_EMBEDDINGS, check_model_name


class LLMManager:
    def __init__(self, llm_provider="upstage", embedding_provider="openai", **kwargs):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.kwargs = kwargs

    def get_llm(self):
        check_model_name(self.llm_provider, self.kwargs.get("model_name", "-"), AVAILABLE_MODELS)
        if self.llm_provider == "openai":
            return ChatOpenAI(model_name=self.kwargs.get("model_name", "gpt-4o"),
                              temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "claude":
            return ChatAnthropic(model=self.kwargs.get("model_name", "claude-3-opus-20240229"),
                                 temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "upstage":
            return ChatUpstage(model=self.kwargs.get("model_name", "solar-pro2-250710"),
                               temperature=self.kwargs.get("temperature", 0))
        elif self.llm_provider == "ollama":
            return ChatOllama(model=self.kwargs.get("model_name", "llama2"),
                              temperature=self.kwargs.get("temperature", 0))
        else:
            raise ValueError(f"지원하지 않는 LLM provider: {self.llm_provider}")

    def get_embeddings(self):
        check_model_name(self.embedding_provider, self.kwargs.get("embedding_name", "-"), AVAILABLE_MODELS_EMBEDDINGS)
        if self.embedding_provider == "openai":
            return OpenAIEmbeddings(model=self.kwargs.get("embedding_name", "text-embedding-3-small"))
        elif self.embedding_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.kwargs.get("embedding_name", "sentence-transformers/all-MiniLM-L6-v2"))
        elif self.embedding_provider == "upstage":
            return UpstageEmbeddings(model=self.kwargs.get("embedding_name", "solar-embedding-1-large"),)
        else:
            raise ValueError(f"지원하지 않는 Embedding provider: {self.embedding_provider}")