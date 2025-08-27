from dataclasses import dataclass, field
from typing import List, Dict, Literal


# 모델 타입 정의
ModelType = Literal["llm", "embedding"]


@dataclass
class ModelInfo:
    name: str
    type: ModelType
    max_tokens: int | None = None
    description: str = ""
    price_per_1k_tokens: float | None = None
    endpoint: str | None = None

@dataclass
class Provider:
    name: str
    models: List[ModelInfo] = field(default_factory=list)

    def get_model(self, model_name: str) -> ModelInfo:
        for model in self.models:
            if model.name == model_name:
                return model
        raise ValueError(f"지원하지 않는 모델: {model_name}")
    

AVAILABLE_PROVIDERS: Dict[str, Provider] = {
    "openai": Provider(
        name="openai",
        models=[
            ModelInfo(name="gpt-4o", type="llm", max_tokens=128000, price_per_1k_tokens=0.01),
            ModelInfo(name="gpt-4o-mini", type="llm", max_tokens=64000),
            ModelInfo(name="gpt-3.5-turbo", type="llm", max_tokens=16000),
            ModelInfo(name="text-embedding-3-small", type="embedding"),
        ]
    ),
    "claude": Provider(
        name="claude",
        models=[
            ModelInfo(name="claude-3-opus-20240229", type="llm"),
            ModelInfo(name="claude-3-sonnet-20240229", type="llm"),
        ]
    ),
    "upstage": Provider(
        name="upstage",
        models=[
            ModelInfo(name="solar-mini-250422", type="llm"),
            ModelInfo(name="solar-pro2-250710", type="llm"),
            ModelInfo(name="solar-embedding-1-large", type="embedding"),
        ]
    ),
    "ollama": Provider(
        name="ollama",
        models=[
            ModelInfo(name="llama2", type="llm"),
            ModelInfo(name="mistral", type="llm"),
            ModelInfo(name="gemma", type="llm"),
        ]
    ),
    "huggingface": Provider(
        name="huggingface",
        models=[
            ModelInfo(name="sentence-transformers/all-MiniLM-L6-v2", type="embedding"),
        ]
    ),
}


# TODO: dataclass로 만들까 했는데, 사용가능한 모델 말고 더 뭐가 필요하지 떠오르는게 없어서 일단 dict으로 진행~
AVAILABLE_MODELS = {
    "openai": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "claude": ["claude-3-opus-20240229", "claude-3-sonnet-20240229"],
    "upstage": ["solar-mini-250422", "solar-pro2-250710"],
    "ollama": ["llama2", "mistral", "gemma"]
}

# TODO: dataclass로 만들까 했는데, 사용가능한 모델 말고 더 뭐가 필요하지 떠오르는게 없어서 일단 dict으로 진행~
AVAILABLE_MODELS_EMBEDDINGS = {
    "openai": ["text-embedding-3-small"],
    "huggingface": ["sentence-transformers/all-MiniLM-L6-v2"],
    "upstage": ["solar-embedding-1-large"]
}

def check_model_name(provider, model_name, available_models: dict):
    if provider not in available_models:
        raise ValueError(f"지원하지 않는 LLM provider: {provider}")
    if model_name not in available_models[provider]:
        raise ValueError(f"지원하지 않는 모델: {model_name}")