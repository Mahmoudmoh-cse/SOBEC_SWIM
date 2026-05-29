from __future__ import annotations

from abc import ABC, abstractmethod

from app.ai.schemas import AIProviderRequest, AIProviderResult


class AIProviderError(RuntimeError):
    pass


class AIProviderConfigError(AIProviderError):
    pass


class AIProvider(ABC):
    name: str

    @abstractmethod
    def generate(self, request: AIProviderRequest) -> AIProviderResult:
        raise NotImplementedError

