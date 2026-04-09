"""Multi-provider LLM client with fallback strategy."""
from typing import Optional, List, Dict, Any
import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is configured and available."""
        pass


class GroqProvider(LLMProvider):
    """Groq LLM provider using langchain_groq."""
    
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.llm = None
        if api_key:
            try:
                from langchain_groq import ChatGroq
                # Set API key in environment for ChatGroq to pick up
                os.environ["GROQ_API_KEY"] = api_key
                self.llm = ChatGroq(
                    model=model,
                    temperature=temperature,
                    max_retries=2,
                )
            except Exception as e:
                print(f"[Groq] Failed to initialize: {e}")
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.llm:
            raise ConnectionError("Groq client not initialized")
        
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        
        response = self.llm.invoke(messages)
        return response.content


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider using langchain_google_genai."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.llm = None
        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                # Set API key in environment for ChatGoogleGenerativeAI to pick up
                os.environ["GOOGLE_API_KEY"] = api_key
                self.llm = ChatGoogleGenerativeAI(
                    model=model,
                    temperature=temperature,
                    max_retries=2,
                )
            except Exception as e:
                print(f"[Gemini] Failed to initialize: {e}")
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.llm:
            raise ConnectionError("Gemini client not initialized")
        
        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt))
        
        response = self.llm.invoke(messages)
        return response.content


class DeepSeekProvider(LLMProvider):
    """DeepSeek LLM provider (OpenAI-compatible API)."""
    
    def __init__(self, api_key: str, model: str = "deepseek-chat", temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.client = None
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
            except Exception:
                pass
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.client:
            raise ConnectionError("DeepSeek client not initialized")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content


class MultiLLMClient:
    """Multi-provider LLM client with fallback strategy."""
    
    def __init__(self, settings):
        self.settings = settings
        self.providers: Dict[str, LLMProvider] = {}
        self.provider_order: List[str] = []
        self._init_providers()
    
    def _init_providers(self):
        """Initialize all providers from settings."""
        provider_priority = [
            p.strip().lower() 
            for p in self.settings.LLM_PROVIDER_PRIORITY.split(",")
        ]
        
        print(f"[LLM] Provider priority: {provider_priority}")
        print(f"[LLM] Groq key present: {bool(self.settings.GROQ_API_KEY)}")
        print(f"[LLM] Gemini key present: {bool(self.settings.GEMINI_API_KEY)}")
        print(f"[LLM] DeepSeek key present: {bool(self.settings.DEEPSEEK_API_KEY)}")
        
        for provider_name in provider_priority:
            provider = self._create_provider(provider_name)
            if provider and provider.is_available():
                self.providers[provider_name] = provider
                self.provider_order.append(provider_name)
                print(f"[LLM] {provider_name} initialized successfully")
            else:
                print(f"[LLM] {provider_name} failed to initialize")
        
        if not self.providers:
            raise ValueError("No LLM providers available. Check your API keys.")
    
    def _create_provider(self, name: str) -> Optional[LLMProvider]:
        """Create a provider instance based on name."""
        if name == "groq" and self.settings.GROQ_API_KEY:
            return GroqProvider(
                api_key=self.settings.GROQ_API_KEY,
                model=self.settings.GROQ_MODEL,
                temperature=self.settings.GROQ_TEMPERATURE
            )
        elif name == "gemini" and self.settings.GEMINI_API_KEY:
            return GeminiProvider(
                api_key=self.settings.GEMINI_API_KEY,
                model=self.settings.GEMINI_MODEL,
                temperature=self.settings.GEMINI_TEMPERATURE
            )
        elif name == "deepseek" and self.settings.DEEPSEEK_API_KEY:
            return DeepSeekProvider(
                api_key=self.settings.DEEPSEEK_API_KEY,
                model=self.settings.DEEPSEEK_MODEL,
                temperature=self.settings.DEEPSEEK_TEMPERATURE
            )
        return None
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Generate text with fallback between providers."""
        errors = []
        
        for provider_name in self.provider_order:
            provider = self.providers[provider_name]
            try:
                result = provider.generate(prompt, system_prompt)
                return {
                    "text": result,
                    "provider": provider_name,
                    "success": True
                }
            except Exception as e:
                errors.append(f"{provider_name}: {str(e)}")
                continue
        
        return {
            "text": f"Error: All LLM providers failed. Errors: {'; '.join(errors)}",
            "provider": None,
            "success": False,
            "errors": errors
        }
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names."""
        return self.provider_order.copy()


# Singleton instance
_llm_client = None


def get_llm_client(settings, force_new=False):
    """Get or create the multi-provider LLM client."""
    global _llm_client
    if _llm_client is None or force_new:
        _llm_client = MultiLLMClient(settings)
    return _llm_client


def reset_llm_client():
    """Reset the singleton (useful for config changes)."""
    global _llm_client
    _llm_client = None
