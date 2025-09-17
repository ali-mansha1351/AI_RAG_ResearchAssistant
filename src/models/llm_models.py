"""llm model implementation for the research assistant"""

import os
from typing import Optional, Any, Dict
from langchain_core.language_models.llms import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import LLMResult, Generation

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


from config.settings import settings

class GroqLLM(LLM):
    """cutom groq llm wrapper for langchain"""

    client: Any = None
    model_name: str = settings.llm_model
    temperature: float = 0.7
    max_tokens: int = 2000

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not GROQ_AVAILABLE:
            raise ImportError("please install groq packaage")
        
        api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables or settings")
        
        self.client = Groq(api_key=api_key)

    def _call(
            self,
            prompt: str,
            stop: Optional[list] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stop=stop,
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error in Groq LLM call: {str(e)}")
    
    @property
    def _llm_type(self) -> str:
        return "groq"
        
def get_llm(model_type: str = "groq", **kwargs) -> LLM:
    """factory function to get appropiate model for now groq only"""
    if model_type.lower() == "groq" and GROQ_AVAILABLE and settings.groq_api_key:
        return GroqLLM(
            model_name=kwargs.get("model_name", settings.llm_model),
            temperature=kwargs.get("temperature", settings.temperature),
            max_tokens=kwargs.get("max_tokens", settings.max_tokens)
        )
    else: 
        return Mock_LLM()
            
class Mock_LLM(LLM):
    """Mock LLM for testing when no API keys are available."""
    
    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return f"Mock response for: {prompt[:100]}..."
    
    @property
    def _llm_type(self) -> str:
        return "mock"
    
    def _generate(
        self,
        prompts: list,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop, run_manager=run_manager, **kwargs)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)