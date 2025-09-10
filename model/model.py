# model.py
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)

class Model(ABC):
    """
    Abstract base class for Large Language Models.
    Defines the interface that all LLM implementations should follow.
    """
    def __init__(self, model_name: str,
                 max_tokens: int = 20000,
                 temperature: float = 0.6,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.0):
        """
        Initialize an LLM instance.
        
        Args:
            model_name: Name of the model in the format 'Provider/ModelName'
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            seed: Random seed for reproducibility
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty parameter
        """
        if '/' not in model_name:
            if model_name == "deepseek-chat" or model_name == "deepseek-reasoner":
                self.model_type = "deepseek"
                self.model_name = model_name
            elif 'gpt' in model_name:
                self.model_type = "openai"
                self.model_name = model_name
            else:
                raise ValueError("model_name must be in the format 'Provider/ModelName'")
        else:
            self.model_name = model_name.split('/')[1]  # e.g., 'gpt-4o-mini'
            self.model_type = model_name.split('/')[0]  # e.g., 'OpenAI'
        self.model_name_full = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
    
    @abstractmethod
    def generate(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """
        Generate responses for a list of prompts.
        
        Args:
            prompts: List of (system_msg, usr_msg) tuples
            batch_size: Number of prompts to process at once
        
        Returns:
            List of generated responses
        """
        pass
    
    def compute_tokens(self, response: str) -> int:
        """
        Compute the number of tokens in a string using the model's tokenizer.
        
        Args:
            response: The string to compute tokens for
            
        Returns:
            Number of tokens in the string
        """
        try:
            tokens = self.tokenizer.encode(response)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error computing tokens: {str(e)}")
            return 0
    
    def get_model_name(self) -> str:
        """
        Returns the full model name that should be used for logging purposes.
        
        Returns:
            Full model name string
        """
        return self.model_name_full