import os
import logging
from typing import List, Tuple, Optional, Type, Any
import concurrent.futures
import math
from pydantic import BaseModel
from openai import OpenAI

from model.model import Model

logger = logging.getLogger(__name__)




class OpenaiModel(Model):
    """
    OpenaiModel is a concrete implementation of the Model interface using either Deepseek or OpenAI APIs.
    
    It supports 'deepseek-chat', 'deepseek-reasoner', as well as the common ChatGPT API.
    """
    def __init__(self, 
                 model_name: str, 
                 max_tokens: int = 3000, 
                 api_key: Optional[str] = None,
                 **kwargs):
        """
        Initialize the OpenaiModel instance.

        Args:
            model_name: Model name in the format 'Provider/ModelName'.
            max_tokens: Maximum number of tokens to generate.
            api_key: Optional API key; if not provided, will be read from the environment.
        
        Raises:
            ValueError: If the required API key is not set or the provider is unsupported.
        """
        super().__init__(model_name, max_tokens, **kwargs)
        
        if self.model_type.lower() == "deepseek":
            # For Deepseek, set API key and base URL using OpenAI's compatible interface.
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable DEEPSEEK_API_KEY must be set for Deepseek models.")
            self.base_url = "https://api.deepseek.com"
            self.client = OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        elif self.model_type.lower() == "openai":
            # For OpenAI, set the API key from the environment.
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("Environment variable OPENAI_API_KEY must be set for OpenAI models.")
            self.client = OpenAI(
                api_key=self.api_key,
            )
        else:
            raise ValueError(f"Unsupported model provider: {self.model_type}")

    def generate(self, 
                 prompts: List[Tuple[str, str]],
                 structured_output_class: Optional[Type[BaseModel]] = None) -> List[str]:
        """
        Generate responses for a list of prompt tuples.
        This method returns only the generated responses as strings.
        
        Args:
            prompts: A list of (system message, user message) tuples.
            structured_output_class: Optional Pydantic model class defining the expected structured output.
            
        Returns:
            A list of responses (as strings or serialized structured outputs).
        """
        def process_prompt(prompt):
            system_msg, user_msg = prompt
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            # Call _generate_with_openai_api with return_probs set to False.
            return self._generate_with_openai_api(messages, structured_output_class, return_probs=False, top_logprobs_num=10)[0]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            responses = list(executor.map(process_prompt, prompts))
        
        return responses

    def generate_prob(self, 
                      prompts: List[Tuple[str, str]],
                      structured_output_class: Optional[Type[BaseModel]] = None,
                      top_logprobs_num: int = 10) -> Tuple[List[str], List[Any]]:
        """
        Generate responses along with probability distributions for a list of prompt tuples.
        This method returns a tuple (responses, probs_list), where each element of probs_list
        is the original logprobs from response.choices[0].logprobs.
        
        Args:
            prompts: A list of (system message, user message) tuples.
            structured_output_class: Optional Pydantic model class defining the expected structured output.
            top_logprobs_num: Number of top probability candidates to retrieve (maximum of 20).
            
        Returns:
            A tuple (responses, probs_list) where:
              - responses: a list of generated responses as strings (or serialized structured outputs).
              - probs_list: a list of logprobs objects from the API response.
        """
        # Ensure top_logprobs_num does not exceed 20
        top_logprobs_num = min(top_logprobs_num, 20)
        
        def process_prompt(prompt):
            system_msg, user_msg = prompt
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            return self._generate_with_openai_api(messages, structured_output_class, return_probs=True, top_logprobs_num=top_logprobs_num)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(process_prompt, prompts))
        
        responses = [res[0] for res in results]
        probs_list = [res[1] for res in results]
        return (responses, probs_list)

    def _generate_with_openai_api(self, 
                                  messages: List[dict],
                                  structured_output_class: Optional[Type[BaseModel]] = None,
                                  return_probs: bool = False,
                                  top_logprobs_num: int = 10) -> Tuple[str, Any]:
        """
        Generate a response using OpenAI's ChatCompletion API (or Deepseek), with optional structured outputs
        and probability distribution information.
        
        Args:
            messages: List of messages (each a dict with 'role' and 'content') representing the conversation.
            structured_output_class: Optional Pydantic model class for structured output.
            return_probs: Whether to request probability distribution information.
            top_logprobs_num: Number of top probability candidates to retrieve.
            
        Returns:
            A tuple (generated_response, logprobs) where:
              - generated_response is the generated response as a string (or serialized structured JSON if structured output is used).
              - logprobs is the original logprobs from response.choices[0].logprobs if return_probs is True, otherwise None.
              
        Raises:
            Exception: If the API returns an error or an unexpected response format.
        """
        extra_args = {}
        if structured_output_class is not None:
            extra_args["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": structured_output_class.schema()
                }
            }
        
        if return_probs:
            extra_args["logprobs"] = True
            extra_args["top_logprobs"] = top_logprobs_num
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            stream=False,
            **extra_args
        )
        try:
            message = response.choices[0].message
            logprobs = response.choices[0].logprobs if return_probs else None
            if structured_output_class is not None:
                if hasattr(message, "parsed") and message.parsed is not None:
                    parsed_output = message.parsed
                else:
                    parsed_output = structured_output_class.model_validate_json(message.content)
                output_json = parsed_output.model_dump_json() if isinstance(parsed_output, BaseModel) else str(parsed_output)
                print(f"{self.model_name} API (Structured Output): {output_json}")
                return (output_json, logprobs)
            else:
                print(f"{self.model_name} API: {message.content}")
                return (message.content, logprobs)
        except (KeyError, IndexError, AttributeError, ValueError) as e:
            logger.error(f"Unexpected response format: {response}")
            raise Exception("Unexpected response format from the API")