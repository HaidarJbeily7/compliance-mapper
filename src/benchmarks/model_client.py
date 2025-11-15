"""
Unified client for interacting with different LLM providers.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic


class ModelClient:
    """Unified client for GPT-4o mini and Claude Haiku."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        """
        Initialize model client.
        
        Args:
            model_name: Either 'gpt-4o-mini' or 'claude-haiku'
            api_key: API key (if not provided, reads from environment)
        """
        self.model_name = model_name
        
        if model_name == 'gpt-4o-mini':
            self.provider = 'openai'
            self.model_id = 'gpt-4o-mini'
            api_key = api_key or os.getenv('OPENAI_API_KEY')
            self.client = AsyncOpenAI(api_key=api_key)
            
        elif model_name == 'claude-haiku':
            self.provider = 'anthropic'
            self.model_id = 'claude-3-haiku-20240307'
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            self.client = AsyncAnthropic(api_key=api_key)
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate text completion.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        if self.provider == 'openai':
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
            
        elif self.provider == 'anthropic':
            response = await self.client.messages.create(
                model=self.model_id,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt or "",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
    
    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        rate_limit_delay: float = 1.2  # seconds between requests
    ) -> List[str]:
        """
        Generate completions for multiple prompts with rate limiting.
        
        Args:
            prompts: List of input prompts
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            system_prompt: Optional system prompt
            rate_limit_delay: Delay between requests in seconds
            
        Returns:
            List of generated texts
        """
        results = []
        for prompt in prompts:
            result = await self.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            results.append(result)
            await asyncio.sleep(rate_limit_delay)
        
        return results

