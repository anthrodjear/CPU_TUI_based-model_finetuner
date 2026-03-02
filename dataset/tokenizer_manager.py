from typing import Optional, Dict, Any, List
import torch
from pathlib import Path

from utils.logger import LoggerMixin, load_config


class TokenizerManager(LoggerMixin):
    """Manage tokenizers for models."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._tokenizer_cache: Dict[str, Any] = {}
    
    def load_tokenizer(self, model_path: str, tokenizer_type: str = "auto") -> Any:
        """Load tokenizer for a model."""
        cache_key = f"{model_path}:{tokenizer_type}"
        
        if cache_key in self._tokenizer_cache:
            return self._tokenizer_cache[cache_key]
        
        if tokenizer_type == "auto":
            tokenizer = self._load_auto_tokenizer(model_path)
        elif tokenizer_type == "llama":
            tokenizer = self._load_llama_tokenizer(model_path)
        elif tokenizer_type == "gpt2":
            tokenizer = self._load_gpt2_tokenizer()
        else:
            tokenizer = self._load_auto_tokenizer(model_path)
        
        self._tokenizer_cache[cache_key] = tokenizer
        return tokenizer
    
    def _load_auto_tokenizer(self, model_path: str) -> Any:
        """Load tokenizer automatically."""
        try:
            from transformers import AutoTokenizer
            
            path = Path(model_path)
            if path.is_dir():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            else:
                model_dir = path.parent
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
            
            self._setup_tokenizer(tokenizer)
            return tokenizer
        except Exception as e:
            self.logger.warning(f"AutoTokenizer failed: {e}, trying GGUF tokenizer")
            return self._load_gguf_tokenizer(model_path)
    
    def _load_llama_tokenizer(self, model_path: str) -> Any:
        """Load LLaMA tokenizer."""
        try:
            from transformers import LlamaTokenizer
            
            tokenizer = LlamaTokenizer.from_pretrained(model_path)
            self._setup_tokenizer(tokenizer)
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load LlamaTokenizer: {e}")
            return self._load_auto_tokenizer(model_path)
    
    def _load_gpt2_tokenizer(self) -> Any:
        """Load GPT-2 tokenizer."""
        try:
            from transformers import GPT2Tokenizer
            
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self._setup_tokenizer(tokenizer)
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load GPT2Tokenizer: {e}")
            raise
    
    def _load_gguf_tokenizer(self, model_path: str) -> Any:
        """Load tokenizer from GGUF file."""
        try:
            from transformers import AutoTokenizer
            
            path = Path(model_path)
            model_name = path.stem
            
            cache_dir = self.config.get('system', {}).get('cache_dir', './cache')
            cache_path = Path(cache_dir) / 'tokenizers'
            cache_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"GGUF tokenizer not directly supported, using fallback")
            tokenizer = AutoTokenizer.from_pretrained(
                'gpt2',
                cache_dir=str(cache_path)
            )
            self._setup_tokenizer(tokenizer)
            return tokenizer
        except Exception as e:
            self.logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def _setup_tokenizer(self, tokenizer: Any):
        """Setup tokenizer with proper padding and eos tokens."""
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        if tokenizer.bos_token is None:
            tokenizer.bos_token = "<s>"
        
        if tokenizer.eos_token is None:
            tokenizer.eos_token = "</s>"
    
    def create_chat_template(
        self,
        tokenizer: Any,
        system_message: str = "You are a helpful assistant."
    ) -> Any:
        """Create chat template for tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            if hasattr(tokenizer, 'chat_template'):
                return tokenizer
            
            tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<s>\\n' + 'System: ' + message['content'] + '\\n\\n' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ 'User: ' + message['content'] + '\\n' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ 'Assistant: ' + message['content'] + '</s>\\n' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ 'Assistant:' }}"
                "{% endif %}"
            )
            
            return tokenizer
        except Exception as e:
            self.logger.warning(f"Could not set chat template: {e}")
            return tokenizer
    
    def get_token_count(self, tokenizer: Any, text: str) -> int:
        """Get token count for text."""
        tokens = tokenizer.encode(text)
        return len(tokens)
    
    def batch_tokenize(
        self,
        tokenizer: Any,
        texts: List[str],
        max_length: int = 512,
        truncation: bool = True
    ) -> Dict[str, Any]:
        """Tokenize batch of texts."""
        return tokenizer(
            texts,
            max_length=max_length,
            truncation=truncation,
            padding='max_length',
            return_tensors='pt'
        )
    
    def clear_cache(self):
        """Clear tokenizer cache."""
        self._tokenizer_cache.clear()
