from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import math

from utils.logger import LoggerMixin, load_config


class DatasetAnalyzer(LoggerMixin):
    """Analyze dataset and provide statistics."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
    
    def analyze(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Analyze dataset and return statistics."""
        if not samples:
            raise ValueError("Empty dataset")
        
        total_samples = len(samples)
        
        if tokenizer is not None:
            token_stats = self._analyze_tokens(samples, tokenizer)
        else:
            char_stats = self._analyze_characters(samples)
            token_stats = {
                "estimated_tokens_per_char": 0.25,
                "estimated_total_tokens": int(char_stats["total_chars"] * 0.25),
                "estimated_avg_tokens": int(char_stats["avg_chars"] * 0.25),
                "estimated_max_tokens": int(char_stats["max_chars"] * 0.25)
            }
        
        warnings = self._generate_warnings(total_samples, token_stats)
        
        estimate = self._estimate_training_time(
            total_samples,
            token_stats["estimated_total_tokens"],
            token_stats["estimated_avg_tokens"]
        )
        
        return {
            "total_samples": total_samples,
            "token_stats": token_stats,
            "warnings": warnings,
            "training_estimate": estimate
        }
    
    def _analyze_tokens(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: Any
    ) -> Dict[str, Any]:
        """Analyze dataset using tokenizer."""
        token_counts = []
        
        for sample in samples:
            text = self._extract_text(sample)
            tokens = tokenizer.encode(text)
            token_counts.append(len(tokens))
        
        if not token_counts:
            return {
                "total_tokens": 0,
                "avg_tokens": 0,
                "max_tokens": 0,
                "min_tokens": 0,
                "token_distribution": {}
            }
        
        counter = Counter(token_counts)
        
        return {
            "total_tokens": sum(token_counts),
            "avg_tokens": sum(token_counts) / len(token_counts),
            "max_tokens": max(token_counts),
            "min_tokens": min(token_counts),
            "token_distribution": dict(counter.most_common(10))
        }
    
    def _analyze_characters(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dataset using character count as proxy."""
        char_counts = []
        
        for sample in samples:
            text = self._extract_text(sample)
            char_counts.append(len(text))
        
        if not char_counts:
            return {
                "total_chars": 0,
                "avg_chars": 0,
                "max_chars": 0,
                "min_chars": 0
            }
        
        return {
            "total_chars": sum(char_counts),
            "avg_chars": sum(char_counts) / len(char_counts),
            "max_chars": max(char_counts),
            "min_chars": min(char_counts)
        }
    
    def _extract_text(self, sample: Dict[str, Any]) -> str:
        """Extract text from a sample."""
        if isinstance(sample, str):
            return sample
        
        text_fields = ['text', 'output', 'response', 'completion', 'content']
        
        for field in text_fields:
            if field in sample:
                return str(sample[field])
        
        return str(sample.get('instruction', '')) + ' ' + str(sample.get('input', '')) + ' ' + str(sample.get('output', ''))
    
    def _generate_warnings(
        self,
        total_samples: int,
        token_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate warnings based on analysis."""
        warnings = []
        
        min_samples = self.config.get('dataset', {}).get('min_samples_warning', 100)
        if total_samples < min_samples:
            warnings.append(
                f"Small dataset ({total_samples} samples). Recommended: >{min_samples}"
            )
        
        max_tokens_warning = self.config.get('dataset', {}).get('max_tokens_warning', 2048)
        if token_stats.get('max_tokens', 0) > max_tokens_warning:
            warnings.append(
                f"Sequences may be too long ({token_stats['max_tokens']} tokens). "
                f"Consider truncating to {max_tokens_warning}"
            )
        
        if token_stats.get('avg_tokens', 0) > 1024:
            warnings.append(
                "Long average sequence length. Training may be slow on CPU."
            )
        
        return warnings
    
    def _estimate_training_time(
        self,
        total_samples: int,
        total_tokens: int,
        avg_tokens: int
    ) -> Dict[str, Any]:
        """Estimate training time and resource requirements."""
        batch_size = self.config.get('training', {}).get('default_batch_size', 2)
        gradient_accumulation = self.config.get('training', {}).get('gradient_accumulation_steps', 4)
        
        steps_per_epoch = math.ceil(total_samples / (batch_size * gradient_accumulation))
        
        model_size_gb = 3.0
        
        if model_size_gb <= 3:
            time_per_step = 5
        elif model_size_gb <= 7:
            time_per_step = 15
        else:
            time_per_step = 25
        
        estimated_seconds = steps_per_epoch * time_per_step
        estimated_hours = estimated_seconds / 3600
        
        ram_estimate_gb = model_size_gb * 1.5
        
        return {
            "steps_per_epoch": steps_per_epoch,
            "time_per_step_seconds": time_per_step,
            "estimated_total_hours": estimated_hours,
            "estimated_ram_gb": ram_estimate_gb
        }
    
    def get_data_splits(
        self,
        samples: List[Dict[str, Any]],
        train_ratio: float = 0.9
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset into train and validation sets."""
        split_idx = int(len(samples) * train_ratio)
        
        train_samples = samples[:split_idx]
        val_samples = samples[split_idx:]
        
        return train_samples, val_samples
    
    def compute_perplexity(
        self,
        samples: List[Dict[str, Any]],
        model: Any,
        tokenizer: Any
    ) -> Optional[float]:
        """Compute perplexity on a sample (placeholder)."""
        self.logger.info("Perplexity computation not implemented - requires model forward pass")
        return None
