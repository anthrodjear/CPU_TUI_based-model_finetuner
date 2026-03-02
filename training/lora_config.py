from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from utils.logger import load_config


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning."""
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    inference_mode: bool = False
    modules_to_save: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
            "inference_mode": self.inference_mode,
            "modules_to_save": self.modules_to_save
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "LoRAConfig":
        """Create from dictionary."""
        return cls(
            r=config.get("r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.05),
            target_modules=config.get("target_modules", ["q_proj", "v_proj"]),
            bias=config.get("bias", "none"),
            task_type=config.get("task_type", "CAUSAL_LM"),
            inference_mode=config.get("inference_mode", False),
            modules_to_save=config.get("modules_to_save")
        )
    
    @classmethod
    def from_preset(cls, preset: str) -> "LoRAConfig":
        """Create configuration from preset."""
        presets = {
            "small": cls(r=4, lora_alpha=8, lora_dropout=0.05),
            "medium": cls(r=8, lora_alpha=16, lora_dropout=0.05),
            "large": cls(r=16, lora_alpha=32, lora_dropout=0.05),
            "xl": cls(r=32, lora_alpha=64, lora_dropout=0.1)
        }
        
        if preset not in presets:
            raise ValueError(f"Unknown preset: {preset}")
        
        return presets[preset]


class LoRAConfigManager:
    """Manager for LoRA configurations."""
    
    PRESETS = {
        "safe": {
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"]
        },
        "balanced": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
        },
        "aggressive": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        }
    }
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.lora_config = self.config.get('lora', {})
    
    def get_config(self, preset: Optional[str] = None) -> LoRAConfig:
        """Get LoRA configuration."""
        if preset and preset in self.PRESETS:
            preset_config = self.PRESETS[preset]
            return LoRAConfig.from_dict(preset_config)
        
        return LoRAConfig.from_dict(self.lora_config)
    
    def get_recommended_config(
        self,
        model_size_gb: float,
        available_memory_gb: float
    ) -> LoRAConfig:
        """Get recommended LoRA configuration based on resources."""
        if model_size_gb <= 3 and available_memory_gb >= 12:
            return self.get_config("aggressive")
        elif model_size_gb <= 7 and available_memory_gb >= 8:
            return self.get_config("balanced")
        else:
            return self.get_config("safe")
    
    def estimate_parameters(
        self,
        r: int,
        target_modules: List[str],
        hidden_size: int = 4096,
        num_layers: int = 32
    ) -> int:
        """Estimate number of trainable LoRA parameters."""
        modules_per_layer = len(target_modules)
        params_per_module = 2 * r * hidden_size
        
        return params_per_module * modules_per_layer * num_layers
