from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    path: str
    size_gb: float
    quantization: Optional[str]
    architecture: Optional[str]
    context_length: int
    file_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'path': self.path,
            'size_gb': self.size_gb,
            'quantization': self.quantization,
            'architecture': self.architecture,
            'context_length': self.context_length,
            'file_type': self.file_type
        }
    
    def __str__(self) -> str:
        quant_str = f", {self.quantization}" if self.quantization else ""
        arch_str = f", {self.architecture}" if self.architecture else ""
        return f"{self.name} ({self.size_gb:.2f} GB{quant_str}{arch_str})"


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    model_name: str
    model_path: str
    dataset_path: str
    output_dir: str
    num_epochs: int = 3
    batch_size: int = 2
    max_seq_length: int = 512
    learning_rate: float = 3e-4
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    warmup_steps: int = 100
    save_steps: int = 200
    eval_steps: int = 100
    logging_steps: int = 10
    gradient_accumulation_steps: int = 4
    resume_from_checkpoint: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_path': self.model_path,
            'dataset_path': self.dataset_path,
            'output_dir': self.output_dir,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'max_seq_length': self.max_seq_length,
            'learning_rate': self.learning_rate,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'warmup_steps': self.warmup_steps,
            'save_steps': self.save_steps,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'resume_from_checkpoint': self.resume_from_checkpoint
        }


@dataclass
class ExportConfig:
    """Configuration for model export."""
    base_model_path: str
    adapter_path: str
    output_path: str
    merge_base: bool = True
    quantize: bool = True
    quantization_type: str = "q5_1"
    ollama_model_name: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'base_model_path': self.base_model_path,
            'adapter_path': self.adapter_path,
            'output_path': self.output_path,
            'merge_base': self.merge_base,
            'quantize': self.quantize,
            'quantization_type': self.quantization_type,
            'ollama_model_name': self.ollama_model_name
        }
