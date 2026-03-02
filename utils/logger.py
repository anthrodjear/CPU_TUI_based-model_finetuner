import logging
import sys
from pathlib import Path
from typing import Optional, Union
import yaml


def load_config(config_path: Optional[Union[str, Path]] = None) -> dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_default_config() -> dict:
    """Return default configuration if config file not found."""
    return {
        "system": {
            "name": "Ollama CPU FineTune Studio",
            "log_dir": "./logs",
            "checkpoint_dir": "./checkpoints",
            "output_dir": "./output",
            "cache_dir": "./cache"
        },
        "training": {
            "mode": "lora",
            "default_batch_size": 2,
            "default_max_seq_length": 512,
            "default_num_epochs": 3,
            "default_learning_rate": 3e-4,
            "default_warmup_steps": 100,
            "gradient_accumulation_steps": 4,
            "eval_steps": 100,
            "save_steps": 200,
            "logging_steps": 10,
            "dataloader_num_workers": 0,
            "use_gradient_checkpointing": True,
            "use_flash_attention": False
        },
        "lora": {
            "r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "inference_mode": False
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_enabled": True,
            "console_enabled": True
        }
    }


def setup_logger(
    name: str,
    config: Optional[dict] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """Setup and return a configured logger."""
    if config is None:
        config = load_config()
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.get("logging", {}).get("level", "INFO")))
    
    log_format = config.get("logging", {}).get(
        "format",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(log_format)
    
    if config.get("logging", {}).get("console_enabled", True):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if config.get("logging", {}).get("file_enabled", True):
        log_dir = Path(config.get("system", {}).get("log_dir", "./logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        if log_file is None:
            log_file = f"{name}.log"
        
        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """Mixin class to add logging capability to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, '_logger'):
            self._logger = setup_logger(self.__class__.__name__)
        return self._logger
