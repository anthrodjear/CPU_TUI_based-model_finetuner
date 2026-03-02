from pathlib import Path
from typing import Optional, Dict, Any
import shutil
import json

from utils.logger import LoggerMixin, load_config


class AdapterExporter(LoggerMixin):
    """Export LoRA adapters."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
    
    def export_adapter(
        self,
        model,
        tokenizer,
        output_dir: str,
        adapter_name: str = "adapter"
    ) -> str:
        """Export LoRA adapter."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            model.save_adapter(str(output_path), adapter_name)
            self.logger.info(f"Adapter saved: {output_path}")
            
            if tokenizer:
                tokenizer.save_pretrained(str(output_path))
                self.logger.info(f"Tokenizer saved: {output_path}")
            
            metadata = {
                "adapter_name": adapter_name,
                "exported_at": str(Path().stat().st_mtime),
                "model_type": "lora"
            }
            
            with open(output_path / "adapter_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Error exporting adapter: {e}")
            raise
    
    def export_merged_model(
        self,
        model,
        tokenizer,
        output_dir: str,
        base_model_path: str
    ) -> str:
        """Export merged model (adapter merged into base)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from peft import PeftModel
            
            if isinstance(model, PeftModel):
                merged_model = model.merge_and_unload()
            else:
                merged_model = model
            
            merged_model.save_pretrained(str(output_path))
            self.logger.info(f"Merged model saved: {output_path}")
            
            if tokenizer:
                tokenizer.save_pretrained(str(output_path))
            
            config_path = Path(base_model_path) / "config.json"
            if config_path.exists():
                shutil.copy(config_path, output_path / "config.json")
            
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Error exporting merged model: {e}")
            raise
    
    def export_safetensors(
        self,
        model,
        output_dir: str
    ) -> str:
        """Export model in safetensors format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        try:
            from safetensors.torch import save_file
            from collections import OrderedDict
            
            state_dict = OrderedDict()
            for name, param in model.named_parameters():
                if param.requires_grad or 'lora' in name.lower():
                    state_dict[name] = param
            
            save_file(state_dict, str(output_path / "model.safetensors"))
            
            self.logger.info(f"Safetensors saved: {output_path}")
            return str(output_path)
        
        except ImportError:
            self.logger.warning("safetensors not installed, skipping")
            return str(output_path)
        except Exception as e:
            self.logger.error(f"Error exporting safetensors: {e}")
            raise
    
    def validate_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """Validate an adapter directory."""
        adapter_path = Path(adapter_path)
        
        if not adapter_path.exists():
            return {"valid": False, "error": "Adapter path does not exist"}
        
        required_files = ["adapter_config.json"]
        
        missing = []
        for file in required_files:
            if not (adapter_path / file).exists():
                missing.append(file)
        
        if missing:
            return {"valid": False, "error": f"Missing files: {missing}"}
        
        try:
            with open(adapter_path / "adapter_config.json", 'r') as f:
                config = json.load(f)
            
            return {
                "valid": True,
                "config": config,
                "path": str(adapter_path)
            }
        
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    def get_adapter_size(self, adapter_path: str) -> float:
        """Get adapter size in MB."""
        adapter_path = Path(adapter_path)
        
        total_size = 0
        for file in adapter_path.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size / (1024 * 1024)
