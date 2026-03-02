from typing import Optional, Dict, Any, List
from pathlib import Path

from utils.logger import LoggerMixin, load_config


class ResourceManager(LoggerMixin):
    """Manage system resources and validate training parameters."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._system_monitor = None
    
    @property
    def system_monitor(self):
        """Lazy load system monitor."""
        if self._system_monitor is None:
            from utils.system_monitor import SystemMonitor
            self._system_monitor = SystemMonitor()
        return self._system_monitor
    
    def validate_training_params(
        self,
        model_size_gb: float,
        batch_size: int,
        seq_length: int,
        lora_r: int,
        preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate training parameters against available resources."""
        if preset:
            preset_config = self.config.get('resource_presets', {}).get(preset, {})
            batch_size = preset_config.get('batch_size', batch_size)
            seq_length = preset_config.get('max_seq_length', seq_length)
            lora_r = preset_config.get('lora_r', lora_r)
        
        memory_estimate = self.system_monitor.estimate_memory_usage(
            model_size_gb=model_size_gb,
            batch_size=batch_size,
            seq_length=seq_length,
            lora_r=lora_r
        )
        
        total_estimate = memory_estimate['total_estimate_gb']
        available_memory = self.system_monitor.get_available_memory_gb()
        
        safety_margin = self.config.get('memory', {}).get('safety_margin', 0.8)
        adjusted_available = available_memory * safety_margin
        
        warnings = []
        errors = []
        
        if total_estimate > adjusted_available:
            errors.append(
                f"Insufficient memory: need {total_estimate:.1f}GB, "
                f"available ~{adjusted_available:.1f}GB"
            )
        
        if total_estimate > available_memory * 0.95:
            warnings.append("Critical memory usage - training may crash")
        
        suggested_batch = batch_size
        suggested_seq = seq_length
        suggested_lora_r = lora_r
        
        if errors:
            if batch_size > 1:
                suggested_batch = 1
                warnings.append("Reduced batch size to 1")
            
            if seq_length > 512:
                suggested_seq = 512
                warnings.append("Reduced sequence length to 512")
            
            if lora_r > 8:
                suggested_lora_r = 8
                warnings.append("Reduced LoRA rank to 8")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "memory_estimate": memory_estimate,
            "available_memory": available_memory,
            "suggested_params": {
                "batch_size": suggested_batch,
                "max_seq_length": suggested_seq,
                "lora_r": suggested_lora_r
            }
        }
    
    def get_optimal_params(
        self,
        model_size_gb: float,
        preset: str = "balanced"
    ) -> Dict[str, int]:
        """Get optimal training parameters based on resources."""
        preset_config = self.config.get('resource_presets', {}).get(preset, {})
        
        available_memory = self.system_monitor.get_available_memory_gb()
        
        if available_memory >= 16:
            return {
                "batch_size": preset_config.get('batch_size', 2),
                "max_seq_length": preset_config.get('max_seq_length', 768),
                "lora_r": preset_config.get('lora_r', 16),
                "gradient_accumulation": preset_config.get('gradient_accumulation', 2)
            }
        elif available_memory >= 12:
            return {
                "batch_size": 2,
                "max_seq_length": 512,
                "lora_r": 8,
                "gradient_accumulation": 4
            }
        else:
            return {
                "batch_size": 1,
                "max_seq_length": 512,
                "lora_r": 4,
                "gradient_accumulation": 8
            }


class JobManager(LoggerMixin):
    """Manage fine-tuning jobs."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.resource_manager = ResourceManager(config)
        self.current_job: Optional[Dict[str, Any]] = None
    
    def create_job(
        self,
        model_path: str,
        dataset_path: str,
        job_name: str,
        num_epochs: int = 3,
        batch_size: int = 2,
        max_seq_length: int = 512,
        learning_rate: float = 3e-4,
        lora_r: int = 8,
        lora_alpha: int = 16,
        preset: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new fine-tuning job."""
        from models.model_info import ModelInfo
        import time
        
        model_info = ModelInfo(
            name=Path(model_path).name,
            path=model_path,
            size_gb=1.0,
            quantization=None,
            architecture=None,
            context_length=4096,
            file_type="bin"
        )
        
        validation = self.resource_manager.validate_training_params(
            model_size_gb=model_info.size_gb,
            batch_size=batch_size,
            seq_length=max_seq_length,
            lora_r=lora_r,
            preset=preset
        )
        
        if not validation["valid"]:
            raise ValueError(f"Invalid parameters: {validation['errors']}")
        
        output_dir = Path(self.config.get('system', {}).get('output_dir', './output'))
        job_output = output_dir / job_name
        job_output.mkdir(parents=True, exist_ok=True)
        
        self.current_job = {
            "name": job_name,
            "model_path": model_path,
            "dataset_path": dataset_path,
            "output_dir": str(job_output),
            "num_epochs": num_epochs,
            "batch_size": validation["suggested_params"]["batch_size"],
            "max_seq_length": validation["suggested_params"]["max_seq_length"],
            "learning_rate": learning_rate,
            "lora_r": validation["suggested_params"]["lora_r"],
            "lora_alpha": lora_alpha,
            "status": "created",
            "created_at": time.time()
        }
        
        return self.current_job
    
    def get_job_status(self) -> Dict[str, Any]:
        """Get current job status."""
        return self.current_job or {"status": "no_job"}
    
    def update_job_status(self, status: str, metrics: Optional[Dict] = None):
        """Update job status."""
        import time
        if self.current_job:
            self.current_job["status"] = status
            self.current_job["updated_at"] = time.time()
            if metrics:
                self.current_job["metrics"] = metrics
