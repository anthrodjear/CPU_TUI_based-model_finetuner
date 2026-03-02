import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import shutil

from utils.logger import LoggerMixin, load_config


class CheckpointManager(LoggerMixin):
    """Manage training checkpoints."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.checkpoint_dir = Path(
            self.config.get('system', {}).get('checkpoint_dir', './checkpoints')
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, job_name: str) -> Path:
        """Get checkpoint directory for a job."""
        job_dir = self.checkpoint_dir / job_name
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir
    
    def save_checkpoint(
        self,
        job_name: str,
        epoch: int,
        step: int,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        metrics: Dict[str, Any],
        additional_state: Optional[Dict[str, Any]] = None
    ) -> str:
        """Save a checkpoint."""
        checkpoint_path = self.get_checkpoint_path(job_name)
        
        checkpoint_file = checkpoint_path / f"checkpoint-epoch{epoch}-step{step}"
        checkpoint_file.mkdir(parents=True, exist_ok=True)
        
        try:
            import torch
            
            torch.save(
                model.state_dict(),
                checkpoint_file / "model.pt"
            )
            
            if optimizer is not None:
                torch.save(
                    optimizer.state_dict(),
                    checkpoint_file / "optimizer.pt"
                )
            
            if scheduler is not None:
                torch.save(
                    scheduler.state_dict(),
                    checkpoint_file / "scheduler.pt"
                )
            
            state = {
                "epoch": epoch,
                "step": step,
                "metrics": metrics,
                "timestamp": str(Path().stat().st_mtime)
            }
            
            if additional_state:
                state.update(additional_state)
            
            with open(checkpoint_file / "training_state.json", 'w') as f:
                json.dump(state, f, indent=2)
            
            self.logger.info(f"Checkpoint saved: {checkpoint_file}")
            return str(checkpoint_file)
        
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise
    
    def load_checkpoint(
        self,
        checkpoint_path: str,
        model: Any,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Load a checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        try:
            import torch
            
            model.load_state_dict(
                torch.load(checkpoint_path / "model.pt", map_location='cpu')
            )
            self.logger.info("Model state loaded")
            
            if optimizer is not None and (checkpoint_path / "optimizer.pt").exists():
                optimizer.load_state_dict(
                    torch.load(checkpoint_path / "optimizer.pt", map_location='cpu')
                )
                self.logger.info("Optimizer state loaded")
            
            if scheduler is not None and (checkpoint_path / "scheduler.pt").exists():
                scheduler.load_state_dict(
                    torch.load(checkpoint_path / "scheduler.pt", map_location='cpu')
                )
                self.logger.info("Scheduler state loaded")
            
            with open(checkpoint_path / "training_state.json", 'r') as f:
                state = json.load(f)
            
            return state
        
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise
    
    def list_checkpoints(self, job_name: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a job."""
        checkpoint_path = self.get_checkpoint_path(job_name)
        
        checkpoints = []
        
        for item in checkpoint_path.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    state_file = item / "training_state.json"
                    if state_file.exists():
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                        checkpoints.append({
                            "path": str(item),
                            "epoch": state.get("epoch", 0),
                            "step": state.get("step", 0),
                            "metrics": state.get("metrics", {})
                        })
                except Exception as e:
                    self.logger.warning(f"Error reading checkpoint {item}: {e}")
        
        checkpoints.sort(key=lambda x: (x["epoch"], x["step"]))
        return checkpoints
    
    def find_latest_checkpoint(self, job_name: str) -> Optional[str]:
        """Find the latest checkpoint for a job."""
        checkpoints = self.list_checkpoints(job_name)
        
        if not checkpoints:
            return None
        
        return checkpoints[-1]["path"]
    
    def find_best_checkpoint(
        self,
        job_name: str,
        metric: str = "eval_loss"
    ) -> Optional[str]:
        """Find the best checkpoint based on a metric."""
        checkpoints = self.list_checkpoints(job_name)
        
        if not checkpoints:
            return None
        
        best_checkpoint = None
        best_metric_value = float('inf')
        
        for cp in checkpoints:
            metrics = cp.get("metrics", {})
            metric_value = metrics.get(metric)
            
            if metric_value is not None and metric_value < best_metric_value:
                best_metric_value = metric_value
                best_checkpoint = cp["path"]
        
        return best_checkpoint
    
    def delete_checkpoint(self, checkpoint_path: str):
        """Delete a checkpoint."""
        path = Path(checkpoint_path)
        
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
            self.logger.info(f"Checkpoint deleted: {checkpoint_path}")
    
    def cleanup_old_checkpoints(
        self,
        job_name: str,
        keep_last: int = 3
    ):
        """Keep only the last N checkpoints."""
        checkpoints = self.list_checkpoints(job_name)
        
        if len(checkpoints) <= keep_last:
            return
        
        for cp in checkpoints[:-keep_last]:
            self.delete_checkpoint(cp["path"])
