import json
import csv
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass, field, asdict

from utils.logger import LoggerMixin, load_config


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int
    epoch: float
    loss: float
    eval_loss: Optional[float] = None
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0
    epoch_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetricsTracker(LoggerMixin):
    """Track and store training metrics."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        
        self.metrics_history: deque = deque(maxlen=10000)
        self.current_metrics: Optional[TrainingMetrics] = None
        
        self.metrics_file: Optional[Path] = None
        self.state_file: Optional[Path] = None
        
        self._start_time: float = 0
        self._step_times: deque = deque(maxlen=100)
        self._tokens_processed: int = 0
    
    def initialize(self, output_dir: str):
        """Initialize metrics tracking."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        metrics_filename = self.config.get('logging', {}).get('metrics_file', 'metrics.csv')
        state_filename = self.config.get('logging', {}).get('training_state_file', 'training_state.json')
        
        self.metrics_file = output_path / metrics_filename
        self.state_file = output_path / state_filename
        
        if self.metrics_file.exists():
            self.metrics_file.unlink()
        
        self._start_time = time.time()
        self.logger.info(f"Metrics tracking initialized: {output_dir}")
    
    def record_step(
        self,
        step: int,
        epoch: float,
        loss: float,
        eval_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
        tokens: int = 0
    ):
        """Record metrics for a training step."""
        current_time = time.time()
        
        if len(self._step_times) > 0:
            step_time = current_time - self._step_times[-1]
            tokens_per_second = tokens / step_time if step_time > 0 else 0.0
        else:
            tokens_per_second = 0.0
        
        self._step_times.append(current_time)
        self._tokens_processed += tokens
        
        self.current_metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens_per_second=tokens_per_second,
            epoch_time=current_time - self._start_time
        )
        
        self.metrics_history.append(self.current_metrics)
        
        if self.metrics_file:
            self._append_to_csv(self.current_metrics)
    
    def _append_to_csv(self, metrics: TrainingMetrics):
        """Append metrics to CSV file."""
        row = {
            'step': metrics.step,
            'epoch': metrics.epoch,
            'loss': metrics.loss,
            'eval_loss': metrics.eval_loss or '',
            'learning_rate': metrics.learning_rate,
            'grad_norm': metrics.grad_norm,
            'tokens_per_second': metrics.tokens_per_second,
            'epoch_time': metrics.epoch_time,
            'timestamp': metrics.timestamp
        }
        
        file_exists = self.metrics_file.exists()
        
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(row)
    
    def save_state(self, state: Dict[str, Any]):
        """Save training state to JSON."""
        if not self.state_file:
            return
        
        state_data = {
            **state,
            'metrics_summary': self.get_summary(),
            'total_steps': len(self.metrics_history),
            'total_tokens': self._tokens_processed
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.debug("Training state saved")
    
    def get_current(self) -> Optional[TrainingMetrics]:
        """Get current metrics."""
        return self.current_metrics
    
    def get_history(self, last_n: Optional[int] = None) -> List[TrainingMetrics]:
        """Get metrics history."""
        if last_n is None:
            return list(self.metrics_history)
        return list(self.metrics_history)[-last_n:]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.metrics_history:
            return {}
        
        losses = [m.loss for m in self.metrics_history if m.loss is not None]
        eval_losses = [m.eval_loss for m in self.metrics_history if m.eval_loss is not None]
        grad_norms = [m.grad_norm for m in self.metrics_history if m.grad_norm > 0]
        
        summary = {
            'total_steps': len(self.metrics_history),
            'total_epochs': self.metrics_history[-1].epoch if self.metrics_history else 0,
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'min_loss': min(losses) if losses else 0,
            'max_loss': max(losses) if losses else 0,
        }
        
        if eval_losses:
            summary['avg_eval_loss'] = sum(eval_losses) / len(eval_losses)
            summary['min_eval_loss'] = min(eval_losses)
        
        if grad_norms:
            summary['avg_grad_norm'] = sum(grad_norms) / len(grad_norms)
            summary['max_grad_norm'] = max(grad_norms)
        
        if self.metrics_history:
            summary['current_lr'] = self.metrics_history[-1].learning_rate
        
        return summary
    
    def detect_overfitting(self, window: int = 10) -> bool:
        """Detect overfitting based on train/eval loss divergence."""
        if len(self.metrics_history) < window * 2:
            return False
        
        recent = list(self.metrics_history)[-window:]
        earlier = list(self.metrics_history)[-window*2:-window]
        
        recent_losses = [m.loss for m in recent if m.loss is not None]
        recent_evals = [m.eval_loss for m in recent if m.eval_loss is not None]
        
        if not recent_losses or not recent_evals:
            return False
        
        recent_train_avg = sum(recent_losses) / len(recent_losses)
        recent_eval_avg = sum(recent_evals) / len(recent_evals)
        
        return recent_eval_avg > recent_train_avg * 1.2
    
    def detect_gradient_explosion(self, threshold: float = 10.0) -> bool:
        """Detect gradient explosion."""
        if not self.current_metrics:
            return False
        
        return self.current_metrics.grad_norm > threshold
    
    def detect_training_divergence(self, threshold: float = 50.0) -> bool:
        """Detect training divergence (loss exploding)."""
        if len(self.metrics_history) < 10:
            return False
        
        recent_losses = [m.loss for m in self.metrics_history[-10:] if m.loss is not None]
        
        if len(recent_losses) < 5:
            return False
        
        return max(recent_losses) > threshold
    
    def get_graph_data(
        self,
        metric: str = "loss",
        last_n: int = 100
    ) -> Dict[str, List]:
        """Get data for plotting graphs."""
        history = self.get_history(last_n)
        
        steps = []
        values = []
        
        for m in history:
            steps.append(m.step)
            if metric == "loss":
                values.append(m.loss)
            elif metric == "eval_loss":
                values.append(m.eval_loss)
            elif metric == "grad_norm":
                values.append(m.grad_norm)
            elif metric == "lr":
                values.append(m.learning_rate)
            elif metric == "tokens_per_sec":
                values.append(m.tokens_per_second)
        
        return {"steps": steps, "values": values}
    
    def export_to_json(self, output_path: str):
        """Export all metrics to JSON."""
        data = {
            "metrics": [asdict(m) for m in self.metrics_history],
            "summary": self.get_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.logger.info(f"Metrics exported to: {output_path}")
