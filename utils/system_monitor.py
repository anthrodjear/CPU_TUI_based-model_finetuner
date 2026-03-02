import os
import time
import threading
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import psutil
from collections import deque

from utils.logger import LoggerMixin, setup_logger
import logging


@dataclass
class SystemMetrics:
    """Container for system resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    swap_percent: float
    disk_usage_percent: float
    process_cpu_percent: float
    process_memory_mb: float


@dataclass
class MetricsHistory:
    """Store historical metrics data."""
    cpu_percent: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_percent: deque = field(default_factory=lambda: deque(maxlen=1000))
    memory_used_gb: deque = field(default_factory=lambda: deque(maxlen=1000))
    timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    process_cpu: deque = field(default_factory=lambda: deque(maxlen=1000))
    process_memory: deque = field(default_factory=lambda: deque(maxlen=1000))


class SystemMonitor(LoggerMixin):
    """Monitor system resources including CPU, RAM, and disk."""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._process = psutil.Process(os.getpid())
        
        self.current_metrics: Optional[SystemMetrics] = None
        self.history = MetricsHistory()
        
        self._callbacks: List[Callable[[SystemMetrics], None]] = []
    
    def start(self):
        """Start the monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.logger.info("System monitor started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.logger.info("System monitor stopped")
    
    def add_callback(self, callback: Callable[[SystemMetrics], None]):
        """Add a callback to be called on each metrics update."""
        self._callbacks.append(callback)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                self._update_metrics()
                for callback in self._callbacks:
                    try:
                        if self.current_metrics is not None:
                            callback(self.current_metrics)
                    except Exception as e:
                        self.logger.warning(f"Callback error: {e}")
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
            
            time.sleep(self.update_interval)
    
    def _update_metrics(self):
        """Update current system metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        disk = psutil.disk_usage('/')
        
        try:
            process_cpu = self._process.cpu_percent()
            process_memory = self._process.memory_info().rss / (1024 * 1024)
        except Exception:
            process_cpu = 0.0
            process_memory = 0.0
        
        self.current_metrics = SystemMetrics(
            timestamp=time.time(),
            cpu_percent=psutil.cpu_percent(interval=0.1),
            memory_percent=mem.percent,
            memory_used_gb=mem.used / (1024 ** 3),
            memory_available_gb=mem.available / (1024 ** 3),
            swap_percent=swap.percent,
            disk_usage_percent=disk.percent,
            process_cpu_percent=process_cpu,
            process_memory_mb=process_memory
        )
        
        self.history.cpu_percent.append(self.current_metrics.cpu_percent)
        self.history.memory_percent.append(self.current_metrics.memory_percent)
        self.history.memory_used_gb.append(self.current_metrics.memory_used_gb)
        self.history.timestamps.append(self.current_metrics.timestamp)
        self.history.process_cpu.append(self.current_metrics.process_cpu_percent)
        self.history.process_memory.append(self.current_metrics.process_memory_mb)
    
    def get_current_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        if self.current_metrics is None:
            self._update_metrics()
        assert self.current_metrics is not None
        return self.current_metrics
    
    def get_total_memory_gb(self) -> float:
        """Get total system memory in GB."""
        return psutil.virtual_memory().total / (1024 ** 3)
    
    def get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        return psutil.virtual_memory().available / (1024 ** 3)
    
    def get_cpu_count(self) -> int:
        """Get number of CPU cores."""
        count = psutil.cpu_count()
        return count if count is not None else 1
    
    def check_memory_sufficient(self, required_gb: float, safety_margin: float = 0.8) -> bool:
        """Check if sufficient memory is available."""
        available = self.get_available_memory_gb()
        return available >= (required_gb / safety_margin)
    
    def estimate_memory_usage(
        self,
        model_size_gb: float,
        batch_size: int,
        seq_length: int,
        lora_r: int,
        hidden_size: int = 4096,
        num_layers: int = 32
    ) -> Dict[str, float]:
        """Estimate memory usage for training."""
        base_memory = model_size_gb
        
        lora_memory = (
            4 * lora_r * hidden_size * num_layers * batch_size * seq_length * 4
        ) / (1024 ** 3)
        
        gradient_memory = model_size_gb * batch_size * seq_length * 4 / (1024 ** 3)
        
        optimizer_memory = model_size_gb * 2
        
        activation_memory = (
            batch_size * seq_length * hidden_size * num_layers * 4 * 2
        ) / (1024 ** 3)
        
        total_estimate = (
            base_memory + lora_memory + gradient_memory + 
            optimizer_memory + activation_memory
        )
        
        return {
            "base_model_gb": base_memory,
            "lora_gb": lora_memory,
            "gradients_gb": gradient_memory,
            "optimizer_gb": optimizer_memory,
            "activations_gb": activation_memory,
            "total_estimate_gb": total_estimate
        }
    
    def get_summary(self) -> Dict:
        """Get a summary of current system state."""
        metrics = self.get_current_metrics()
        return {
            "cpu_percent": metrics.cpu_percent,
            "memory_percent": metrics.memory_percent,
            "memory_used_gb": metrics.memory_used_gb,
            "memory_available_gb": metrics.memory_available_gb,
            "total_memory_gb": self.get_total_memory_gb(),
            "cpu_count": self.get_cpu_count()
        }


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return setup_logger(name)
