from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time
import threading
from collections import deque

from utils.logger import LoggerMixin, load_config
from utils.system_monitor import SystemMetrics


@dataclass
class LiveMetrics:
    """Container for live training metrics."""
    step: int
    epoch: float
    loss: float
    eval_loss: Optional[float]
    learning_rate: float
    grad_norm: float
    tokens_per_second: float
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    eta_seconds: float


class LiveMetricsDisplay(LoggerMixin):
    """Display live metrics in terminal."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.update_interval = self.config.get('visualization', {}).get('update_interval_ms', 500) / 1000
        
        self._running = False
        self._display_thread: Optional[threading.Thread] = None
        
        self.current_metrics: Optional[LiveMetrics] = None
        self.warnings: List[str] = []
        
        self.loss_history: deque = deque(maxlen=100)
        self.eval_loss_history: deque = deque(maxlen=100)
    
    def start(self):
        """Start the metrics display."""
        if self._running:
            return
        
        self._running = True
        self._display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self._display_thread.start()
    
    def stop(self):
        """Stop the metrics display."""
        self._running = False
        if self._display_thread:
            self._display_thread.join(timeout=2.0)
    
    def update(
        self,
        step: int,
        epoch: float,
        loss: float,
        eval_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        grad_norm: float = 0.0,
        tokens_per_second: float = 0.0,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        memory_used_gb: float = 0.0,
        eta_seconds: float = 0.0
    ):
        """Update current metrics."""
        self.current_metrics = LiveMetrics(
            step=step,
            epoch=epoch,
            loss=loss,
            eval_loss=eval_loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            tokens_per_second=tokens_per_second,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            eta_seconds=eta_seconds
        )
        
        self.loss_history.append(loss)
        if eval_loss is not None:
            self.eval_loss_history.append(eval_loss)
        
        self._check_warnings()
    
    def _check_warnings(self):
        """Check for warning conditions."""
        self.warnings = []
        
        if not self.current_metrics:
            return
        
        if self.current_metrics.memory_percent > 85:
            self.warnings.append(f"HIGH RAM: {self.current_metrics.memory_percent:.1f}%")
        
        if self.current_metrics.grad_norm > 10:
            self.warnings.append(f"GRADIENT EXPLOSION: {self.current_metrics.grad_norm:.2f}")
        
        if len(self.loss_history) >= 10 and len(self.eval_loss_history) >= 10:
            recent_loss = sum(list(self.loss_history)[-10:]) / 10
            recent_eval = sum(list(self.eval_loss_history)[-10:]) / 10
            
            if recent_eval > recent_loss * 1.3:
                self.warnings.append("OVERFITTING DETECTED")
        
        if self.current_metrics.loss > 50:
            self.warnings.append("TRAINING DIVERGENCE")
    
    def _display_loop(self):
        """Main display loop."""
        try:
            from rich.console import Console
            from rich.layout import Layout
            from rich.panel import Panel
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
            from rich.table import Table
            from rich.text import Text
            from rich.live import Live
            
            console = Console()
            
            with Live(console=console, refresh_per_second=2, screen=True) as live:
                while self._running:
                    try:
                        layout = self._build_layout()
                        live.update(layout)
                        time.sleep(self.update_interval)
                    except Exception as e:
                        self.logger.debug(f"Display error: {e}")
                        break
        except ImportError:
            self._simple_display_loop()
    
    def _build_layout(self):
        """Build the display layout."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from rich.layout import Layout
        from rich.progress import Progress
        
        console = Console()
        
        if not self.current_metrics:
            return Panel("Waiting for metrics...", title="Ollama FineTune Studio")
        
        m = self.current_metrics
        
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="white")
        
        metrics_table.add_row("Step", f"{m.step}")
        metrics_table.add_row("Epoch", f"{m.epoch:.2f}")
        metrics_table.add_row("Loss", f"{m.loss:.4f}")
        
        if m.eval_loss is not None:
            metrics_table.add_row("Eval Loss", f"{m.eval_loss:.4f}")
        
        metrics_table.add_row("Learning Rate", f"{m.learning_rate:.6f}")
        metrics_table.add_row("Grad Norm", f"{m.grad_norm:.4f}")
        metrics_table.add_row("Tokens/sec", f"{m.tokens_per_second:.1f}")
        
        system_table = Table(show_header=False, box=None, padding=(0, 2))
        system_table.add_column("Resource", style="cyan")
        system_table.add_column("Usage", style="white")
        
        cpu_color = "green"
        if m.cpu_percent > 80:
            cpu_color = "red"
        elif m.cpu_percent > 60:
            cpu_color = "yellow"
        
        mem_color = "green"
        if m.memory_percent > 85:
            mem_color = "red"
        elif m.memory_percent > 70:
            mem_color = "yellow"
        
        system_table.add_row("CPU", f"[{cpu_color}]{m.cpu_percent:.1f}%[/{cpu_color}]")
        system_table.add_row("RAM", f"[{mem_color}]{m.memory_percent:.1f}% ({m.memory_used_gb:.1f} GB)[/{mem_color}]")
        
        if m.eta_seconds > 0:
            eta_min = int(m.eta_seconds // 60)
            eta_sec = int(m.eta_seconds % 60)
            system_table.add_row("ETA", f"{eta_min}m {eta_sec}s")
        
        warnings_text = ""
        if self.warnings:
            warning_lines = [f"[red]{w}[/red]" for w in self.warnings]
            warnings_text = "\n".join(warning_lines)
        
        main_panel = Panel(
            metrics_table,
            title="[bold blue]Training Metrics[/bold blue]",
            border_style="blue"
        )
        
        system_panel = Panel(
            system_table,
            title="[bold green]System Resources[/bold green]",
            border_style="green"
        )
        
        warning_panel = None
        if warnings_text:
            warning_panel = Panel(
                warnings_text,
                title="[bold red]Warnings[/bold red]",
                border_style="red"
            )
        
        from rich.console import Group
        
        layout = Group(main_panel, system_panel)
        if warning_panel:
            layout = Group(layout, warning_panel)
        
        return Panel(
            layout,
            title="[bold]Ollama CPU FineTune Studio[/bold]",
            border_style="cyan"
        )
    
    def _simple_display_loop(self):
        """Simple text-based display loop."""
        while self._running:
            self._print_simple_status()
            time.sleep(self.update_interval)
    
    def _print_simple_status(self):
        """Print simple status to console."""
        if not self.current_metrics:
            print("\rWaiting for metrics...", end="", flush=True)
            return
        
        m = self.current_metrics
        
        status = (
            f"\rStep: {m.step} | "
            f"Epoch: {m.epoch:.2f} | "
            f"Loss: {m.loss:.4f} | "
            f"LR: {m.learning_rate:.6f} | "
            f"CPU: {m.cpu_percent:.1f}% | "
            f"RAM: {m.memory_percent:.1f}%"
        )
        
        print(status, end="", flush=True)
        
        if self.warnings:
            print(f" | WARNINGS: {', '.join(self.warnings)}", end="", flush=True)


def create_live_display(config: Optional[dict] = None) -> LiveMetricsDisplay:
    """Create a live metrics display."""
    return LiveMetricsDisplay(config)
