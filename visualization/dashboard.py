import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Any, List
import json

from utils.logger import LoggerMixin, load_config


class Dashboard(LoggerMixin):
    """Dashboard for visualizing training metrics and saving plots."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.output_dir: Optional[Path] = None
    
    def initialize(self, output_dir: str):
        """Initialize dashboard with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Dashboard initialized: {self.output_dir}")
    
    def save_training_plot(
        self,
        metrics_data: Dict[str, List],
        filename: str = "training_loss.png"
    ):
        """Save training loss plot."""
        if not self.output_dir:
            return
        
        steps = metrics_data.get('steps', [])
        loss_values = metrics_data.get('loss', [])
        
        if not steps or not loss_values:
            self.logger.warning("No data for training plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, loss_values, label='Training Loss', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training plot saved: {output_path}")
    
    def save_eval_plot(
        self,
        metrics_data: Dict[str, List],
        filename: str = "eval_loss.png"
    ):
        """Save evaluation loss plot."""
        if not self.output_dir:
            return
        
        steps = metrics_data.get('steps', [])
        eval_values = metrics_data.get('eval_loss', [])
        
        if not steps or not eval_values:
            self.logger.warning("No data for eval plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, eval_values, label='Eval Loss', linewidth=2, color='orange')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Evaluation Loss Over Time')
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Eval plot saved: {output_path}")
    
    def save_combined_plot(
        self,
        metrics_data: Dict[str, List],
        filename: str = "combined_loss.png"
    ):
        """Save combined training and eval loss plot."""
        if not self.output_dir:
            return
        
        steps = metrics_data.get('steps', [])
        loss_values = metrics_data.get('loss', [])
        eval_values = metrics_data.get('eval_loss', [])
        
        if not steps:
            return
        
        plt.figure(figsize=(10, 6))
        
        if loss_values:
            plt.plot(steps[:len(loss_values)], loss_values, label='Training Loss', linewidth=2)
        
        if eval_values:
            plt.plot(steps[:len(eval_values)], eval_values, label='Eval Loss', linewidth=2, color='orange')
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training vs Evaluation Loss')
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Combined plot saved: {output_path}")
    
    def save_learning_rate_plot(
        self,
        metrics_data: Dict[str, List],
        filename: str = "learning_rate.png"
    ):
        """Save learning rate plot."""
        if not self.output_dir:
            return
        
        steps = metrics_data.get('steps', [])
        lr_values = metrics_data.get('lr', [])
        
        if not steps or not lr_values:
            self.logger.warning("No data for LR plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, lr_values, label='Learning Rate', linewidth=2, color='green')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"LR plot saved: {output_path}")
    
    def save_grad_norm_plot(
        self,
        metrics_data: Dict[str, List],
        filename: str = "grad_norm.png"
    ):
        """Save gradient norm plot."""
        if not self.output_dir:
            return
        
        steps = metrics_data.get('steps', [])
        grad_values = metrics_data.get('grad_norm', [])
        
        if not steps or not grad_values:
            self.logger.warning("No data for grad norm plot")
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(steps, grad_values, label='Gradient Norm', linewidth=2, color='red')
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Over Time')
        plt.grid(True)
        plt.legend()
        
        output_path = self.output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Grad norm plot saved: {output_path}")
    
    def generate_report(
        self,
        summary: Dict[str, Any],
        output_filename: str = "training_report.html"
    ):
        """Generate HTML report."""
        if not self.output_dir:
            return
        
        html_content = self._create_html_report(summary)
        
        output_path = self.output_dir / output_filename
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"HTML report saved: {output_path}")
    
    def _create_html_report(self, summary: Dict[str, Any]) -> str:
        """Create HTML report content."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Training Report - Ollama FineTune Studio</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metric { display: flex; justify-content: space-between; padding: 10px; border-bottom: 1px solid #eee; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { color: #333; }
        .warning { color: #dc3545; font-weight: bold; }
        .success { color: #28a745; font-weight: bold; }
        .plot { margin: 20px 0; text-align: center; }
        img { max-width: 100%; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎓 Ollama FineTune Studio - Training Report</h1>
"""
        
        html += "<h2>Training Summary</h2>"
        
        for key, value in summary.items():
            if isinstance(value, float):
                html += f'<div class="metric"><span class="metric-label">{key}</span><span class="metric-value">{value:.4f}</span></div>'
            else:
                html += f'<div class="metric"><span class="metric-label">{key}</span><span class="metric-value">{value}</span></div>'
        
        if self.output_dir:
            plots = ['training_loss.png', 'eval_loss.png', 'combined_loss.png', 'learning_rate.png', 'grad_norm.png']
            
            html += "<h2>Training Plots</h2>"
            
            for plot in plots:
                plot_path = self.output_dir / plot
                if plot_path.exists():
                    html += f'<div class="plot"><img src="{plot}" alt="{plot}"></div>'
        
        html += """
    </div>
</body>
</html>
"""
        return html
    
    def export_metrics_json(self, metrics_history: List[Dict], filename: str = "metrics.json"):
        """Export metrics to JSON."""
        if not self.output_dir:
            return
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(metrics_history, f, indent=2)
        
        self.logger.info(f"Metrics exported: {output_path}")


def create_dashboard(config: Optional[dict] = None) -> Dashboard:
    """Create a dashboard instance."""
    return Dashboard(config)
