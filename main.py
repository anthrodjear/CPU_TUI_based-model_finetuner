#!/usr/bin/env python3
"""
Ollama CPU FineTune Studio - Main CLI Entry Point

A production-grade CPU-only fine-tuning system for Ollama models using LoRA.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import load_config, setup_logger
from utils.system_monitor import SystemMonitor
from models.model_scanner import ModelScanner
from models.ollama_bridge import OllamaBridge
from dataset.dataset_loader import DatasetLoader
from dataset.dataset_analyzer import DatasetAnalyzer
from training.trainer import Trainer
from training.lora_config import LoRAConfig
from core.orchestrator import Orchestrator
from core.resource_manager import ResourceManager


def print_banner():
    """Print application banner."""
    banner = """
╔═══════════════════════════════════════════════════════════╗
║       Ollama CPU FineTune Studio v1.0.0                   ║
║       Production-Grade LoRA Fine-Tuning System           ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(banner)


def cmd_scan_models(args):
    """Scan for available Ollama models."""
    print("\n🔍 Scanning for Ollama models...")
    
    config = load_config(args.config)
    scanner = ModelScanner(config)
    
    models = scanner.scan_models()
    
    if not models:
        print("❌ No models found.")
        return 1
    
    print(f"\n📦 Found {len(models)} model(s):\n")
    print(f"{'Name':<40} {'Size':<12} {'Quant':<10} {'Context':<10}")
    print("-" * 75)
    
    for model in models:
        quant = model.quantization or "unknown"
        print(f"{model.name:<40} {model.size_gb:.2f} GB   {quant:<10} {model.context_length}")
    
    return 0


def cmd_analyze_dataset(args):
    """Analyze a dataset."""
    print(f"\n📊 Analyzing dataset: {args.dataset}")
    
    config = load_config(args.config)
    loader = DatasetLoader(config)
    analyzer = DatasetAnalyzer(config)
    
    samples = loader.load(args.dataset)
    
    analysis = analyzer.analyze(samples)
    
    print(f"\n📈 Dataset Statistics:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Token stats: {analysis['token_stats']}")
    print(f"  Training estimate: {analysis['training_estimate']}")
    
    if analysis['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in analysis['warnings']:
            print(f"  - {warning}")
    
    return 0


def cmd_system_info(args):
    """Display system information."""
    print("\n💻 System Information:")
    
    config = load_config(args.config)
    monitor = SystemMonitor()
    
    summary = monitor.get_summary()
    
    print(f"  CPU Cores: {summary['cpu_count']}")
    print(f"  Total RAM: {summary['total_memory_gb']:.2f} GB")
    print(f"  Available RAM: {summary['memory_available_gb']:.2f} GB")
    print(f"  Current RAM Usage: {summary['memory_percent']:.1f}%")
    print(f"  Current CPU Usage: {summary['cpu_percent']:.1f}%")
    
    return 0


def cmd_train(args):
    """Run training."""
    print("\n🚀 Starting training...")
    
    config = load_config(args.config)
    
    orchestrator = Orchestrator(config)
    
    job_config = {
        'model_path': args.model,
        'dataset_path': args.dataset,
        'output_dir': args.output or './output',
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'max_seq_length': args.max_length,
        'learning_rate': args.lr,
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
    }
    
    try:
        orchestrator.run_training(job_config, args.resume)
        
        print("\n✅ Training completed!")
        
        if args.export:
            print("\n📦 Exporting adapter...")
            export_path = orchestrator.export_adapter(
                job_config,
                args.export,
                merge_base=args.merge
            )
            print(f"Adapter saved to: {export_path}")
        
        return 0
    
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return 1


def cmd_export(args):
    """Export a trained model."""
    print("\n📦 Exporting model...")
    
    config = load_config(args.config)
    orchestrator = Orchestrator(config)
    
    output_path = args.output or './exported_model'
    
    try:
        result = orchestrator.export_adapter(
            {},
            output_path,
            merge_base=args.merge
        )
        
        print(f"✅ Model exported to: {result}")
        return 0
    
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return 1


def cmd_install_ollama(args):
    """Install model to Ollama."""
    print(f"\n📥 Installing to Ollama: {args.model_name}")
    
    config = load_config(args.config)
    orchestrator = Orchestrator(config)
    
    try:
        success = orchestrator.install_in_ollama(
            args.model_path,
            args.model_name,
            temperature=args.temperature
        )
        
        if success:
            print(f"✅ Model installed: {args.model_name}")
            return 0
        else:
            print(f"❌ Installation failed")
            return 1
    
    except Exception as e:
        print(f"❌ Installation failed: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ollama CPU FineTune Studio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s scan                    # Scan for available models
  %(prog)s system-info             # Show system information
  %(prog)s analyze-dataset data.jsonl
  %(prog)s train --model model --dataset data.jsonl
  %(prog)s export --output ./exported_model
        """
    )
    
    parser.add_argument(
        '-c', '--config',
        default=None,
        help='Path to config file'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    parser_scan = subparsers.add_parser('scan', help='Scan for Ollama models')
    parser_scan.set_defaults(func=cmd_scan_models)
    
    parser_analyze = subparsers.add_parser('analyze-dataset', help='Analyze dataset', aliases=['analyse-dataset'])
    parser_analyze.add_argument('dataset', help='Path to dataset file')
    parser_analyze.set_defaults(func=cmd_analyze_dataset)
    
    # Also handle analyse-dataset as alias
    parser_analyse = subparsers.add_parser('analyse-dataset', help='Analyze dataset (alias)', aliases=[])
    parser_analyse.add_argument('dataset', help='Path to dataset file')
    parser_analyse.set_defaults(func=cmd_analyze_dataset)
    
    parser_system = subparsers.add_parser('system-info', help='Show system information')
    parser_system.set_defaults(func=cmd_system_info)
    
    parser_train = subparsers.add_parser('train', help='Run training')
    parser_train.add_argument('--model', required=True, help='Model path or name')
    parser_train.add_argument('--dataset', required=True, help='Dataset path')
    parser_train.add_argument('--output', default='./output', help='Output directory')
    parser_train.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser_train.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser_train.add_argument('--max-length', type=int, default=512, help='Max sequence length')
    parser_train.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser_train.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser_train.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser_train.add_argument('--resume', help='Resume from checkpoint')
    parser_train.add_argument('--export', help='Export path after training')
    parser_train.add_argument('--merge', action='store_true', help='Merge with base model')
    parser_train.set_defaults(func=cmd_train)
    
    parser_export = subparsers.add_parser('export', help='Export trained model')
    parser_export.add_argument('--model-path', required=True, help='Model path')
    parser_export.add_argument('--output', default='./exported_model', help='Output path')
    parser_export.add_argument('--merge', action='store_true', help='Merge with base model')
    parser_export.set_defaults(func=cmd_export)
    
    parser_install = subparsers.add_parser('install-ollama', help='Install model to Ollama')
    parser_install.add_argument('--model-path', required=True, help='Model path')
    parser_install.add_argument('--model-name', required=True, help='Ollama model name')
    parser_install.add_argument('--temperature', type=float, default=0.7, help='Temperature')
    parser_install.set_defaults(func=cmd_install_ollama)
    
    args = parser.parse_args()
    
    print_banner()
    
    if not args.command:
        parser.print_help()
        return 1
    
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
