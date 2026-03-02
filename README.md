# Ollama CPU FineTune Studio

A production-grade CPU-only fine-tuning system for Ollama models using LoRA (Low-Rank Adaptation).

## Features

- **Automatic Model Detection**: Scans and detects locally installed Ollama models
- **LoRA Fine-tuning**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **CPU-Only**: Runs entirely on CPU, no GPU required
- **Memory-Aware**: Validates parameters and prevents OOM crashes
- **Rich Visualization**: Real-time TUI dashboard with training metrics
- **Checkpoint Support**: Save/resume training from checkpoints
- **Model Export**: Export adapters, merge with base model, convert to GGUF
- **Ollama Integration**: Automatically install fine-tuned models to Ollama

## Requirements

- Python 3.11+
- 16GB RAM (recommended)
- Ollama installed (optional, for model management)

## Installation

```bash
cd ollama_finetune_studio
pip install -r requirements.txt
```

## Quick Start

### 1. Scan for Models

```bash
python main.py scan
```

### 2. Check System Resources

```bash
python main.py system-info
```

### 3. Analyze Dataset

```bash
python main.py analyze-dataset examples/dataset.json
```

### 4. Run Training

```bash
python main.py train \
    --model /path/to/model \
    --dataset examples/dataset.json \
    --output ./output/my_model \
    --epochs 3 \
    --batch-size 2
```

### 5. Export Adapter

```bash
python main.py export \
    --model-path ./output/my_model \
    --output ./exported_adapter
```

### 6. Install to Ollama

```bash
python main.py install-ollama \
    --model-path ./exported_model \
    --model-name my_finetuned_model
```

## Dataset Format

### JSON Format

```json
[
  {
    "instruction": "Your instruction",
    "input": "Input text (optional)",
    "output": "Expected output"
  }
]
```

### JSONL Format

```jsonl
{"instruction": "Summarize", "input": "Text", "output": "Summary"}
{"instruction": "Explain", "input": "", "output": "Explanation"}
```

### Plain Text Format

```
First text line
Second text line
Third text line
```

## Configuration

Edit `config.yaml` to customize:

- Training parameters (batch size, learning rate, epochs)
- LoRA configuration (rank, alpha, target modules)
- Resource presets (safe, balanced, aggressive)
- Visualization settings
- Export options

## Project Structure

```
ollama_finetune_studio/
├── core/               # Core orchestration
│   ├── orchestrator.py
│   ├── job_manager.py
│   └── resource_manager.py
├── models/             # Model detection & Ollama
│   ├── model_scanner.py
│   ├── model_info.py
│   └── ollama_bridge.py
├── dataset/            # Dataset handling
│   ├── dataset_loader.py
│   ├── dataset_analyzer.py
│   └── tokenizer_manager.py
├── training/           # Training engine
│   ├── trainer.py
│   ├── lora_config.py
│   ├── checkpoint_manager.py
│   └── metrics_tracker.py
├── visualization/      # Dashboard & metrics
│   ├── dashboard.py
│   └── live_metrics.py
├── export/             # Model export
│   ├── adapter_exporter.py
│   └── gguf_converter.py
├── utils/             # Utilities
│   ├── logger.py
│   └── system_monitor.py
├── config.yaml        # Configuration
├── main.py           # CLI entry point
└── examples/         # Example datasets
```

## Memory Recommendations

| Model Size | RAM Required | Batch Size | LoRA Rank |
|------------|---------------|------------|-----------|
| 3B params  | 8GB           | 2          | 8-16      |
| 7B params  | 12GB          | 2          | 8         |
| 8B params  | 16GB          | 1          | 4-8       |

## Troubleshooting

### Out of Memory Errors

- Reduce batch size (`--batch-size 1`)
- Reduce max sequence length (`--max-length 256`)
- Use smaller LoRA rank (`--lora-r 4`)

### Slow Training

- Ensure sufficient RAM
- Use quantized models (Q4, Q5)
- Reduce sequence length

### Model Not Found

- Ensure Ollama is installed
- Check model path exists
- Run `python main.py scan` to see available models

## License

MIT License

## Credits

Built with:
- PyTorch
- HuggingFace Transformers
- PEFT (LoRA)
- Rich (TUI)
