import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Dict, Any, List, Callable
from pathlib import Path
import json

from utils.logger import LoggerMixin, load_config
from utils.system_monitor import SystemMonitor
from training.lora_config import LoRAConfig, LoRAConfigManager
from training.checkpoint_manager import CheckpointManager
from training.metrics_tracker import MetricsTracker


class TextDataset(Dataset):
    """Simple text dataset for fine-tuning."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class Trainer(LoggerMixin):
    """Main trainer class for LoRA fine-tuning."""
    
    def __init__(
        self,
        config: Optional[dict] = None,
        system_monitor: Optional[SystemMonitor] = None
    ):
        self.config = config or load_config()
        self.system_monitor = system_monitor or SystemMonitor()
        
        self.lora_config_manager = LoRAConfigManager(self.config)
        self.checkpoint_manager = CheckpointManager(self.config)
        self.metrics_tracker = MetricsTracker(self.config)
        
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        self._setup_cpu()
    
    def _setup_cpu(self):
        """Setup CPU for training."""
        num_threads = self.config.get('cpu', {}).get('num_threads')
        
        if num_threads is None:
            num_threads = self.system_monitor.get_cpu_count()
        
        if num_threads:
            torch.set_num_threads(num_threads)
            self.logger.info(f"Using {num_threads} CPU threads")
        
        self.logger.info("CPU-only mode enabled")
    
    def _resolve_model_path(self, model_path: str) -> str:
        """Resolve model path - handle Ollama models, GGUF files, and HuggingFace paths."""
        import subprocess
        import os
        from pathlib import Path
        
        path_str = str(model_path)
        
        if path_str.startswith('hf.co/'):
            model_name = path_str.replace('hf.co/', '')
            if ':' in model_name:
                model_name = model_name.split(':')[0]
            
            ollama_path = self._find_ollama_gguf(model_name)
            if ollama_path:
                self.logger.info(f"Resolved Ollama GGUF: {path_str} -> {ollama_path}")
                return ollama_path
            
            all_gguf = self._find_all_gguf()
            if all_gguf:
                self.logger.warning(f"Cannot find specific GGUF for {model_name}, using first available: {all_gguf[0]}")
                return all_gguf[0]
            
            self.logger.error(f"No GGUF files found. Please provide a direct path to the GGUF file.")
            raise ValueError(f"Cannot find GGUF for {model_name}. Provide --model with a direct .gguf file path instead.")
        
        if ':' in path_str and not os.path.exists(path_str):
            model_name = path_str.split(':')[0]
            ollama_path = self._find_ollama_gguf(model_name)
            if ollama_path:
                self.logger.info(f"Resolved Ollama model: {path_str} -> {ollama_path}")
                return ollama_path
            
            all_gguf = self._find_all_gguf()
            if all_gguf:
                self.logger.warning(f"Cannot find GGUF for {model_name}, using: {all_gguf[0]}")
                return all_gguf[0]
        
        if os.path.exists(path_str):
            return os.path.abspath(path_str)
        
        all_gguf = self._find_all_gguf()
        if all_gguf:
            self.logger.warning(f"Using first available GGUF: {all_gguf[0]}")
            return all_gguf[0]
        
        return model_path
    
    def _find_all_gguf(self) -> list:
        """Find all GGUF files in Ollama models directory."""
        from pathlib import Path
        import os
        
        if os.name == 'nt':
            base = Path(os.environ.get('USERPROFILE', '~'))
        else:
            base = Path.home()
        
        ollama_models = base / '.ollama' / 'models'
        
        if not ollama_models.exists():
            return []
        
        gguf_files = sorted(
            ollama_models.rglob('*.gguf'),
            key=lambda x: x.stat().st_size,
            reverse=True
        )
        
        return [str(g) for g in gguf_files if g.stat().st_size > 100 * 1024 * 1024]
    
    def _find_ollama_gguf(self, model_name: str) -> Optional[str]:
        """Find GGUF file for an Ollama model."""
        import os
        import re
        from pathlib import Path
        
        if os.name == 'nt':
            base = Path(os.environ.get('USERPROFILE', '~'))
        else:
            base = Path.home()
        
        ollama_models = base / '.ollama' / 'models'
        
        if not ollama_models.exists():
            return None
        
        search_terms = set()
        search_terms.add(model_name.lower())
        search_terms.add(model_name.split('/')[-1].lower())
        
        model_base = model_name.split('/')[-1]
        match = re.match(r'^([a-zA-Z0-9]+[-_]?[0-9]*[a-zA-Z0-9]*)', model_base)
        if match:
            search_terms.add(match.group(1).lower())
        
        key_parts = re.findall(r'[a-zA-Z][a-zA-Z0-9]*[-_]?[0-9]*\.?[0-9]*[bB]?', model_base)
        for part in key_parts:
            if len(part) > 3:
                search_terms.add(part.lower())
        
        self.logger.debug(f"Searching GGUF with terms: {search_terms}")
        
        gguf_files = list(ollama_models.rglob('*.gguf'))
        self.logger.debug(f"Found GGUF files: {[g.name for g in gguf_files]}")
        
        model_gguf = None
        max_size = 0
        
        for gguf in gguf_files:
            gguf_name = str(gguf).lower()
            for term in search_terms:
                if term in gguf_name and len(term) > 3:
                    size = gguf.stat().st_size
                    if size > max_size:
                        max_size = size
                        model_gguf = gguf
                        break
        
        if model_gguf:
            return str(model_gguf)
        
        return None
    
    def prepare_model(
        self,
        model_path: str,
        lora_config: Optional[LoRAConfig] = None
    ):
        """Prepare model for training with LoRA."""
        resolved_path = self._resolve_model_path(model_path)
        self.logger.info(f"Loading model from: {resolved_path}")
        
        if resolved_path.lower().endswith('.gguf'):
            self._prepare_gguf_model(resolved_path, lora_config)
        else:
            self._prepare_hf_model(resolved_path, lora_config)
        
        return self.model, self.tokenizer
    
    def _prepare_gguf_model(self, gguf_path: str, lora_config: Optional[LoRAConfig] = None):
        """Load GGUF model using llama-cpp-python for training."""
        try:
            from llama_cpp import Llama
            from transformers import PreTrainedTokenizerFast
        except ImportError:
            self.logger.error("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
            raise ImportError("llama-cpp-python required for GGUF training. Run: pip install llama-cpp-python")
        
        self.logger.info(f"Loading GGUF model: {gguf_path}")
        
        self.llm = Llama(
            model_path=gguf_path,
            n_ctx=512,
            n_threads=self.config.get('cpu', {}).get('num_threads', 4),
            use_mmap=True,
            use_mlock=False,
        )
        
        tokenizer = self.llm.tokenizer()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.logger.warning("GGUF model loaded. Note: LoRA training with GGUF requires unsloth or manual implementation.")
        self.logger.info("For full LoRA support with GGUF, please convert to HuggingFace format first.")
        
        self.model = None
    
    def _prepare_hf_model(self, model_path: str, lora_config: Optional[LoRAConfig] = None):
        """Load HuggingFace model for training."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map='cpu',
            low_cpu_mem_usage=True
        )
        
        if lora_config is None:
            lora_config = self.lora_config_manager.get_config()
        
        from peft import LoraConfig as PEFTLoRAConfig, get_peft_model, TaskType
        
        task_type = TaskType.CAUSAL_LM
        if lora_config.task_type == "SEQ_CLS":
            task_type = TaskType.SEQ_CLS
        
        peft_config = PEFTLoRAConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=task_type,
            inference_mode=lora_config.inference_mode,
            modules_to_save=lora_config.modules_to_save
        )
        
        self.model = get_peft_model(model, peft_config)
        
        self.model.print_trainable_parameters()
        
        from peft import LoraConfig as PEFTLoRAConfig, get_peft_model, TaskType
        
        task_type = TaskType.CAUSAL_LM
        if lora_config.task_type == "SEQ_CLS":
            task_type = TaskType.SEQ_CLS
        
        peft_config = PEFTLoRAConfig(
            r=lora_config.r,
            lora_alpha=lora_config.lora_alpha,
            lora_dropout=lora_config.lora_dropout,
            target_modules=lora_config.target_modules,
            bias=lora_config.bias,
            task_type=task_type,
            inference_mode=lora_config.inference_mode,
            modules_to_save=lora_config.modules_to_save
        )
        
        self.model = get_peft_model(model, peft_config)
        
        self.model.print_trainable_parameters()
        
        return self.model, self.tokenizer
    
    def prepare_dataset(
        self,
        dataset_path: str,
        max_length: int = 512,
        train_ratio: float = 0.9
    ) -> tuple:
        """Prepare dataset for training."""
        from dataset.dataset_loader import DatasetLoader
        
        loader = DatasetLoader(self.config)
        samples = loader.load(dataset_path)
        
        formatted = loader.create_chat_format(samples)
        texts = [s['text'] for s in formatted]
        
        split_idx = int(len(texts) * train_ratio)
        train_texts = texts[:split_idx]
        eval_texts = texts[split_idx:]
        
        train_dataset = TextDataset(train_texts, self.tokenizer, max_length)
        eval_dataset = TextDataset(eval_texts, self.tokenizer, max_length)
        
        self.logger.info(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval")
        
        return train_dataset, eval_dataset
    
    def train(
        self,
        model_path: str,
        dataset_path: str,
        output_dir: str,
        num_epochs: int = 3,
        batch_size: int = 2,
        max_length: int = 512,
        learning_rate: float = 3e-4,
        lora_config: Optional[LoRAConfig] = None,
        warmup_steps: int = 100,
        save_steps: int = 200,
        eval_steps: int = 100,
        logging_steps: int = 10,
        gradient_accumulation_steps: int = 4,
        resume_from_checkpoint: Optional[str] = None
    ):
        """Run training."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR
        from torch.utils.data import DataLoader
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.metrics_tracker.initialize(str(output_path))
        
        self.prepare_model(model_path, lora_config)
        train_dataset, eval_dataset = self.prepare_dataset(dataset_path, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.config.get('training', {}).get('dataloader_num_workers', 0)
        )
        
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate * 0.1)
        
        start_epoch = 0
        global_step = 0
        
        if resume_from_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            state = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint,
                self.model,
                optimizer,
                scheduler
            )
            start_epoch = state.get('epoch', 0)
            global_step = state.get('step', 0)
        
        self.system_monitor.start()
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(start_epoch, num_epochs):
            self.model.train()
            
            epoch_loss = 0
            step = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    if global_step % logging_steps == 0:
                        current_lr = scheduler.get_last_lr()[0]
                        
                        metrics = self.system_monitor.get_summary()
                        
                        self.metrics_tracker.record_step(
                            step=global_step,
                            epoch=epoch + 1,
                            loss=loss.item() * gradient_accumulation_steps,
                            learning_rate=current_lr,
                            grad_norm=0.0,
                            tokens=input_ids.numel()
                        )
                        
                        self.logger.info(
                            f"Epoch {epoch+1}/{num_epochs} | "
                            f"Step {global_step} | "
                            f"Loss: {loss.item() * gradient_accumulation_steps:.4f} | "
                            f"LR: {current_lr:.6f} | "
                            f"RAM: {metrics['memory_percent']:.1f}%"
                        )
                    
                    if global_step % eval_steps == 0 and len(eval_loader) > 0:
                        eval_loss = self._evaluate(eval_loader)
                        
                        self.logger.info(f"Eval loss: {eval_loss:.4f}")
                    
                    if global_step % save_steps == 0:
                        self.checkpoint_manager.save_checkpoint(
                            job_name=output_path.name,
                            epoch=epoch + 1,
                            step=global_step,
                            model=self.model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            metrics=self.metrics_tracker.get_summary()
                        )
                
                epoch_loss += loss.item()
                step += 1
            
            avg_epoch_loss = epoch_loss / len(train_loader)
            self.logger.info(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
        
        self.system_monitor.stop()
        
        self.metrics_tracker.save_state({
            'model_path': model_path,
            'dataset_path': dataset_path,
            'num_epochs': num_epochs,
            'output_dir': output_dir
        })
        
        self.logger.info("Training completed!")
        
        return self.model
    
    def _evaluate(self, eval_loader) -> float:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['labels']
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        self.model.train()
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_adapter(self, output_path: str):
        """Save the LoRA adapter."""
        if self.model is None:
            raise ValueError("Model not prepared")
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_adapter(str(output_path), "adapter")
        
        if self.tokenizer:
            self.tokenizer.save_pretrained(str(output_path))
        
        self.logger.info(f"Adapter saved to: {output_path}")
