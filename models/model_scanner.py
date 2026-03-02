import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict
import os

from utils.logger import LoggerMixin, load_config


@dataclass
class ModelInfo:
    """Information about an Ollama model."""
    name: str
    path: str
    size_gb: float
    quantization: Optional[str]
    architecture: Optional[str]
    context_length: int
    file_type: str
    
    def __str__(self) -> str:
        return f"{self.name} ({self.size_gb:.2f} GB, {self.quantization or 'unknown'})"


class ModelScanner(LoggerMixin):
    """Scan and detect Ollama models."""
    
    QUANTIZATION_PATTERNS = {
        'q2_k': r'q2_k',
        'q3_k_s': r'q3_k_s',
        'q3_k_m': r'q3_k_m',
        'q3_k_l': r'q3_k_l',
        'q4_0': r'q4_0',
        'q4_1': r'q4_1',
        'q4_k': r'q4_k',
        'q4_k_s': r'q4_k_s',
        'q4_k_m': r'q4_k_m',
        'q5_0': r'q5_0',
        'q5_1': r'q5_1',
        'q5_k': r'q5_k',
        'q5_k_s': r'q5_k_s',
        'q5_k_m': r'q5_k_m',
        'q6_k': r'q6_k',
        'q8_0': r'q8_0',
        'f16': r'f16',
        'f32': r'f32',
    }
    
    ARCHITECTURE_PATTERNS = {
        'llama': r'llama',
        'mistral': r'mistral',
        'mixtral': r'mixtral',
        'qwen': r'qwen',
        'phi': r'phi',
        'gemma': r'gemma',
        'falcon': r'falcon',
        'stablelm': r'stable',
        'm2': r'm2',
    }
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.models_path = self._get_ollama_models_path()
    
    def _get_ollama_models_path(self) -> Path:
        """Get the Ollama models directory path."""
        config_path = self.config.get('model_detection', {}).get('ollama_models_path')
        
        if config_path:
            return Path(config_path)
        
        if os.name == 'nt':
            base = Path(os.environ.get('USERPROFILE', '~'))
            return base / '.ollama' / 'models'
        else:
            home = Path.home()
            if 'HOME' in os.environ:
                return Path(os.environ.get('HOME', str(home))) / '.ollama' / 'models'
            return home / '.ollama' / 'models'
    
    def scan_models(self) -> List[ModelInfo]:
        """Scan for Ollama models using ollama list command."""
        models = []
        
        # First, try to get models from ollama list command
        ollama_models = self._get_ollama_list_models()
        
        if ollama_models:
            self.logger.info(f"Found {len(ollama_models)} models from 'ollama list'")
            for model_data in ollama_models:
                models.append(model_data)
        
        # Also scan the file system for GGUF files
        file_models = self._scan_for_gguf_files()
        
        # Merge, avoiding duplicates
        existing_names = {m.name for m in models}
        for model in file_models:
            if model.name not in existing_names:
                models.append(model)
        
        models.sort(key=lambda m: m.size_gb, reverse=True)
        self.logger.info(f"Found {len(models)} total models")
        return models
    
    def _get_ollama_list_models(self) -> List[ModelInfo]:
        """Get models from 'ollama list' command."""
        models = []
        
        try:
            # Use shell=True for cross-platform compatibility
            if os.name == 'nt':  # Windows
                cmd = 'ollama list'
            else:  # Linux/Mac
                cmd = 'ollama list'
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                self.logger.warning(f"ollama list failed: {result.stderr}")
                return models
            
            self.logger.info(f"ollama list output: {result.stdout[:200]}")
            
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            
            for line in lines:
                if not line.strip():
                    continue
                
                # Handle variable-width columns - find name (first column) and size (second column)
                # Format: NAME  ID  SIZE  MODIFIED
                # Size can be "4.7GB", "640MB", or "-" for cloud models
                
                parts = line.split()
                if len(parts) < 2:
                    continue
                
                name = parts[0]
                
                # Find size - it's the third element, or "-" for cloud
                size_str = "0B"
                if len(parts) >= 3:
                    size_str = parts[2] if parts[2] != '-' else "0B"
                elif len(parts) == 2:
                    size_str = "0B"
                
                # Parse size (e.g., "4.7GB", "640MB")
                size_gb = self._parse_size(size_str)
                
                # Detect quantization from name
                quantization = self._detect_quantization_from_name(name)
                architecture = self._detect_architecture_from_name(name)
                
                # For Ollama models, we don't have direct file path
                # Use ollama as the "path" indicator
                model_info = ModelInfo(
                    name=name,
                    path=f"ollama://{name}",
                    size_gb=size_gb,
                    quantization=quantization,
                    architecture=architecture,
                    context_length=self._estimate_context_length(size_gb, quantization),
                    file_type="ollama"
                )
                models.append(model_info)
                
        except FileNotFoundError:
            self.logger.warning("ollama command not found")
        except Exception as e:
            self.logger.error(f"Error running ollama list: {e}")
        
        return models
    
    def _parse_size(self, size_str: str) -> float:
        """Parse size string like '4.7GB' or '640MB' to GB."""
        size_str = size_str.upper().strip()
        
        try:
            if 'GB' in size_str:
                return float(size_str.replace('GB', ''))
            elif 'MB' in size_str:
                return float(size_str.replace('MB', '')) / 1024
            elif 'TB' in size_str:
                return float(size_str.replace('TB', '')) * 1024
            elif 'B' in size_str:
                return float(size_str.replace('B', '')) / (1024**3)
            else:
                return float(size_str)
        except (ValueError, AttributeError):
            return 0.0
    
    def _detect_quantization_from_name(self, name: str) -> Optional[str]:
        """Detect quantization from model name."""
        name_lower = name.lower()
        
        for quant, pattern in self.QUANTIZATION_PATTERNS.items():
            if pattern in name_lower:
                return quant
        
        return None
    
    def _detect_architecture_from_name(self, name: str) -> Optional[str]:
        """Detect architecture from model name."""
        name_lower = name.lower()
        
        for arch, pattern in self.ARCHITECTURE_PATTERNS.items():
            if pattern in name_lower:
                return arch
        
        return None
    
    def _scan_for_gguf_files(self) -> List[ModelInfo]:
        """Scan for GGUF model files in common locations."""
        models = []
        search_paths = [
            Path.home() / '.ollama' / 'models',
            Path('/usr/share/ollama'),
            Path('/var/lib/ollama'),
            Path.home() / 'llama.cpp' / 'models',
            Path.home() / 'models',
            Path('./models'),
            Path('../models'),
        ]
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            self.logger.info(f"Scanning: {search_path}")
            
            for model_file in search_path.rglob('*.gguf'):
                try:
                    model_info = self._analyze_model_file(model_file)
                    if model_info and model_info.size_gb > 0.1:
                        models.append(model_info)
                except Exception as e:
                    self.logger.debug(f"Error analyzing {model_file}: {e}")
        
        return models
    
    def _find_model_files(self) -> List[Path]:
        """Find all potential model files."""
        model_files = []
        
        if not self.models_path.exists():
            return model_files
        
        for item in self.models_path.rglob('*'):
            if item.is_file():
                if self._is_model_file(item):
                    model_files.append(item)
        
        return model_files
    
    def _is_model_file(self, path: Path) -> bool:
        """Check if file is likely a model file."""
        model_extensions = {'.gguf', '.bin', '.pt', '.pth', '.safetensors'}
        return path.suffix.lower() in model_extensions
    
    def _analyze_model_file(self, path: Path) -> Optional[ModelInfo]:
        """Analyze a model file and extract information."""
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        
        if size_gb < 0.1:
            return None
        
        name = self._extract_model_name(path)
        quantization = self._detect_quantization(path)
        architecture = self._detect_architecture(path, name)
        context_length = self._estimate_context_length(size_gb, quantization)
        
        return ModelInfo(
            name=name,
            path=str(path),
            size_gb=size_gb,
            quantization=quantization,
            architecture=architecture,
            context_length=context_length,
            file_type=path.suffix.lower().lstrip('.')
        )
    
    def _extract_model_name(self, path: Path) -> str:
        """Extract model name from path."""
        path_str = str(path)
        
        if 'manifests' in path_str:
            return path_str.split('manifests')[-1].strip('/\\')
        
        parts = path_str.split('/')
        for part in reversed(parts):
            if part and not part.startswith('.'):
                part = re.sub(r'\.(gguf|bin|pt|pth|safetensors)$', '', part)
                part = re.sub(r'-(q[0-9]|f[0-9]+)$', '', part)
                return part
        
        return path.stem
    
    def _detect_quantization(self, path: Path) -> Optional[str]:
        """Detect quantization type from filename."""
        path_str = str(path).lower()
        
        for quant, pattern in self.QUANTIZATION_PATTERNS.items():
            if re.search(pattern, path_str):
                return quant
        
        return None
    
    def _detect_architecture(self, path: Path, name: str) -> Optional[str]:
        """Detect model architecture."""
        search_str = f"{path} {name}".lower()
        
        for arch, pattern in self.ARCHITECTURE_PATTERNS.items():
            if re.search(pattern, search_str):
                return arch
        
        return None
    
    def _estimate_context_length(self, size_gb: float, quantization: Optional[str]) -> int:
        """Estimate context length based on model size and quantization."""
        base_context = 4096
        
        if 'q2' in str(quantization):
            return 2048
        elif 'q3' in str(quantization):
            return 3072
        elif 'q4' in str(quantization):
            return 4096
        elif 'q5' in str(quantization):
            return 4096
        elif 'q6' in str(quantization):
            return 4096
        elif 'q8' in str(quantization):
            return 4096
        elif 'f16' in str(quantization):
            return 4096
        
        if size_gb > 20:
            return 8192
        elif size_gb > 10:
            return 4096
        elif size_gb > 5:
            return 4096
        else:
            return 2048
    
    def validate_model(self, model: ModelInfo) -> Dict[str, any]:
        """Validate a model for fine-tuning."""
        warnings = []
        errors = []
        
        size_gb = model.size_gb
        
        if size_gb > 8:
            warnings.append(f"Large model ({size_gb:.1f}GB). Training may be slow.")
        
        if size_gb > 16:
            errors.append(f"Model too large for CPU training ({size_gb:.1f}GB)")
        
        if model.quantization and 'q2' in model.quantization:
            warnings.append("Q2 quantization may degrade fine-tuning quality")
        
        if not model.quantization:
            warnings.append("Full precision model - large RAM usage expected")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def estimate_model_parameters(self, model: ModelInfo) -> int:
        """Estimate number of parameters based on model size."""
        size_gb = model.size_gb
        
        if model.quantization:
            quant_bits = 16
            if 'q2' in model.quantization:
                quant_bits = 2
            elif 'q3' in model.quantization:
                quant_bits = 3
            elif 'q4' in model.quantization:
                quant_bits = 4
            elif 'q5' in model.quantization:
                quant_bits = 5
            elif 'q6' in model.quantization:
                quant_bits = 6
            elif 'q8' in model.quantization:
                quant_bits = 8
            
            estimated_params = int(size_gb * 1024 / (quant_bits / 8) * 1.1)
        else:
            estimated_params = int(size_gb * 1024 / 0.125)
        
        return estimated_params
