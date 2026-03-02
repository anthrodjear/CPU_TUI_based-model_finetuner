import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import os

from utils.logger import LoggerMixin, load_config


class OllamaBridge(LoggerMixin):
    """Bridge for interacting with Ollama CLI."""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self._ollama_path = self._find_ollama()
    
    def _find_ollama(self) -> str:
        """Find the Ollama executable path."""
        config_path = self.config.get('ollama', {}).get('model_install_path')
        if config_path:
            return config_path
        
        ollama_path = shutil.which('ollama')
        if ollama_path:
            return ollama_path
        
        return 'ollama'
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            result = subprocess.run(
                [self._ollama_path, '--version'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List installed Ollama models."""
        try:
            result = subprocess.run(
                [self._ollama_path, 'list'],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode != 0:
                return []
            
            models = []
            lines = result.stdout.strip().split('\n')[1:]
            for line in lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 3:
                        models.append({
                            'name': parts[0],
                            'size': parts[1],
                            'modified': ' '.join(parts[2:])
                        })
            return models
        except Exception as e:
            self.logger.error(f"Error listing models: {e}")
            return []
    
    def create_model(
        self,
        model_name: str,
        modelfile_path: str,
        timeout: int = 600
    ) -> bool:
        """Create a new Ollama model from a Modelfile."""
        try:
            self.logger.info(f"Creating Ollama model: {model_name}")
            result = subprocess.run(
                [self._ollama_path, 'create', model_name, '-f', modelfile_path],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                self.logger.info(f"Model {model_name} created successfully")
                return True
            else:
                self.logger.error(f"Failed to create model: {result.stderr}")
                return False
        except subprocess.TimeoutExpired:
            self.logger.error("Model creation timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error creating model: {e}")
            return False
    
    def delete_model(self, model_name: str) -> bool:
        """Delete an Ollama model."""
        try:
            result = subprocess.run(
                [self._ollama_path, 'rm', model_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Error deleting model: {e}")
            return False
    
    def run_model(
        self,
        model_name: str,
        prompt: str,
        timeout: int = 120
    ) -> Optional[str]:
        """Run inference on a model."""
        try:
            result = subprocess.run(
                [self._ollama_path, 'run', model_name, prompt],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self.logger.error(f"Inference failed: {result.stderr}")
                return None
        except Exception as e:
            self.logger.error(f"Error running model: {e}")
            return None
    
    def generate_modelfile(
        self,
        base_model: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_ctx: int = 4096,
        output_path: Optional[str] = None
    ) -> str:
        """Generate a Modelfile content."""
        template = self.config.get('ollama', {}).get(
            'modelfile_template',
            'FROM {{base_model}}\nPARAMETER temperature {{temperature}}\n'
            'PARAMETER top_p {{top_p}}\nPARAMETER top_k {{top_k}}\n'
            'PARAMETER num_ctx {{num_ctx}}\nPARAMETER num_gpu 0'
        )
        
        modelfile_content = template.format(
            base_model=base_model,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_ctx=num_ctx
        )
        
        if output_path:
            Path(output_path).write_text(modelfile_content)
            self.logger.info(f"Modelfile saved to: {output_path}")
        
        return modelfile_content
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a model."""
        try:
            result = subprocess.run(
                [self._ollama_path, 'show', model_name],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                info = {}
                current_key = None
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
                return info
            return None
        except Exception as e:
            self.logger.error(f"Error getting model info: {e}")
            return None
    
    def pull_model(self, model_name: str, timeout: int = 600) -> bool:
        """Pull a model from Ollama registry."""
        try:
            self.logger.info(f"Pulling model: {model_name}")
            result = subprocess.run(
                [self._ollama_path, 'pull', model_name],
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            return result.returncode == 0
        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
            return False
