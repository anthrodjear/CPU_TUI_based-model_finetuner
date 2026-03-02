import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List
import os

from utils.logger import LoggerMixin, load_config


class GGUFConverter(LoggerMixin):
    """Convert models to GGUF format using llama.cpp."""
    
    QUANTIZATION_TYPES = {
        "q4_0": "Q4_0",
        "q4_1": "Q4_1",
        "q5_0": "Q5_0",
        "q5_1": "Q5_1",
        "q8_0": "Q8_0",
        "f16": "F16",
        "f32": "F32"
    }
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.llama_cpp_path = self._find_llama_cpp()
    
    def _find_llama_cpp(self) -> Optional[str]:
        """Find llama.cpp conversion tools."""
        paths_to_check = [
            "./llama.cpp",
            "../llama.cpp",
            "~/.llama.cpp",
            "/usr/local/bin/llama",
            "/usr/bin/llama"
        ]
        
        for path in paths_to_check:
            expanded = os.path.expanduser(path)
            if Path(expanded).exists():
                return expanded
        
        if shutil.which("llama"):
            return "llama"
        
        return None
    
    def is_available(self) -> bool:
        """Check if llama.cpp is available."""
        return self.llama_cpp_path is not None
    
    def convert_to_gguf(
        self,
        model_path: str,
        output_path: str,
        quantization: str = "q5_1",
        use_gpu: bool = False
    ) -> bool:
        """Convert model to GGUF format."""
        if not self.is_available():
            self.logger.warning("llama.cpp not found, skipping GGUF conversion")
            return False
        
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        output_file = output_path / f"model-{quantization}.gguf"
        
        if output_file.exists():
            self.logger.info(f"GGUF file already exists: {output_file}")
            return True
        
        try:
            quant_type = self.QUANTIZATION_TYPES.get(quantization, "Q5_1")
            
            cmd = [
                str(Path(self.llama_cpp_path) / "convert"),
                str(model_path),
                "--outfile", str(output_file),
                "--outtype", quant_type.lower()
            ]
            
            self.logger.info(f"Converting to GGUF: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600
            )
            
            if result.returncode == 0:
                self.logger.info(f"GGUF conversion successful: {output_file}")
                return True
            else:
                self.logger.error(f"GGUF conversion failed: {result.stderr}")
                return False
        
        except subprocess.TimeoutExpired:
            self.logger.error("GGUF conversion timed out")
            return False
        except Exception as e:
            self.logger.error(f"Error during GGUF conversion: {e}")
            return False
    
    def quantize(
        self,
        input_path: str,
        output_path: str,
        quantization: str
    ) -> bool:
        """Quantize a GGUF model."""
        if not self.is_available():
            self.logger.warning("llama.cpp not found, skipping quantization")
            return False
        
        input_path = Path(input_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            quant_type = self.QUANTIZATION_TYPES.get(quantization, "Q5_1")
            
            cmd = [
                str(Path(self.llama_cpp_path) / "quantize"),
                str(input_path),
                str(output_path),
                quant_type
            ]
            
            self.logger.info(f"Quantizing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode == 0:
                self.logger.info(f"Quantization successful: {output_path}")
                return True
            else:
                self.logger.error(f"Quantization failed: {result.stderr}")
                return False
        
        except Exception as e:
            self.logger.error(f"Error during quantization: {e}")
            return False
    
    def get_gguf_info(self, gguf_path: str) -> Dict[str, Any]:
        """Get information about a GGUF file."""
        gguf_path = Path(gguf_path)
        
        if not gguf_path.exists():
            return {"error": "File not found"}
        
        size_mb = gguf_path.stat().st_size / (1024 * 1024)
        
        return {
            "path": str(gguf_path),
            "size_mb": size_mb,
            "size_gb": size_mb / 1024,
            "exists": True
        }
    
    def estimate_quantized_size(
        self,
        original_size_gb: float,
        quantization: str
    ) -> float:
        """Estimate size after quantization."""
        ratios = {
            "q4_0": 0.25,
            "q4_1": 0.28,
            "q5_0": 0.35,
            "q5_1": 0.40,
            "q8_0": 0.55,
            "f16": 1.0,
            "f32": 2.0
        }
        
        ratio = ratios.get(quantization, 0.4)
        return original_size_gb * ratio
    
    def list_available_quantizations(self) -> List[str]:
        """List available quantization types."""
        return list(self.QUANTIZATION_TYPES.keys())
