import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Union
import logging

from utils.logger import LoggerMixin, load_config


class DatasetLoader(LoggerMixin):
    """Load datasets in various formats."""
    
    SUPPORTED_FORMATS = ['json', 'jsonl', 'txt']
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
    
    def detect_format(self, path: Path) -> str:
        """Detect dataset format from file extension."""
        ext = path.suffix.lower().lstrip('.')
        
        if ext == 'jsonlines':
            return 'jsonl'
        
        if ext in self.SUPPORTED_FORMATS:
            return ext
        
        raise ValueError(f"Unsupported format: {ext}")
    
    def load(
        self,
        dataset_path: str,
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        streaming: bool = False
    ) -> Union[List[Dict[str, Any]], Iterator[Dict[str, Any]]]:
        """Load dataset from file."""
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        format_type = self.detect_format(path)
        
        if streaming:
            return self._load_streaming(path, format_type, instruction_field, input_field, output_field)
        
        if format_type == 'json':
            return self._load_json(path, instruction_field, input_field, output_field)
        elif format_type == 'jsonl':
            return self._load_jsonl(path, instruction_field, input_field, output_field)
        elif format_type == 'txt':
            return self._load_txt(path, instruction_field, input_field, output_field)
        
        raise ValueError(f"Unknown format: {format_type}")
    
    def _load_json(
        self,
        path: Path,
        instruction_field: str,
        input_field: str,
        output_field: str
    ) -> List[Dict[str, Any]]:
        """Load JSON dataset."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return self._normalize_dataset(data, instruction_field, input_field, output_field)
        elif isinstance(data, dict):
            if 'data' in data:
                data = data['data']
            return self._normalize_dataset(data, instruction_field, input_field, output_field)
        
        raise ValueError("Invalid JSON format")
    
    def _load_jsonl(
        self,
        path: Path,
        instruction_field: str,
        input_field: str,
        output_field: str
    ) -> List[Dict[str, Any]]:
        """Load JSONL dataset."""
        data = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        return self._normalize_dataset(data, instruction_field, input_field, output_field)
    
    def _load_txt(
        self,
        path: Path,
        instruction_field: str,
        input_field: str,
        output_field: str
    ) -> List[Dict[str, Any]]:
        """Load plain text dataset."""
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        data = []
        for line in lines:
            line = line.strip()
            if line:
                data.append({
                    instruction_field: "",
                    input_field: "",
                    output_field: line
                })
        
        return data
    
    def _load_streaming(
        self,
        path: Path,
        format_type: str,
        instruction_field: str,
        input_field: str,
        output_field: str
    ) -> Iterator[Dict[str, Any]]:
        """Load dataset in streaming mode."""
        if format_type == 'json':
            raise NotImplementedError("Streaming JSON not supported")
        
        elif format_type == 'jsonl':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        
        elif format_type == 'txt':
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield {
                            instruction_field: "",
                            input_field: "",
                            output_field: line
                        }
    
    def _normalize_dataset(
        self,
        data: List[Dict[str, Any]],
        instruction_field: str,
        input_field: str,
        output_field: str
    ) -> List[Dict[str, Any]]:
        """Normalize dataset to standard format."""
        normalized = []
        
        for item in data:
            if isinstance(item, str):
                normalized.append({
                    instruction_field: "",
                    input_field: "",
                    output_field: item
                })
            elif isinstance(item, dict):
                normalized_item = {
                    instruction_field: item.get(instruction_field, ""),
                    input_field: item.get(input_field, ""),
                    output_field: item.get(output_field, "")
                }
                normalized.append(normalized_item)
        
        return normalized
    
    def create_chat_format(
        self,
        samples: List[Dict[str, Any]],
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output"
    ) -> List[Dict[str, str]]:
        """Convert dataset to chat format."""
        formatted = []
        
        for sample in samples:
            instruction = sample.get(instruction_field, "")
            input_text = sample.get(input_field, "")
            output = sample.get(output_field, "")
            
            if instruction:
                text = f"Instruction: {instruction}\n"
                if input_text:
                    text += f"Input: {input_text}\n"
                text += f"Output: {output}"
            else:
                text = output
            
            formatted.append({
                "text": text
            })
        
        return formatted
