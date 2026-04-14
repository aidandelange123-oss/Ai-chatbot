"""
Data Management Module
Handles loading, preprocessing, and managing training data
"""

import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
import re


@dataclass
class TrainingSample:
    """Represents a single training sample"""
    text: str
    label: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __hash__(self):
        return hash(self.text)


class DataProcessor:
    """Handles data preprocessing and transformation"""
    
    def __init__(self):
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return tokens
    
    def build_vocab(self, texts: List[str], max_size: int = 10000) -> None:
        """Build vocabulary from texts"""
        word_counts: Dict[str, int] = {}
        
        for text in texts:
            tokens = self.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add special tokens first
        self.vocab = self.special_tokens.copy()
        
        # Add most common words
        for word, _ in sorted_words[:max_size - len(self.special_tokens)]:
            idx = len(self.vocab)
            self.vocab[word] = idx
        
        # Create inverse mapping
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to indices"""
        tokens = self.tokenize(text)
        indices = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens]
        
        # Add BOS and EOS
        indices = [self.vocab['<BOS>']] + indices + [self.vocab['<EOS>']]
        
        # Pad or truncate
        if len(indices) < max_length:
            indices += [self.vocab['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
        
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text"""
        tokens = [self.inv_vocab.get(idx, '<UNK>') for idx in indices]
        tokens = [t for t in tokens if t not in ['<PAD>', '<BOS>', '<EOS>']]
        return ' '.join(tokens)


class DataManager:
    """Manages training data storage and retrieval"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processor = DataProcessor()
        self.samples: List[TrainingSample] = []
        self.conversations: List[List[TrainingSample]] = []
        
    def load_from_file(self, filepath: str, format: str = 'json') -> int:
        """Load data from file"""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        count = 0
        if format == 'json':
            count = self._load_json(path)
        elif format == 'txt':
            count = self._load_txt(path)
        elif format == 'pickle':
            count = self._load_pickle(path)
        
        return count
    
    def _load_json(self, path: Path) -> int:
        """Load data from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        count = 0
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and 'text' in item:
                    sample = TrainingSample(
                        text=item['text'],
                        label=item.get('label'),
                        metadata=item.get('metadata')
                    )
                    self.samples.append(sample)
                    count += 1
                elif isinstance(item, list):
                    # Conversation format
                    conv = [TrainingSample(text=t) if isinstance(t, str) 
                            else TrainingSample(**t) for t in item]
                    self.conversations.append(conv)
                    count += 1
        return count
    
    def _load_txt(self, path: Path) -> int:
        """Load data from text file (one sample per line)"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(TrainingSample(text=line))
        return len(self.samples)
    
    def _load_pickle(self, path: Path) -> int:
        """Load data from pickle file"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, TrainingSample):
                    self.samples.append(item)
        return len(self.samples)
    
    def add_sample(self, text: str, label: Optional[str] = None) -> None:
        """Add a single training sample"""
        sample = TrainingSample(text=text, label=label)
        self.samples.append(sample)
    
    def add_conversation(self, messages: List[str]) -> None:
        """Add a conversation thread"""
        conv = [TrainingSample(text=msg) for msg in messages]
        self.conversations.append(conv)
    
    def save_data(self, filename: str, format: str = 'json') -> None:
        """Save current data to file"""
        filepath = self.data_dir / filename
        
        if format == 'json':
            data = [{'text': s.text, 'label': s.label} for s in self.samples]
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(self.samples, f)
    
    def get_batch(self, batch_size: int, shuffle: bool = False) -> Generator:
        """Get batches of training data"""
        import random
        
        indices = list(range(len(self.samples)))
        if shuffle:
            random.shuffle(indices)
        
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = [self.samples[idx] for idx in batch_indices]
            yield batch
    
    def prepare_training_data(self, max_length: int = 512) -> Tuple:
        """Prepare data for model training"""
        texts = [s.text for s in self.samples]
        self.processor.build_vocab(texts)
        
        encoded_data = []
        for sample in self.samples:
            encoded = self.processor.encode(sample.text, max_length)
            encoded_data.append(encoded)
        
        return encoded_data, self.processor.vocab
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __repr__(self) -> str:
        return f"DataManager(samples={len(self.samples)}, conversations={len(self.conversations)})"
