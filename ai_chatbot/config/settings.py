"""
AI Chatbot Configuration Module
Handles all configuration settings for the chatbot system
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json


class Config:
    """Main configuration class for the AI Chatbot"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.training_dir = self.base_dir / "training"
        
        # Default settings - optimized for low memory environments
        self.settings: Dict[str, Any] = {
            "model_name": "custom_chatbot",
            "embedding_dim": 32,
            "hidden_dim": 64,
            "num_layers": 1,
            "dropout": 0.1,
            "learning_rate": 0.001,
            "batch_size": 4,
            "epochs": 50,
            "max_seq_length": 32,
            "temperature": 0.7,
            "top_k": 50,
            "top_p": 0.95,
            "vocab_size": 10000,
            "save_interval": 10,
            "log_level": "INFO",
        }
        
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
    
    def load_config(self, path: str) -> None:
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            custom_config = json.load(f)
            self.settings.update(custom_config)
    
    def save_config(self, path: str) -> None:
        """Save configuration to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.settings, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value"""
        self.settings[key] = value
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        required_keys = ["model_name", "embedding_dim", "learning_rate"]
        for key in required_keys:
            if key not in self.settings:
                return False
        return True
    
    def __repr__(self) -> str:
        return f"Config({len(self.settings)} settings)"


# Global config instance
global_config = Config()
