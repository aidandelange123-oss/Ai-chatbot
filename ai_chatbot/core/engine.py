"""
Core Engine Module
Main chatbot engine that coordinates all components
"""

import torch
from pathlib import Path
from typing import Optional, List, Dict, Any
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Config, global_config
from data.data_manager import DataManager, DataProcessor
from models.model import ChatbotModel, create_model
from training.trainer import Trainer, ChatDataset
from torch.utils.data import DataLoader


class ChatbotEngine:
    """Main engine coordinating all chatbot components"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or global_config
        self.base_dir = Path(__file__).parent.parent
        
        # Initialize components
        self.data_manager = DataManager(str(self.base_dir / "data"))
        self.model: Optional[ChatbotModel] = None
        self.trainer: Optional[Trainer] = None
        self.processor: Optional[DataProcessor] = None
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        # Model state
        self.is_trained = False
        self.vocab: Optional[Dict[str, int]] = None
    
    def load_data(self, filepath: str, format: str = 'json') -> int:
        """Load training data from file"""
        count = self.data_manager.load_from_file(filepath, format)
        print(f"Loaded {count} samples from {filepath}")
        return count
    
    def add_training_data(self, text: str, label: Optional[str] = None) -> None:
        """Add a single training sample"""
        self.data_manager.add_sample(text, label)
    
    def add_conversation_data(self, messages: List[str]) -> None:
        """Add conversation thread for training"""
        self.data_manager.add_conversation(messages)
    
    def prepare_for_training(self) -> None:
        """Prepare data and model for training"""
        print("Preparing data for training...")
        
        # Build vocabulary and encode data
        encoded_data, self.vocab = self.data_manager.prepare_training_data(
            max_length=self.config.get('max_seq_length', 512)
        )
        
        self.processor = self.data_manager.processor
        
        # Create model
        vocab_size = len(self.vocab)
        print(f"Vocabulary size: {vocab_size}")
        
        self.model = create_model(vocab_size, self.config.settings)
        
        # Create trainer
        self.trainer = Trainer(self.model, self.config.settings)
        
        print("Model created and ready for training")
    
    def train(self, epochs: Optional[int] = None, batch_size: Optional[int] = None) -> Dict:
        """Train the model"""
        if not self.model or not self.trainer:
            raise RuntimeError("Must call prepare_for_training() before training")
        
        batch_size = batch_size or self.config.get('batch_size', 32)
        epochs = epochs or self.config.get('epochs', 100)
        
        # Prepare dataset
        encoded_data = self.data_manager.processor.encode(
            ' '.join([s.text for s in self.data_manager.samples]),
            max_length=self.config.get('max_seq_length', 512)
        )
        
        # For demonstration, create simple dataset
        # In production, you'd want proper train/val split
        dataset = ChatDataset(
            [self.data_manager.processor.encode(s.text) for s in self.data_manager.samples],
            len(self.vocab)
        )
        
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting training for {epochs} epochs...")
        results = self.trainer.train(train_loader, epochs=epochs)
        
        self.is_trained = True
        print("Training completed!")
        
        return results
    
    def save_model(self, path: Optional[str] = None) -> str:
        """Save trained model"""
        if not self.model:
            raise RuntimeError("No model to save")
        
        if path is None:
            models_dir = self.base_dir / "models" / "saved"
            models_dir.mkdir(parents=True, exist_ok=True)
            path = str(models_dir / f"{self.config.get('model_name')}_final.pt")
        
        self.model.save_model(path)
        print(f"Model saved to {path}")
        return path
    
    def load_model(self, path: str, vocab_path: Optional[str] = None) -> None:
        """Load a trained model"""
        vocab_size = self.config.get('vocab_size', 10000)
        
        # Try to load vocab if available
        if vocab_path:
            import json
            with open(vocab_path, 'r') as f:
                self.vocab = json.load(f)
            vocab_size = len(self.vocab)
            self.processor = DataProcessor()
            self.processor.vocab = self.vocab
            self.processor.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Create and load model
        self.model = create_model(vocab_size, self.config.settings)
        self.model.load_model(path)
        
        self.is_trained = True
        print(f"Model loaded from {path}")
    
    def generate_response(self, input_text: str, max_length: int = 100) -> str:
        """Generate a response to input text"""
        if not self.is_trained:
            return "Model not trained yet. Please train the model first."
        
        if not self.processor:
            return "Processor not initialized."
        
        # Encode input
        input_ids = self.processor.encode(input_text, max_length=50)
        input_tensor = torch.tensor([input_ids], dtype=torch.long)
        
        # Generate response
        output_tensor = self.model.generate(
            input_tensor,
            max_length=max_length,
            temperature=self.config.get('temperature', 0.7),
            top_k=self.config.get('top_k', 50),
        )
        
        # Decode response
        response = self.processor.decode(output_tensor[0].tolist())
        
        # Store in history
        self.conversation_history.append({
            'user': input_text,
            'bot': response,
        })
        
        return response
    
    def chat(self) -> None:
        """Interactive chat mode"""
        print("\n" + "="*50)
        print("AI Chatbot - Interactive Mode")
        print("="*50)
        print("Type 'quit' or 'exit' to end conversation")
        print("Type 'history' to see conversation history")
        print("Type 'clear' to clear history")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Bot: Goodbye! Have a great day!")
                    break
                
                if user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                if user_input.lower() == 'clear':
                    self.conversation_history = []
                    print("Conversation history cleared.")
                    continue
                
                # Generate response
                response = self.generate_response(user_input)
                print(f"Bot: {response}")
                
            except KeyboardInterrupt:
                print("\nBot: Goodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def _show_history(self) -> None:
        """Display conversation history"""
        if not self.conversation_history:
            print("No conversation history yet.")
            return
        
        print("\n--- Conversation History ---")
        for i, turn in enumerate(self.conversation_history, 1):
            print(f"{i}. You: {turn['user']}")
            print(f"   Bot: {turn['bot']}\n")
        print("--- End History ---\n")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chatbot statistics"""
        stats = {
            'is_trained': self.is_trained,
            'vocab_size': len(self.vocab) if self.vocab else 0,
            'samples_count': len(self.data_manager),
            'conversation_count': len(self.data_manager.conversations),
            'history_length': len(self.conversation_history),
            'config': dict(self.config.settings),
        }
        
        if self.model:
            stats['model_params'] = sum(p.numel() for p in self.model.parameters())
        
        return stats
    
    def __repr__(self) -> str:
        status = "trained" if self.is_trained else "untrained"
        return f"ChatbotEngine({status}, {len(self.data_manager)} samples)"
