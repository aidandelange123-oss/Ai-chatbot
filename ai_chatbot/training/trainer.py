"""
Training Module
Handles model training, evaluation, and checkpointing
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging


class ChatDataset(Dataset):
    """PyTorch Dataset for chatbot training"""
    
    def __init__(self, encoded_data: List[List[int]], vocab_size: int):
        self.data = encoded_data
        self.vocab_size = vocab_size
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.data[idx]
        x = torch.tensor(sample[:-1], dtype=torch.long)
        y = torch.tensor(sample[1:], dtype=torch.long)
        return x, y


class Trainer:
    """Main training class for the chatbot model"""
    
    def __init__(self, model: nn.Module, config: Dict, device: Optional[str] = None):
        self.config = config
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = model.to(self.device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training state
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.training_history: List[Dict] = []
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup training logger"""
        logger = logging.getLogger('chatbot_trainer')
        logger.setLevel(getattr(logging, self.config.get('log_level', 'INFO')))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(x)
            
            # Reshape for loss calculation - use smaller chunks to save memory
            batch_size, seq_len, vocab_size = output.shape
            
            # Calculate loss using the criterion (more memory efficient)
            output_flat = output.contiguous().view(-1, vocab_size)
            y_flat = y.contiguous().view(-1)
            loss = self.criterion(output_flat, y_flat)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Progress logging
            if (batch_idx + 1) % 5 == 0:
                avg_loss = total_loss / num_batches
                self.logger.info(f"Batch {batch_idx + 1}/{len(dataloader)} - Loss: {avg_loss:.4f}")
        
        return total_loss / max(num_batches, 1)
    
    def evaluate(self, dataloader: DataLoader) -> float:
        """Evaluate model on validation data"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                output = self.model(x)
                
                batch_size, seq_len, vocab_size = output.shape
                output = output.view(-1, vocab_size)
                y = y.view(-1)
                
                loss = self.criterion(output, y)
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
              epochs: Optional[int] = None) -> Dict:
        """Full training loop"""
        epochs = epochs or self.config.get('epochs', 100)
        save_interval = self.config.get('save_interval', 10)
        
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Free up memory
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        for epoch in range(self.current_epoch, epochs):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                self.scheduler.step(val_loss)
            
            epoch_time = time.time() - start_time
            
            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Loss: {train_loss:.4f}"
                f"{f' - Val Loss: {val_loss:.4f}' if val_loss else ''} - "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save history
            self.training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'time': epoch_time,
            })
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, train_loss)
            
            # Save best model
            if val_loss and val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch + 1, train_loss, best=True)
            
            self.current_epoch = epoch + 1
        
        return {
            'final_train_loss': self.training_history[-1]['train_loss'],
            'best_val_loss': self.best_loss,
            'history': self.training_history,
        }
    
    def save_checkpoint(self, epoch: int, loss: float, best: bool = False) -> None:
        """Save model checkpoint"""
        suffix = '_best' if best else f'_epoch_{epoch}'
        checkpoint_path = self.checkpoint_dir / f"model{suffix}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'training_history': self.training_history,
            'config': self.config,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', [])
        
        self.logger.info(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
        
        return checkpoint
