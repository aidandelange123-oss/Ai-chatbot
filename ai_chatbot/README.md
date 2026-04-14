# AI Chatbot

A complex terminal-based AI chatbot with custom training capabilities built on a transformer architecture.

## Features

- **Custom Training**: Train the chatbot on your own data
- **Transformer Architecture**: Uses multi-head attention and positional encoding
- **Interactive Terminal UI**: Colorful, user-friendly interface with typing effects
- **Conversation History**: Track and review past conversations
- **Model Persistence**: Save and load trained models
- **Configurable**: Extensive configuration options for model hyperparameters

## Project Structure

```
ai_chatbot/
├── run.py                 # Main entry point
├── config/
│   ├── __init__.py
│   └── settings.py        # Configuration management
├── core/
│   ├── __init__.py
│   └── engine.py          # Main chatbot engine
├── data/
│   ├── __init__.py
│   └── data_manager.py    # Data loading and preprocessing
├── models/
│   ├── __init__.py
│   └── model.py           # Transformer model architecture
├── training/
│   ├── __init__.py
│   └── trainer.py         # Training loop and checkpointing
├── interface/
│   ├── __init__.py
│   └── terminal.py        # Terminal UI components
└── utils/
    ├── __init__.py
    └── helpers.py         # Utility functions
```

## Installation

### Prerequisites

- Python 3.8+
- PyTorch

### Install Dependencies

```bash
pip install torch
```

## Usage

### Quick Start (Demo Mode)

Run the chatbot with sample training data:

```bash
cd ai_chatbot
python run.py --demo
```

### Train on Custom Data

1. Prepare your training data in JSON format:
```json
[
    {"text": "Hello, how are you?"},
    {"text": "I'm doing well, thank you!"},
    {"text": "What is machine learning?"}
]
```

2. Train the model:
```bash
python run.py --train --data path/to/your/data.json --epochs 50
```

### Interactive Chat

```bash
python run.py
```

### Command Line Options

```
--train          Train the model before starting chat
--chat           Start chat mode (default)
--demo           Run demo mode with pre-loaded sample data
--data PATH      Path to custom training data (JSON format)
--epochs N       Number of training epochs (default: 20)
--samples N      Number of sample data points (default: 100)
--model PATH     Path to saved model to load
--no-colors      Disable colored output
```

### In-Chat Commands

- `/help` - Show help message
- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/stats` - Show bot statistics
- `/config` - Show configuration
- `/quit` - Exit the chatbot

## Architecture

### Model Components

1. **Positional Encoding**: Adds position information to embeddings
2. **Multi-Head Attention**: Captures relationships between tokens
3. **Feed-Forward Networks**: Processes attention outputs
4. **Transformer Blocks**: Stacked encoder layers
5. **Output Layer**: Generates token predictions

### Training Process

1. **Data Loading**: Load and preprocess training data
2. **Vocabulary Building**: Create token-to-index mappings
3. **Encoding**: Convert text to numerical sequences
4. **Training Loop**: Forward pass, loss calculation, backpropagation
5. **Checkpointing**: Save model weights periodically

## Configuration

Edit `config/settings.py` to customize:

- `embedding_dim`: Model embedding dimension (default: 256)
- `hidden_dim`: Feed-forward hidden dimension (default: 512)
- `num_layers`: Number of transformer layers (default: 4)
- `learning_rate`: Training learning rate (default: 0.001)
- `batch_size`: Training batch size (default: 32)
- `epochs`: Number of training epochs (default: 100)
- `dropout`: Dropout rate (default: 0.1)

## Example: Training on Your Data

```python
from core.engine import ChatbotEngine
from config.settings import Config

# Initialize
config = Config()
engine = ChatbotEngine(config)

# Add your training data
engine.add_training_data("Hello! How can I help you?")
engine.add_training_data("The weather is nice today.")
engine.add_conversation_data(["Hi there!", "Hello! How are you?", "I'm great!"])

# Or load from file
engine.load_data("my_training_data.json")

# Prepare and train
engine.prepare_for_training()
engine.train(epochs=50)

# Save the model
engine.save_model("my_custom_model.pt")

# Chat with your trained bot
engine.chat()
```

## License

MIT License - See LICENSE file for details
