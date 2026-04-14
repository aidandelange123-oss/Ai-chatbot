#!/usr/bin/env python3
"""
AI Chatbot - Main Entry Point
A complex terminal-based AI chatbot with custom training capabilities

Usage:
    python run.py              - Start interactive chat mode
    python run.py --train      - Train the model with sample data
    python run.py --chat       - Start chat mode explicitly
    python run.py --demo       - Run demo with pre-loaded data
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.engine import ChatbotEngine
from config.settings import Config
from interface.terminal import TerminalUI
from utils.helpers import create_sample_dataset


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='AI Chatbot - Train and chat with your own custom AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                  Start interactive chat
  python run.py --train          Train with sample data
  python run.py --data mydata.json   Train with custom data
  python run.py --epochs 50      Train for 50 epochs
        """
    )
    
    parser.add_argument(
        '--train', 
        action='store_true',
        help='Train the model before starting chat'
    )
    
    parser.add_argument(
        '--chat',
        action='store_true',
        help='Start chat mode (default)'
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode with pre-loaded sample data'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to custom training data (JSON format)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    
    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of sample data points to generate (default: 100)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        help='Path to saved model to load'
    )
    
    parser.add_argument(
        '--no-colors',
        action='store_true',
        help='Disable colored output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Initialize components
    config = Config()
    engine = ChatbotEngine(config)
    ui = TerminalUI(bot_name="AI Assistant")
    
    if args.no_colors:
        ui.show_colors = False
    
    # Print welcome header
    ui.print_header()
    ui.print_info("Welcome to the AI Chatbot System!")
    ui.print_info("This is a complex transformer-based chatbot you can train on your own data.")
    
    # Handle model loading
    if args.model:
        ui.print_info(f"Loading model from {args.model}...")
        try:
            engine.load_model(args.model)
            ui.print_success("Model loaded successfully!")
        except Exception as e:
            ui.print_error(f"Failed to load model: {e}")
    
    # Handle training
    if args.train or args.data or args.demo:
        ui.print_info("\nPreparing training data...")
        
        # Generate or load training data
        if args.demo:
            ui.print_info("Generating sample training data...")
            sample_data_path = str(project_root / "data" / "sample_training.json")
            create_sample_dataset(sample_data_path, num_samples=args.samples)
            engine.load_data(sample_data_path)
        elif args.data:
            if Path(args.data).exists():
                engine.load_data(args.data)
            else:
                ui.print_error(f"Data file not found: {args.data}")
                return
        else:
            # Generate default sample data
            sample_data_path = str(project_root / "data" / "sample_training.json")
            create_sample_dataset(sample_data_path, num_samples=args.samples)
            engine.load_data(sample_data_path)
        
        # Prepare and train
        ui.print_info("\nPreparing model for training...")
        engine.prepare_for_training()
        
        ui.print_info(f"\nStarting training for {args.epochs} epochs...")
        ui.print_warning("Note: Training may take some time depending on your data size.")
        
        try:
            results = engine.train(epochs=args.epochs, batch_size=16)
            ui.print_success("Training completed!")
            
            # Save the trained model
            model_path = engine.save_model()
            ui.print_info(f"Model saved to: {model_path}")
            
        except Exception as e:
            ui.print_error(f"Training failed: {e}")
            ui.print_info("You can still try chat mode, but responses will be random.")
    
    # Show stats
    stats = engine.get_stats()
    ui.print_stats(stats)
    
    # Start interactive chat
    ui.print_menu()
    ui.print_info("Starting chat mode... Type /help for commands.\n")
    
    while True:
        user_input = ui.get_user_input()
        
        if user_input is None:
            ui.goodbye()
            break
        
        # Handle commands
        if user_input.startswith('/'):
            command = user_input.lower().strip()
            
            if command in ['/quit', '/exit', '/bye']:
                ui.goodbye()
                break
            
            elif command == '/help':
                ui.print_menu()
            
            elif command == '/clear':
                engine.conversation_history = []
                ui.print_success("Conversation history cleared.")
            
            elif command == '/history':
                ui.print_history(engine.conversation_history)
            
            elif command == '/stats':
                stats = engine.get_stats()
                ui.print_stats(stats)
            
            elif command == '/config':
                ui.print_config(config.settings)
            
            elif command == '/train':
                ui.print_info("Use --train flag from command line to train.")
            
            elif command == '/save':
                if engine.is_trained:
                    path = engine.save_model()
                    ui.print_success(f"Model saved to: {path}")
                else:
                    ui.print_warning("Model not trained yet.")
            
            else:
                ui.print_warning(f"Unknown command: {command}")
                ui.print_menu()
        
        else:
            # Generate response
            if engine.is_trained:
                response = engine.generate_response(user_input)
                ui.print_bot_message(response, use_typing=True)
            else:
                ui.print_warning("Model not trained yet.")
                ui.print_info("Run with --train flag to train the model first.")
                ui.print_bot_message("I'm not trained yet! Please train me using the --train option.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
