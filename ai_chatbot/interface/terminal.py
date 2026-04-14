"""
Terminal Interface Module
Provides interactive terminal UI for the chatbot
"""

import sys
import time
from typing import Optional, List, Dict
from pathlib import Path


class TerminalUI:
    """Terminal-based user interface for the chatbot"""
    
    # ANSI color codes
    COLORS = {
        'RESET': '\033[0m',
        'BOLD': '\033[1m',
        'DIM': '\033[2m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'RED': '\033[91m',
    }
    
    def __init__(self, bot_name: str = "AI Bot"):
        self.bot_name = bot_name
        self.show_colors = True
        self.typing_effect = True
        self.typing_speed = 0.02
    
    def _color(self, text: str, color: str) -> str:
        """Apply color to text"""
        if not self.show_colors:
            return text
        return f"{self.COLORS.get(color, '')}{text}{self.COLORS['RESET']}"
    
    def clear_screen(self) -> None:
        """Clear terminal screen"""
        print("\033[2J\033[H", end="")
    
    def print_header(self) -> None:
        """Print application header"""
        self.clear_screen()
        print(self._color("=" * 60, 'CYAN'))
        print(self._color(f"  {self.bot_name} - Terminal Chat Interface", 'BOLD'))
        print(self._color("=" * 60, 'CYAN'))
        print()
    
    def print_menu(self) -> None:
        """Print main menu options"""
        print(self._color("\n--- Commands ---", 'YELLOW'))
        print("  /help     - Show this help message")
        print("  /clear    - Clear conversation history")
        print("  /history  - Show conversation history")
        print("  /stats    - Show bot statistics")
        print("  /train    - Train the model")
        print("  /save     - Save current model")
        print("  /load     - Load a saved model")
        print("  /config   - Show configuration")
        print("  /quit     - Exit the chatbot")
        print(self._color("----------------\n", 'YELLOW'))
    
    def print_bot_message(self, message: str, use_typing: bool = False) -> None:
        """Print bot message with optional typing effect"""
        print(self._color(f"\n{self.bot_name}: ", 'GREEN'), end="")
        
        if use_typing and self.typing_effect:
            for char in message:
                print(char, end="", flush=True)
                time.sleep(self.typing_speed)
            print()
        else:
            print(message)
    
    def get_user_input(self) -> Optional[str]:
        """Get input from user"""
        try:
            prompt = self._color("\nYou: ", 'BLUE')
            user_input = input(prompt).strip()
            return user_input if user_input else None
        except (EOFError, KeyboardInterrupt):
            return None
    
    def print_error(self, message: str) -> None:
        """Print error message"""
        print(self._color(f"Error: {message}", 'RED'))
    
    def print_success(self, message: str) -> None:
        """Print success message"""
        print(self._color(f"✓ {message}", 'GREEN'))
    
    def print_info(self, message: str) -> None:
        """Print info message"""
        print(self._color(f"ℹ {message}", 'CYAN'))
    
    def print_warning(self, message: str) -> None:
        """Print warning message"""
        print(self._color(f"⚠ {message}", 'YELLOW'))
    
    def show_loading(self, message: str = "Loading...", duration: float = 1.0) -> None:
        """Show loading animation"""
        print(self._color(f"\r{message}", 'DIM'), end="", flush=True)
        time.sleep(duration)
        print("\r" + " " * 50 + "\r", end="", flush=True)
    
    def print_history(self, history: List[Dict[str, str]]) -> None:
        """Print conversation history"""
        if not history:
            self.print_info("No conversation history yet.")
            return
        
        print(self._color("\n--- Conversation History ---", 'MAGENTA'))
        for i, turn in enumerate(history[-10:], 1):  # Last 10 turns
            print(f"{i}. {self._color('You:', 'BLUE')} {turn.get('user', '')}")
            print(f"   {self._color(f'{self.bot_name}:', 'GREEN')} {turn.get('bot', '')}\n")
        print(self._color("--- End History ---", 'MAGENTA'))
    
    def print_stats(self, stats: Dict) -> None:
        """Print bot statistics"""
        print(self._color("\n--- Bot Statistics ---", 'CYAN'))
        print(f"  Status: {'Trained ✓' if stats.get('is_trained') else 'Not Trained'}")
        print(f"  Vocabulary Size: {stats.get('vocab_size', 0):,}")
        print(f"  Training Samples: {stats.get('samples_count', 0):,}")
        print(f"  Conversations: {stats.get('conversation_count', 0)}")
        print(f"  Model Parameters: {stats.get('model_params', 0):,}")
        print(f"  History Length: {stats.get('history_length', 0)}")
        print(self._color("----------------------\n", 'CYAN'))
    
    def print_config(self, config: Dict) -> None:
        """Print configuration"""
        print(self._color("\n--- Configuration ---", 'YELLOW'))
        for key, value in list(config.items())[:10]:
            print(f"  {key}: {value}")
        if len(config) > 10:
            print(f"  ... and {len(config) - 10} more settings")
        print(self._color("---------------------\n", 'YELLOW'))
    
    def confirm(self, message: str) -> bool:
        """Ask for confirmation"""
        response = input(f"{message} (y/n): ").strip().lower()
        return response in ['y', 'yes']
    
    def goodbye(self) -> None:
        """Print goodbye message"""
        print()
        print(self._color("=" * 60, 'CYAN'))
        print(self._color(f"  Thank you for chatting with {self.bot_name}!", 'BOLD'))
        print(self._color("  Goodbye! Have a great day!", 'GREEN'))
        print(self._color("=" * 60, 'CYAN'))
        print()
