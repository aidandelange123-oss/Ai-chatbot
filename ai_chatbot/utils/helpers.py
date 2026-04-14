"""
Utility Functions Module
Helper functions for the chatbot system
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime


def save_json(data: Any, filepath: str, indent: int = 2) -> None:
    """Save data to JSON file"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Any:
    """Load data from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_sample_dataset(output_path: str, num_samples: int = 100) -> None:
    """Create a sample training dataset"""
    
    # Sample conversation patterns
    greetings = [
        "Hello!",
        "Hi there!",
        "Good morning!",
        "Good afternoon!",
        "Hey, how are you?",
        "Greetings!",
    ]
    
    responses_to_greetings = [
        "Hello! How can I help you today?",
        "Hi! Nice to meet you!",
        "Hey there! What's on your mind?",
        "Greetings! How's it going?",
        "Hello! Hope you're having a great day!",
    ]
    
    questions = [
        "What is your name?",
        "How are you?",
        "What can you do?",
        "Tell me something interesting.",
        "What's the weather like?",
        "Do you like learning?",
    ]
    
    responses_to_questions = [
        "I'm an AI chatbot learning from data.",
        "I'm doing well, thank you for asking!",
        "I can chat with you and learn from our conversations.",
        "Did you know that AI models learn patterns from data?",
        "I don't have access to real-time weather data.",
        "Absolutely! Learning is what makes me better.",
    ]
    
    farewells = [
        "Goodbye!",
        "See you later!",
        "Take care!",
        "Bye for now!",
        "Have a great day!",
    ]
    
    samples = []
    
    for i in range(num_samples):
        pattern_type = random.choice(['greeting', 'question', 'farewell', 'statement'])
        
        if pattern_type == 'greeting':
            text = random.choice(greetings) + " " + random.choice(responses_to_greetings)
        elif pattern_type == 'question':
            text = random.choice(questions) + " " + random.choice(responses_to_questions)
        elif pattern_type == 'farewell':
            text = random.choice(farewells)
        else:
            statements = [
                "The sky is blue on a clear day.",
                "Learning new things is exciting.",
                "Technology continues to advance rapidly.",
                "Communication is key to understanding.",
                "Every conversation teaches something new.",
                "Knowledge grows through sharing.",
                "Patterns emerge from repeated interactions.",
                "Context matters in conversations.",
            ]
            text = random.choice(statements)
        
        samples.append({
            'text': text,
            'label': pattern_type,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'id': i,
            }
        })
    
    # Save to file
    save_json(samples, output_path)
    print(f"Created {num_samples} sample training records at {output_path}")


def format_conversation(messages: List[Dict[str, str]]) -> str:
    """Format conversation history as string"""
    formatted = []
    for msg in messages:
        role = msg.get('role', 'user')
        content = msg.get('content', '')
        formatted.append(f"{role}: {content}")
    return "\n".join(formatted)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def count_tokens(text: str) -> int:
    """Simple token counter (word-based)"""
    return len(text.split())


def get_timestamp() -> str:
    """Get current timestamp string"""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def ensure_directory(path: str) -> Path:
    """Ensure directory exists, create if needed"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


class MetricsTracker:
    """Track training and inference metrics"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.start_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Start tracking"""
        self.start_time = datetime.now()
    
    def record(self, name: str, value: float) -> None:
        """Record a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
    
    def average(self, name: str) -> Optional[float]:
        """Get average of recorded metric"""
        if name not in self.metrics or not self.metrics[name]:
            return None
        return sum(self.metrics[name]) / len(self.metrics[name])
    
    def elapsed_time(self) -> float:
        """Get elapsed time since start"""
        if not self.start_time:
            return 0.0
        return (datetime.now() - self.start_time).total_seconds()
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""
        summary = {}
        for name, values in self.metrics.items():
            summary[name] = {
                'count': len(values),
                'avg': sum(values) / len(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
            }
        summary['elapsed_time'] = self.elapsed_time()
        return summary
