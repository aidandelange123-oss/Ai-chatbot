#!/bin/bash

# Terminal AI Chatbot Launcher
# Color codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "=========================================="
echo "  Starting Terminal AI Chatbot"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python3 is not installed or not in PATH.${NC}"
    echo "Please install Python3 first."
    exit 1
fi

echo -e "${GREEN}[OK]${NC} Python found: $(python3 --version)"
echo ""

# Check if requirements need to be installed
if [ -f "requirements.txt" ]; then
    echo "Checking dependencies..."
    pip3 install -r requirements.txt --quiet
    if [ $? -ne 0 ]; then
        echo -e "${RED}[WARNING] Some dependencies might have failed to install.${NC}"
    else
        echo -e "${GREEN}[OK]${NC} Dependencies verified."
    fi
fi
echo ""

# Create data directory if it doesn't exist
if [ ! -f "data/training_data.json" ]; then
    echo -e "${BLUE}[INFO]${NC} No training data found. Generating demo data..."
    mkdir -p data
fi

echo "=========================================="
echo "  Launching Chatbot..."
echo "=========================================="
echo ""
echo "* To train on custom data, edit 'data/training_data.json'"
echo "* Type 'quit' in the chat to exit."
echo ""

# Run the chatbot with demo mode and light training for quick start
python3 run.py --demo --epochs 3 --samples 20
