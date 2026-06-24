"""
train.py — Thin wrapper entrypoint script for training SmolLM2-135M-Instruct with LoRA.
"""

import sys
from pathlib import Path

from text2cypher.train import main

# Add src/ folder to Python path if running script directly
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if __name__ == "__main__":
    main()
