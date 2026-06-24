"""
evaluate.py — Thin wrapper entrypoint script for evaluating
SmolLM2-135M-Instruct Text2Cypher.
"""

import sys
from pathlib import Path

from text2cypher.evaluate import main

# Add src/ folder to Python path if running script directly
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

if __name__ == "__main__":
    main()
