"""
HEIST inference — root-level entry point.

Usage:
    python inference.py                  # heuristic, direct mode
    python inference.py --seed 42
    python inference.py --max-steps 30
    python inference.py --direct         # no server needed
    python inference.py --quiet          # suppress step output
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))

from inference import main  # noqa: F401

if __name__ == "__main__":
    main()
