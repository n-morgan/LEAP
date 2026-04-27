"""Pytest config: add the optimization/ folder to sys.path so tests can import
metrics, evaluator, etc. with their natural top-level names — same as the CLI
entrypoints in this folder.
"""

import sys
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
