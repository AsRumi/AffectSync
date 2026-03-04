"""
Pytest configuration.

Adds the project root to sys.path so all imports work
regardless of where pytest is invoked from.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
