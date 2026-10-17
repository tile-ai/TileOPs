"""Put ``scripts/lint/`` on the import path, as running one of its scripts does."""

import sys
from pathlib import Path

LINT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "lint"
if str(LINT_DIR) not in sys.path:
    sys.path.insert(0, str(LINT_DIR))
