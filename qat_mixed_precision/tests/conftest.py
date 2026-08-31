"""Make the flat qat_mixed_precision/ modules importable from tests/ regardless
of pytest's rootdir/import-mode or where pytest is invoked from."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
