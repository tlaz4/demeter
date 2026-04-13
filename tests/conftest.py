import sys
from pathlib import Path

# Allow bare imports (from solar import ...) consistent with how the worker runs
sys.path.insert(0, str(Path(__file__).parent.parent / "demeter"))
