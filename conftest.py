import sys
from pathlib import Path

# Make `research_agent` and `eval` importable when pytest runs from the repository root.
sys.path.insert(0, str(Path(__file__).parent))
