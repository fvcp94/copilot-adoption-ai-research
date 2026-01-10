import sys
from pathlib import Path

# Add repo root to Python path (fixes Streamlit Cloud imports)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
