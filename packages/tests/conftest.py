# packages/tests/conftest.py
import os

# Force non-interactive backend for *all* tests before anything imports pyplot
os.environ.setdefault("MPLBACKEND", "Agg")

# Optional: silence tqdm monitor threads in tests (prevents noisy shutdowns)
os.environ.setdefault("TQDM_DISABLE", "1")
