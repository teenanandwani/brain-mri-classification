"""One-command download-and-train helper for the public Kaggle dataset.

Optional arguments are forwarded to train.py. This downloads a dataset and runs
training; it does not guarantee clinical accuracy or change any GitHub repo.
"""
import subprocess
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError as exc:
    raise SystemExit("Install dependencies first: pip install -r requirements.txt") from exc

print("Downloading or locating the cached Kaggle dataset...", flush=True)
path = kagglehub.dataset_download("navoneel/brain-mri-images-for-brain-tumor-detection")
print("Dataset path:", path, flush=True)
command = [sys.executable, str(Path(__file__).with_name("train.py")),
           "--data-dir", str(path), "--epochs", "15"] + sys.argv[1:]
subprocess.run(command, check=True)
