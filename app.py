"""Run MediAI locally or with an explicitly configured deployment bind address."""

import os
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    address = os.getenv("STREAMLIT_SERVER_ADDRESS", "127.0.0.1")
    port = os.getenv("PORT", "8501")
    raise SystemExit(subprocess.call([
        sys.executable, "-m", "streamlit", "run", "Home.py",
        f"--server.address={address}", f"--server.port={port}",
        "--server.headless=true", "--browser.gatherUsageStats=false",
    ], cwd=Path(__file__).resolve().parent))
