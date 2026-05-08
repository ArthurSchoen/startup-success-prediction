"""Centralised .env loader -- replaces the duplicated snippet in every script."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def load_env() -> Optional[Path]:
    """Find and load the first .env file from common project locations.

    Search order:
        1. Parent of the Code/ directory  (Data_thesis/.env)
        2. The Code/ directory itself      (Code/.env)
        3. ~/Desktop/Data_thesis/.env      (absolute fallback)

    Returns the Path that was loaded, or None if no .env was found.
    """
    candidates = [
        Path(__file__).resolve().parent.parent.parent / ".env",   # Data_thesis/.env
        Path(__file__).resolve().parent.parent / ".env",          # Code/.env
        Path.home() / "Desktop" / "Data_thesis" / ".env",        # absolute fallback
    ]

    for env_path in candidates:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
            print(f"(loaded .env from {env_path})")
            return env_path

    print("(no .env file found)")
    return None
