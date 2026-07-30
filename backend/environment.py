from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

ENV_FILE: Path = Path(__file__).resolve().parent / '.env'

_loaded: bool = False


def load_environment() -> bool:
    """
    Load `backend/.env` into the process environment, once.

    Values already present in the real environment are left alone, so a shell
    variable or a value injected by the container runtime still wins over the
    file. That matches how Docker Compose and Vite treat their own `.env` files,
    so all three layers agree on precedence: environment, then file, then the
    default baked into the code.

    :returns: True if a file was found and read.
    """
    global _loaded
    if _loaded:
        return ENV_FILE.is_file()

    _loaded = True
    return load_dotenv(dotenv_path=ENV_FILE, override=False)
