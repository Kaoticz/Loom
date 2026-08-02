from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from protocol_export import write_artefact  # noqa: E402

"""
Regenerate the shared protocol artefact.

Run after changing any enum the frontend mirrors, and commit the result:

    python scripts/export_protocol.py
"""

if __name__ == '__main__':
    print(f'Wrote {write_artefact()}')
