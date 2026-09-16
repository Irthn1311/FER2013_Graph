from __future__ import annotations

import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE_SRC = PROJECT / "research" / "pixel_relational_motif_e0" / "src"
if str(PACKAGE_SRC) not in sys.path:
    sys.path.insert(0, str(PACKAGE_SRC))

from pixel_relational_motif_e0.crs_train_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
