from __future__ import annotations

from pathlib import Path

import numpy as np


REFERENCE_DIR = Path(__file__).resolve().parent
CH3CN_REFERENCE_20DIM_LATEST = REFERENCE_DIR / "CH3CN_reference_20dim_latest.npz"


def load_ch3cn_reference_20dim(path=None):
    """Load the saved CH3CN 20-dimensional reference data as a dictionary."""
    ref_path = CH3CN_REFERENCE_20DIM_LATEST if path is None else Path(path)
    with np.load(ref_path) as data:
        return {key: data[key] for key in data.files}
