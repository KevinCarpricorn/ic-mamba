"""Minimal utilities for working with the M4 dataset.

These definitions provide the small subset of functionality required by
`Dataset_SS` and the evaluation utilities.  They expect the dataset to be
stored as a NumPy `.npz` file containing three arrays:
    - ``values``: array of shape (N, T) with the time series values.
    - ``groups``: array of shape (N,) with the seasonal pattern labels
      (e.g. 'Yearly', 'Monthly').
    - ``ids``: array of shape (N,) with an identifier string for each
      series.
The exact preprocessing of the official M4 dataset is left to the user.
"""
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


class M4Meta:
    """Metadata describing the M4 dataset frequencies and horizons."""

    seasonal_patterns = [
        "Yearly",
        "Quarterly",
        "Monthly",
        "Weekly",
        "Daily",
        "Hourly",
    ]

    horizons_map: Dict[str, int] = {
        "Yearly": 13,
        "Quarterly": 8,
        "Monthly": 18,
        "Weekly": 13,
        "Daily": 14,
        "Hourly": 48,
    }

    frequency_map: Dict[str, str] = {
        "Yearly": "1Y",
        "Quarterly": "3M",
        "Monthly": "1M",
        "Weekly": "1W",
        "Daily": "1D",
        "Hourly": "1H",
    }

    # Approximate seasonal lengths used for window sampling.
    history_size: Dict[str, int] = {
        "Yearly": 1,
        "Quarterly": 4,
        "Monthly": 12,
        "Weekly": 52,
        "Daily": 365,
        "Hourly": 24,
    }


@dataclass
class _M4Dataset:
    values: np.ndarray
    groups: np.ndarray
    ids: np.ndarray

    @classmethod
    def load(cls, training: bool, dataset_file: str) -> "_M4Dataset":
        """Load dataset from a NumPy `.npz` archive.

        Parameters
        ----------
        training : bool
            Unused flag kept for API compatibility with original implementation.
        dataset_file : str
            Path to the `.npz` file containing the arrays ``values``, ``groups`` and ``ids``.
        """
        data = np.load(dataset_file, allow_pickle=True)
        return cls(values=data["values"], groups=data["groups"], ids=data["ids"])


# Expose class name expected by the rest of the code
M4Dataset = _M4Dataset
