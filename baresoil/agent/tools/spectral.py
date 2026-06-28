"""Spectral index tools for BareSoil-Agent."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def ndvi(nir: np.ndarray, red: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return (nir - red) / (nir + red + eps)


def bsi(swir: np.ndarray, red: np.ndarray, blue: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Bare Soil Index (BSI) — higher values indicate more bare soil."""
    return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue) + eps)


def bsi_sentinel2(bands: Dict[str, np.ndarray]) -> np.ndarray:
    """
    BSI from Sentinel-2 bands dict with keys B02, B04, B08, B11 (or b02, b04, b08, b11).
    """
    def g(key: str) -> np.ndarray:
        for k in (key, key.lower(), key.upper()):
            if k in bands:
                return bands[k].astype(np.float32)
        raise KeyError(f"Band {key} not in {list(bands.keys())}")

    b02, b04, b08, b11 = g("B02"), g("B04"), g("B08"), g("B11")
    return ((b11 + b04) - (b08 + b02)) / ((b11 + b04) + (b08 + b02) + 1e-8)


def summarize_bare_soil_indices(
    ndvi_map: np.ndarray,
    bsi_map: np.ndarray,
    bare_ndvi_thresh: float = 0.2,
    bare_bsi_thresh: float = 0.1,
) -> Dict[str, Any]:
    valid = np.isfinite(ndvi_map) & np.isfinite(bsi_map)
    if not valid.any():
        return {"bare_fraction": 0.0, "mean_ndvi": 0.0, "mean_bsi": 0.0}
    bare_mask = (ndvi_map < bare_ndvi_thresh) & (bsi_map > bare_bsi_thresh) & valid
    return {
        "bare_fraction": float(bare_mask.sum() / valid.sum()),
        "mean_ndvi": float(ndvi_map[valid].mean()),
        "mean_bsi": float(bsi_map[valid].mean()),
        "bare_pixel_count": int(bare_mask.sum()),
        "total_pixels": int(valid.sum()),
    }
