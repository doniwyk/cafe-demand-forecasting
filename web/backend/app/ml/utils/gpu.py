from __future__ import annotations

import os
import logging

logger = logging.getLogger(__name__)

_gpu_available: bool | None = None


def is_gpu_available() -> bool:
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    try:
        from xgboost import XGBRegressor
        import numpy as np
        m = XGBRegressor(n_estimators=1, device="cuda", tree_method="gpu_hist")
        m.fit(np.array([[1, 2]]), np.array([1]))
        _gpu_available = True
    except Exception:
        _gpu_available = False
    return _gpu_available


def get_device() -> str:
    return "cuda" if is_gpu_available() else "cpu"


def get_xgboost_params() -> dict:
    if is_gpu_available():
        logger.info("GPU detected — using CUDA acceleration for XGBoost")
        return {"device": "cuda", "tree_method": "gpu_hist", "n_jobs": -1}
    logger.info("No GPU detected — using CPU for XGBoost")
    return {"tree_method": "hist", "n_jobs": -1}
