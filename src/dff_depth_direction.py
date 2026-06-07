from __future__ import annotations

import numpy as np


# Canonical project convention:
# focus-stack image 0 is the high focal plane, and later images scan downward.
# Keep the raw stack order unchanged; only convert the selected focus index to
# a display/training height through this module.
STACK_ORDER = "high_to_low"


def focus_index_to_relative_height(index: np.ndarray, layer_count: int) -> np.ndarray:
    """Convert best-focus layer index to canonical relative height [0, 1]."""
    denom = max(int(layer_count) - 1, 1)
    relative_layer = index.astype(np.float32) / float(denom)
    if STACK_ORDER == "high_to_low":
        return (1.0 - relative_layer).astype(np.float32)
    if STACK_ORDER == "low_to_high":
        return relative_layer.astype(np.float32)
    raise ValueError(f"Unsupported STACK_ORDER: {STACK_ORDER}")


def focus_index_to_height_um(index: np.ndarray, layer_count: int, height_range_um: float) -> np.ndarray:
    """Convert best-focus layer index to height in micrometers."""
    return focus_index_to_relative_height(index, layer_count) * float(height_range_um)


def focus_positions_norm(layer_count: int, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    """Return focus positions in the same order as the project focus stack."""
    if STACK_ORDER == "high_to_low":
        return np.linspace(high, low, int(layer_count), dtype=np.float32)
    if STACK_ORDER == "low_to_high":
        return np.linspace(low, high, int(layer_count), dtype=np.float32)
    raise ValueError(f"Unsupported STACK_ORDER: {STACK_ORDER}")


def focus_positions_um(height_range_um: float, layer_count: int) -> np.ndarray:
    """Return physical focus positions in micrometers for synthetic stacks."""
    return focus_positions_norm(layer_count, low=0.0, high=float(height_range_um))
