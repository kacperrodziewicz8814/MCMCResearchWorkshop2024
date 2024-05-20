import numpy as np

def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """Calculating distance in R^3"""
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5