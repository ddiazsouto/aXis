import numpy as np


def cosine_similarity(
        vector1: np.ndarray, vector2: np.ndarray) -> float:
    denominator = (
        np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
    )
    return np.dot(vector1, vector2) / denominator