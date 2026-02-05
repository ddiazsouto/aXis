import numpy as np


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    epsilon = 1e-12
    denominator = (
        np.linalg.norm(vector1) * np.linalg.norm(vector2) + epsilon
    )
    return np.dot(vector1, vector2) / denominator