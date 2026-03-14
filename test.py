import numpy as np
from numpy.typing import NDArray
import json

Vector = NDArray[np.float64]

def cos_similarity(a: Vector, b: Vector) -> np.float64:
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def k_most_similar(word: str, k: int) -> list[str]:
    W_in = np.load("W_in.npy")
    with open("word_dict.json", "r") as f:
        word_dict = json.load(f)

    idx_dict = {val : key for key, val in word_dict.items()}
    needle_row = W_in[word_dict[word]]
    top_k = []
    worst_accepted = -1

    for row, idx in enumerate(W_in):
        if row is not needle_row:
            score = cos_similarity(row, needle_row)
            if score > worst_accepted or len(top_k) < k:
                top_k.append((idx_dict[idx], score))
                worst_accepted = min([score for _, score in top_k])

    return top_k

print(k_most_similar("king", 8))