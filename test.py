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

    for idx, row in enumerate(W_in):
        if row is not needle_row:
            score = cos_similarity(row, needle_row)
            if len(top_k) < k:
                top_k.append((idx_dict[idx], score))

            elif score > worst_accepted:
                pos = 0
                while(score < top_k[pos][1]):
                    pos += 1
                top_k.insert(pos, (idx_dict[idx], score))
                top_k.pop()
                worst_accepted = top_k[-1][1]

    return top_k

print(k_most_similar("king", 8))
