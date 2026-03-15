import numpy as np
from numpy.typing import NDArray
from math import floor, sqrt
from typing import Any
import json

FREQ_EXP = 0.75
DISCARD_THRESHOLD = 10**-5
SAMPLING_SIZE = 1_000_000
CORP_LIMIT = 6_000_000
CORP_PATH = "text8"
EMBED_DIM = 100
WINDOW_SIZE = 5
NEGATIVES_TO_SAMPLE = 5
EPSILON = 10**-7
LEARNING_RATE = 0.008
PRINT_TIME = 10000
Matrix = NDArray[np.float64]
Vector = NDArray[np.float64]
Float = np.float64


def sigmoid(t: np.float64 | Vector) -> np.float64 | Vector:
    return 1 / (1 + np.exp(-np.clip(t, -500, 500))) # will work for both scalars and vectors since np.exp does both


def loss(vc: Vector, uo: Vector, uk: Matrix) -> Any:
    return -np.log(sigmoid(np.dot(vc, uo)) + 1e-10) - np.sum(np.log(sigmoid(np.dot(uk, -vc))) + 1e-10)


def load_corpus(path: str) -> tuple[list[int], dict[int, int], dict[str, int]]:
    corpus: list[int] = []
    with open(path, "r") as f:
        seen: dict[str, int] = {}
        freqs: dict[int, int] = {} 
        last_used_id = -1
        paragraphs = f.read().split("\n")
        for paragraph in paragraphs:
            words = paragraph.split() # split by any number of whitespaces
            for word in words[:CORP_LIMIT]:
                word = word.lower()
                if word in seen:
                    corpus.append(seen[word])
                else:
                    last_used_id += 1
                    corpus.append(last_used_id)
                    seen[word] = last_used_id

                freqs[seen[word]] = freqs.get(seen[word], 0) + 1

    return corpus, freqs, seen


def generate_pairs(corpus: list[int], freqs: dict[int, int], window_size: int) -> list[tuple[int, int]]:
    pairs = []
    for center_idx in range(len(corpus)):
        left = max(0, center_idx - window_size)
        right = min(len(corpus), center_idx + window_size)
        for context_idx in range(left, right):
            discard_p = 1 - sqrt(DISCARD_THRESHOLD / (freqs[corpus[center_idx]] / len(corpus)))
            if context_idx != center_idx and np.random.random() > discard_p:
                pairs.append((corpus[center_idx], corpus[context_idx]))        

    return pairs


def negative_sampling_table(freqs: dict[int, int]) -> Vector:
    shares = []
    freqs_sum = sum([freq**FREQ_EXP for freq in freqs.values()])
    for word, freq in freqs.items():
        share = (freq**FREQ_EXP / freqs_sum) * SAMPLING_SIZE
        shares.append((floor(share), word))
    
    table = np.concatenate([np.repeat(word, share) for share, word in shares])
    
    return table


def forward_pass(center_id: int, positive_id: int, negative_ids: list[int],
                 W_in: Matrix, W_out: Matrix) ->  \
                 tuple[Vector, Vector, Matrix, np.float64]:
    center_row = W_in[center_id] # v_c
    neg_grads = []
    
    # loop through negative ids and compute the gradient for each, also keep the sum for center grad
    negative_sum = 0
    neg_sigmoids = []
    for neg_id in negative_ids: #u_k
        neg_sigmoids.append(sigmoid(np.dot(-center_row, W_out[neg_id])))
        dif = (1 - neg_sigmoids[-1]) * center_row
        neg_grads.append(dif)

        negative_sum += (1 - neg_sigmoids[-1]) * W_out[neg_id]

    pos_sigmoid = sigmoid(np.dot(center_row, W_out[positive_id]))
    pos_grad: Vector = -(1 - pos_sigmoid) * center_row
    center_grad: Vector = -(1 - pos_sigmoid) * W_out[positive_id] - negative_sum

    total_loss: Float = -np.log(np.array(pos_sigmoid) + 1e-10) - np.sum(np.log(np.array(neg_sigmoids) + 1e-10))

    return np.clip(center_grad, -5, 5), np.clip(pos_grad, -5, 5), np.clip(np.array(neg_grads), -5, 5), total_loss


# https://www.youtube.com/watch?v=QrzApibhohY
def grad_check(W_in: Matrix, W_out: Matrix,
               center_grad: Vector, pos_grad: Vector, neg_grads: Matrix,
               center_id: int, positive_id: int, negative_ids: list[int]) -> None:

    aproxs = []
    center_row = W_in[center_id]
    neg_rows = [W_out[neg_id] for neg_id in negative_ids]
    pos_row = W_out[positive_id]
    all_rows = neg_rows.copy() # we want this to be shallow, no need for a deep copy here
    all_rows.append(pos_row)
    all_rows.append(center_row)

    for row in all_rows:
        for i in range(len(row)):
            row[i] += EPSILON
            loss1 = loss(center_row, pos_row, neg_rows)
            row[i] -= 2 * EPSILON
            loss2 = loss(center_row, pos_row, neg_rows)
            row[i] += EPSILON
            aproxs.append((loss1 - loss2) / (2 * EPSILON))

    all_grads = np.concatenate([np.ndarray.flatten(neg_grads), pos_grad, center_grad]) 

    aproxs = np.array(aproxs)
    check = np.linalg.norm(aproxs - all_grads) / (np.linalg.norm(all_grads) + np.linalg.norm(aproxs))

    if check >= 10**(-5):
        print("LOOKS A BIT WRONG")
    if check >= 10**(-3):
        print("ITS DEFINETLY WRONG")
        raise SystemError
    

def train() -> None:
    corp, freqs, seen = load_corpus(CORP_PATH)
    with open("word_dict.json", "w") as f:
        json.dump(seen, f)
    
    pairs = generate_pairs(corp,freqs, WINDOW_SIZE)
    sampling_table = negative_sampling_table(freqs)
    W_in = np.random.uniform(-0.5 / EMBED_DIM, 0.5/EMBED_DIM, (len(freqs), EMBED_DIM))
    W_out = np.zeros((len(freqs), EMBED_DIM), dtype=Float)
    prnt_tick = 0
    loss_sum = 0

    # call grad check on one arbitrary pair
    arbitrary_negs = np.random.choice(sampling_table, NEGATIVES_TO_SAMPLE)
    center_grad, pos_grad, neg_grads, total_loss = \
            forward_pass(0, 1, arbitrary_negs, W_in, W_out)
    grad_check(W_in, W_out, center_grad, pos_grad, neg_grads, 0, 1, arbitrary_negs) 

    for center_id, pos_id in pairs:
        negative_ids = np.random.choice(sampling_table, NEGATIVES_TO_SAMPLE)
        center_grad, pos_grad, neg_grads, total_loss = \
            forward_pass(center_id, pos_id, negative_ids, W_in, W_out)
        
        W_out[pos_id] -= pos_grad * LEARNING_RATE
        for idx, grad in enumerate(neg_grads):
            W_out[negative_ids[idx]] -= grad * LEARNING_RATE

        W_in[center_id] -= center_grad *LEARNING_RATE
        
        if prnt_tick == 0: 
            print(loss_sum / PRINT_TIME)
            loss_sum = 0

        loss_sum += total_loss
        prnt_tick = (prnt_tick + 1) % PRINT_TIME

    np.save("W_in.npy", W_in)
    np.save("W_out.npy", W_out)


if __name__ == "__main__":
    train()
