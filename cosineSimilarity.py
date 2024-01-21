import numpy as np

def cosine_similarity(X, Y=None):
    # If Y is not provided, compute pairwise similarity for X
    if Y is None:
        Y = X

    # Compute dot product
    dot_product = np.dot(X, Y.T)

    # Compute L2 norms
    norm_X = np.linalg.norm(X, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)

    # Compute cosine similarity
    similarity = dot_product / (np.outer(norm_X, norm_Y) + 1e-8)  # Adding a small epsilon to avoid division by zero

    return similarity