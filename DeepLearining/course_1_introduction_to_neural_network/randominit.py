import numpy as np
 
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
m = X.shape[0]
 
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
def init_weights(method, n_in, n_out, rng):
    """Return a weight matrix of shape (n_in, n_out) using the chosen method."""
    if method == "zeros":
        return np.zeros((n_in, n_out))
    if method == "small_random":
        return rng.standard_normal((n_in, n_out)) * 0.5
    if method == "too_large":
        return rng.standard_normal((n_in, n_out)) * 5.0
    if method == "xavier":
        return rng.standard_normal((n_in, n_out)) * np.sqrt(1 / n_in)
    if method == "he":
        return rng.standard_normal((n_in, n_out)) * np.sqrt(2 / n_in)
    raise ValueError(f"Unknown method: {method}")
 
 
def train(method, epochs=5000, lr=1.0, seed=42):
    rng = np.random.default_rng(seed)
 
    # Weights use the chosen method. Biases always start at zero.
    W1 = init_weights(method, 2, 4, rng)
    b1 = np.zeros((1, 4))
    W2 = init_weights(method, 4, 1, rng)
    b2 = np.zeros((1, 1))
 
    for _ in range(epochs):
        # forward
        a1 = sigmoid(X @ W1 + b1)
        a2 = sigmoid(a1 @ W2 + b2)
 
        # backward
        dz2 = (2 * (a2 - y) / m) * a2 * (1 - a2)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
 
        # update
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
 
    a1 = sigmoid(X @ W1 + b1)
    a2 = sigmoid(a1 @ W2 + b2)
    loss = np.mean((a2 - y) ** 2)
    return loss, a2, W1
 
 
if __name__ == "__main__":
    methods = ["zeros", "small_random", "too_large", "xavier", "he"]
 
    print(f"{'method':<14}{'final loss':<14}predictions for [00, 01, 10, 11]")
    print("-" * 70)
    for method in methods:
        loss, preds, W1 = train(method)
        print(f"{method:<14}{loss:<14.6f}{np.round(preds.ravel(), 3)}")
 
    # Show the symmetry problem clearly
    _, _, W1_zero = train("zeros")
    print("\nW1 after training with ZERO init (each column = one hidden neuron):")
    print(np.round(W1_zero, 4))
    print("All 4 columns are identical -> the 4 neurons act like 1 neuron.")