import numpy as np

# ── Deep L-layer Neural Network from scratch ──────────────────────────────────
# Architecture : [n0] → [n1] → ... → [nL]   ReLU hidden, Sigmoid output
# Supports any number of layers via layer_dims list
# Example: [2, 4, 3, 1] = 2 inputs, 2 hidden layers, 1 output

np.random.seed(1)

# ── 1. INITIALISE PARAMETERS ──────────────────────────────────────────────────
def init_params(layer_dims):
    params = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2 / layer_dims[l-1])
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return params

# ── 2. ACTIVATIONS ────────────────────────────────────────────────────────────
def relu(Z):    return np.maximum(0, Z), Z
def sigmoid(Z): S = 1 / (1 + np.exp(-Z)); return S, Z

def relu_back(dA, Z):    return dA * (Z > 0)
def sigmoid_back(dA, Z):
    S = 1 / (1 + np.exp(-Z))
    return dA * S * (1 - S)

# ── 3. FORWARD PROPAGATION ────────────────────────────────────────────────────
def forward(X, params):
    caches, A = [], X
    L = len(params) // 2
    for l in range(1, L):                          # hidden layers: ReLU
        Z = params[f'W{l}'] @ A + params[f'b{l}']
        A, cache = relu(Z)
        caches.append((A, params[f'W{l}'], params[f'b{l}'], cache))
    Z = params[f'W{L}'] @ A + params[f'b{L}']     # output layer: Sigmoid
    AL, cache = sigmoid(Z)
    caches.append((AL, params[f'W{L}'], params[f'b{L}'], cache))
    return AL, caches

# ── 4. COST ───────────────────────────────────────────────────────────────────
def compute_cost(AL, Y):
    m = Y.shape[1]
    AL = np.clip(AL, 1e-15, 1 - 1e-15)
    return -np.mean(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))

# ── 5. BACKWARD PROPAGATION ───────────────────────────────────────────────────
def backward(AL, Y, caches):
    grads, L = {}, len(caches)
    m = Y.shape[1]
    dA = -(Y / np.clip(AL, 1e-15, 1-1e-15) - (1-Y) / np.clip(1-AL, 1e-15, 1-1e-15))

    for l in reversed(range(L)):
        A_prev, W, b, Z = caches[l]
        A_prev_layer = caches[l-1][0] if l > 0 else None

        dZ = sigmoid_back(dA, Z) if l == L-1 else relu_back(dA, Z)
        A_in = caches[l-1][0] if l > 0 else caches[0][0]  # handled below

        # A_in is input to this layer
        if l == 0:
            # first layer: A_in is original X, stored before loop — pass separately
            pass  # handled after loop
        grads[f'dW{l+1}'] = dZ @ (caches[l-1][0] if l > 0 else None).T / m if l > 0 else None
        grads[f'db{l+1}'] = np.mean(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ
    return grads

# cleaner backward (self-contained):
def backward_v2(X, AL, Y, caches, params):
    grads, L = {}, len(caches)
    m = Y.shape[1]
    dA = -(Y / np.clip(AL,1e-15,1-1e-15) - (1-Y) / np.clip(1-AL,1e-15,1-1e-15))

    for l in reversed(range(1, L + 1)):
        _, W, b, Z = caches[l - 1]
        A_prev = caches[l-2][0] if l > 1 else X
        dZ = sigmoid_back(dA, Z) if l == L else relu_back(dA, Z)
        grads[f'dW{l}'] = dZ @ A_prev.T / m
        grads[f'db{l}'] = np.mean(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ
    return grads

# ── 6. UPDATE PARAMETERS ──────────────────────────────────────────────────────
def update_params(params, grads, lr):
    L = len(params) // 2
    for l in range(1, L + 1):
        params[f'W{l}'] -= lr * grads[f'dW{l}']
        params[f'b{l}'] -= lr * grads[f'db{l}']
    return params

# ── 7. TRAIN ──────────────────────────────────────────────────────────────────
def train(X, Y, layer_dims, lr=0.01, epochs=1000, print_every=200):
    params = init_params(layer_dims)
    for i in range(epochs):
        AL, caches = forward(X, params)
        cost       = compute_cost(AL, Y)
        grads      = backward_v2(X, AL, Y, caches, params)
        params     = update_params(params, grads, lr)
        if i % print_every == 0:
            print(f"Epoch {i:4d}  cost={cost:.4f}")
    return params

def predict(X, params):
    AL, _ = forward(X, params)
    return (AL >= 0.5).astype(int)

# ── 8. TEST ───────────────────────────────────────────────────────────────────
# XOR problem — not linearly separable, needs hidden layers
X = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)   # (2, 4)
Y = np.array([[0,1,1,0]],           dtype=float)   # (1, 4)

print("Training on XOR (needs ≥1 hidden layer):")
params = train(X, Y, layer_dims=[2, 4, 1], lr=0.5, epochs=5000, print_every=1000)
preds  = predict(X, params)
print(f"\nX      : {X.T.tolist()}")
print(f"Y true : {Y.tolist()}")
print(f"Y pred : {preds.tolist()}")
print(f"Accuracy: {(preds == Y).mean()*100:.0f}%")