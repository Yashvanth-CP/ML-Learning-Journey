import numpy as np

np.random.seed(42)

sigmoid = lambda z: 1 / (1 + np.exp(-z))
relu    = lambda z: np.maximum(0, z)

# ── FORWARD PASS WITH DROPOUT ─────────────────────────────────────────────────
def forward_with_dropout(X, params, keep_prob, training=True):
    """
    keep_prob : fraction of neurons to KEEP  (e.g. 0.8 = drop 20%)
    training  : True → apply dropout | False → no dropout (inference)
    Returns   : AL (output), caches (for backprop)
    """
    caches = []
    A = X
    L = len(params) // 2

    for l in range(1, L):                              # hidden layers
        Z    = params[f'W{l}'] @ A + params[f'b{l}']
        A    = relu(Z)
        mask = None
        if training:
            mask = (np.random.rand(*A.shape) < keep_prob)  # 1=keep, 0=drop
            A    = (A * mask) / keep_prob                   # inverted dropout
        caches.append((A, params[f'W{l}'], params[f'b{l}'], Z, mask))

    Z  = params[f'W{L}'] @ A + params[f'b{L}']        # output layer (no dropout)
    AL = sigmoid(Z)
    caches.append((AL, params[f'W{L}'], params[f'b{L}'], Z, None))
    return AL, caches

# ── BACKWARD PASS WITH DROPOUT ────────────────────────────────────────────────
def backward_with_dropout(X, AL, Y, caches, params, keep_prob):
    grads = {}
    L     = len(caches)
    m     = Y.shape[1]
    dA    = -(Y / np.clip(AL,1e-15,1-1e-15) - (1-Y) / np.clip(1-AL,1e-15,1-1e-15))

    for l in reversed(range(1, L + 1)):
        _, W, b, Z, mask = caches[l - 1]
        A_prev = caches[l-2][0] if l > 1 else X

        dZ = dA * (Z > 0) if l < L else dA * AL * (1 - AL)   # ReLU or sigmoid
        grads[f'dW{l}'] = dZ @ A_prev.T / m
        grads[f'db{l}'] = np.mean(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ

        if mask is not None:                            # reverse dropout
            dA = (dA * mask) / keep_prob
    return grads

# ── INIT & UPDATE ─────────────────────────────────────────────────────────────
def init_he(dims):
    p = {}
    for l in range(1, len(dims)):
        p[f'W{l}'] = np.random.randn(dims[l], dims[l-1]) * np.sqrt(2/dims[l-1])
        p[f'b{l}'] = np.zeros((dims[l], 1))
    return p

def update(params, grads, lr):
    for l in range(1, len(params)//2 + 1):
        params[f'W{l}'] -= lr * grads[f'dW{l}']
        params[f'b{l}'] -= lr * grads[f'db{l}']
    return params

# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(X, Y, dims, lr=0.1, epochs=2000, keep_prob=0.8):
    params = init_he(dims)
    for i in range(epochs):
        AL, caches = forward_with_dropout(X, params, keep_prob, training=True)
        cost       = -np.mean(Y*np.log(np.clip(AL,1e-15,1)) + (1-Y)*np.log(np.clip(1-AL,1e-15,1)))
        grads      = backward_with_dropout(X, AL, Y, caches, params, keep_prob)
        params     = update(params, grads, lr)
        if i % 500 == 0:
            print(f"  Epoch {i:4d}  cost={cost:.4f}")
    return params

# ── DEMO ──────────────────────────────────────────────────────────────────────
print("Training WITH dropout (keep_prob=0.8):")
X = np.random.randn(5, 200)
Y = (X[0:1] * X[1:2] > 0).astype(float)         # simple non-linear pattern

params = train(X, Y, dims=[5, 8, 4, 1], lr=0.1, epochs=2000, keep_prob=0.8)

# Inference — training=False → NO dropout
AL_test, _ = forward_with_dropout(X, params, keep_prob=1.0, training=False)
preds      = (AL_test >= 0.5).astype(int)
print(f"\nTest accuracy : {(preds == Y).mean()*100:.1f}%")
print("\nKEY RULE: dropout ON during training, OFF during inference")

