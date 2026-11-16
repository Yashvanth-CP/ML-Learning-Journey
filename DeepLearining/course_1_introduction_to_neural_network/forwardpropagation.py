import numpy as np

# =============================================================================
# FORWARD PROPAGATION IN DEEP LEARNING — from scratch
# =============================================================================
# What happens in forward prop:
#   Input X → Layer1 → Layer2 → ... → LayerL → Output AL
#
# Each layer does TWO things:
#   Step 1: LINEAR    Z = W @ A_prev + b
#   Step 2: ACTIVATE  A = g(Z)   where g = ReLU (hidden) or Sigmoid (output)
#
# We CACHE (A_prev, W, b, Z) at every layer — needed later for backprop.
# =============================================================================

np.random.seed(42)

# ── ACTIVATIONS ───────────────────────────────────────────────────────────────

def sigmoid(Z):
    """
    sigmoid(Z) = 1 / (1 + e^-Z)
    Output range: (0, 1)  — used at output layer for binary classification
    """
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    """
    ReLU(Z) = max(0, Z)
    Output range: [0, ∞)  — used at hidden layers
    """
    return np.maximum(0, Z)

def tanh(Z):
    """
    tanh(Z) = (e^Z - e^-Z) / (e^Z + e^-Z)
    Output range: (-1, 1)  — alternative to ReLU for hidden layers
    """
    return np.tanh(Z)

def softmax(Z):
    """
    softmax(Z)_k = e^Zk / sum(e^Z)
    Output range: (0,1), sums to 1 — used for multi-class output
    Z shape: (n_classes, m)
    """
    Z_shift = Z - Z.max(axis=0, keepdims=True)   # numerical stability
    e = np.exp(Z_shift)
    return e / e.sum(axis=0, keepdims=True)

# ── INITIALISE PARAMETERS ─────────────────────────────────────────────────────

def init_params(layer_dims):
    """
    layer_dims : list e.g. [3, 5, 4, 1]
                 index 0 = input features
                 index 1..L = neurons in each layer

    He initialisation: W *= sqrt(2 / n_prev)
    Why? Prevents vanishing/exploding gradients with ReLU.
    """
    params = {}
    L = len(layer_dims) - 1          # number of layers (excluding input)
    for l in range(1, L + 1):
        n_curr = layer_dims[l]
        n_prev = layer_dims[l - 1]
        params[f'W{l}'] = np.random.randn(n_curr, n_prev) * np.sqrt(2 / n_prev)
        params[f'b{l}'] = np.zeros((n_curr, 1))
        print(f"  Layer {l}: W{l}.shape={params[f'W{l}'].shape}  "
              f"b{l}.shape={params[f'b{l}'].shape}")
    return params

# ── ONE LAYER FORWARD ─────────────────────────────────────────────────────────

def linear_forward(A_prev, W, b):
    """
    Z = W @ A_prev + b

    A_prev : (n_prev, m)  — activations from previous layer
    W      : (n_curr, n_prev)
    b      : (n_curr, 1)
    Z      : (n_curr, m)   — linear output
    """
    Z = W @ A_prev + b          # matrix multiply + bias
    return Z

def activation_forward(A_prev, W, b, activation):
    """
    Linear → Activation for one layer.
    Returns A (output), cache (A_prev, W, b, Z)
    """
    Z = linear_forward(A_prev, W, b)
    if   activation == 'relu':    A = relu(Z)
    elif activation == 'sigmoid': A = sigmoid(Z)
    elif activation == 'tanh':    A = tanh(Z)
    elif activation == 'softmax': A = softmax(Z)
    cache = (A_prev, W, b, Z)   # save everything — needed in backprop
    return A, cache

# ── FULL L-LAYER FORWARD PASS ─────────────────────────────────────────────────

def L_layer_forward(X, params, hidden_act='relu', output_act='sigmoid'):
    """
    Full forward propagation through all layers.

    X       : (n0, m)  — input  (n0 features, m examples)
    params  : dict with W1,b1, W2,b2, ... WL,bL
    Returns : AL (final output), caches (list, one per layer)
    """
    caches = []
    A      = X
    L      = len(params) // 2    # number of layers

    # Hidden layers (1 to L-1): use hidden_act (usually ReLU)
    for l in range(1, L):
        A, cache = activation_forward(A, params[f'W{l}'], params[f'b{l}'], hidden_act)
        caches.append(cache)
        print(f"  Layer {l} ({hidden_act}): A{l}.shape = {A.shape}")

    # Output layer L: use output_act (sigmoid or softmax)
    AL, cache = activation_forward(A, params[f'W{L}'], params[f'b{L}'], output_act)
    caches.append(cache)
    print(f"  Layer {L} ({output_act}): AL.shape = {AL.shape}  ← final output")

    return AL, caches

# ── RUN AND VERIFY ────────────────────────────────────────────────────────────

print("=" * 55)
print("Example: layer_dims = [4, 5, 3, 1]")
print("  4 input features, hidden [5,3], 1 output")
print("=" * 55)

layer_dims = [4, 5, 3, 1]
m          = 6                           # 6 training examples
X          = np.random.randn(4, m)       # (4, 6)

print("\nInitialising parameters:")
params = init_params(layer_dims)

print("\nForward pass:")
AL, caches = L_layer_forward(X, params)

print(f"\nFinal output AL:\n{AL.round(4)}")
print(f"AL range: [{AL.min():.4f}, {AL.max():.4f}]  (should be 0–1 for sigmoid)")
print(f"Number of caches stored: {len(caches)}  (one per layer)")

# ── SHAPE CHECK ───────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Shape walkthrough (trace the data through the network):")
print("=" * 55)
A_check = X
for l in range(1, len(layer_dims)):
    W = params[f'W{l}']
    b = params[f'b{l}']
    Z = W @ A_check + b
    A_check = relu(Z) if l < len(layer_dims)-1 else sigmoid(Z)
    print(f"  Layer {l}: ({W.shape[1]},{m}) --W{l}{W.shape}--> Z{l}{Z.shape} --> A{l}{A_check.shape}")

# ── WHAT IS STORED IN CACHE? ──────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Cache contents (layer 1):")
print("=" * 55)
A_prev_c, W_c, b_c, Z_c = caches[0]
print(f"  A_prev (input to layer 1) : {A_prev_c.shape}")
print(f"  W1                        : {W_c.shape}")
print(f"  b1                        : {b_c.shape}")
print(f"  Z1 (pre-activation)       : {Z_c.shape}")
print("  → These are reused in backprop to compute gradients")