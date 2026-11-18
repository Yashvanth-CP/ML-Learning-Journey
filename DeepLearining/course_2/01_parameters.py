import numpy as np

np.random.seed(42)

print("=" * 55)
print("1. Parameters vs Hyperparameters")
print("=" * 55)
print("""
PARAMETERS (learned from data):
  W1, b1, W2, b2, ..., WL, bL

HYPERPARAMETERS (you choose):
  learning_rate  = 0.01
  n_layers       = 4
  n_hidden       = [64, 32, 16]
  activation     = 'relu'
  epochs         = 1000
  lambda_reg     = 0.1
  keep_prob      = 0.8  (dropout)
""")


# =============================================================================
# 2. ZERO INITIALISATION — WHY IT FAILS
# =============================================================================
# If W = 0 everywhere:
#   Every neuron computes EXACTLY the same Z, same A, same gradients.
#   All neurons update identically → they stay identical forever.
#   This is called the SYMMETRY PROBLEM.
#   The network behaves like it has only ONE neuron per layer.

print("=" * 55)
print("2. Zero Initialisation — Symmetry Problem")
print("=" * 55)

def init_zeros(layer_dims):
    params = {}
    for l in range(1, len(layer_dims)):
        params[f'W{l}'] = np.zeros((layer_dims[l], layer_dims[l-1]))
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return params

layer_dims = [3, 4, 2, 1]
params_zero = init_zeros(layer_dims)

print("W1 (zero init):")
print(params_zero['W1'])
print("All neurons identical → symmetry NOT broken → network can't learn\n")


def init_random(layer_dims, scale=0.01):
    params = {}
    for l in range(1, len(layer_dims)):
        params[f'W{l}'] = np.random.randn(layer_dims[l], layer_dims[l-1]) * scale
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return params

params_rand = init_random(layer_dims, scale=0.01)
print(f"W1 mean : {params_rand['W1'].mean():.4f}")
print(f"W1 std  : {params_rand['W1'].std():.4f}")
print(f"Scale=0.01 → small weights → good starting point\n")

# Show saturation with large weights
sigmoid = lambda z: 1 / (1 + np.exp(-z))
z_small = np.array([-0.1, 0.0, 0.1])
z_large = np.array([-10.0, 0.0, 10.0])
print(f"sigmoid({z_small}) = {sigmoid(z_small).round(4)}  ← gradient active")
print(f"sigmoid({z_large}) = {sigmoid(z_large).round(4)}  ← gradient ≈ 0, saturated!\n")


# =============================================================================
# 4. HE INITIALISATION — BEST FOR ReLU
# =============================================================================
# Problem with plain random: variance of Z grows with n_prev (more inputs).
# Fix: scale W by sqrt(2 / n_prev)   ← the "2" is because ReLU kills half
#
# He init:  W = np.random.randn(n_curr, n_prev) * sqrt(2 / n_prev)
#
# This keeps the variance of activations constant across layers.
# Invented by Kaiming He (2015) specifically for ReLU networks.

print("=" * 55)
print("4. He Initialisation (for ReLU)")
print("=" * 55)

def init_he(layer_dims):
    """He initialisation — recommended for ReLU hidden layers."""
    params = {}
    for l in range(1, len(layer_dims)):
        n_prev = layer_dims[l-1]
        params[f'W{l}'] = np.random.randn(layer_dims[l], n_prev) * np.sqrt(2 / n_prev)
        params[f'b{l}'] = np.zeros((layer_dims[l], 1))
    return params

params_he = init_he(layer_dims)
print("He init W1:")
print(f"  Shape : {params_he['W1'].shape}")
print(f"  Mean  : {params_he['W1'].mean():.4f}")
print(f"  Std   : {params_he['W1'].std():.4f}")
print(f"  Scale = sqrt(2 / {layer_dims[0]}) = {np.sqrt(2/layer_dims[0]):.4f}\n")


# =============================================================================
# 5. XAVIER INITIALISATION — BEST FOR tanh / sigmoid
# =============================================================================
# Xavier init:  W = np.random.randn(n_curr, n_prev) * sqrt(1 / n_prev)
#  or (Glorot): W = np.random.randn(n_curr, n_prev) * sqrt(2 / (n_prev + n_curr))
#
# Why different from He?
#   tanh is symmetric around 0 and doesn't kill half the activations.
#   The "2" in He is for ReLU's dead half — not needed for tanh.

print("=" * 55)
print("5. Xavier Initialisation (for tanh/sigmoid)")
print("=" * 55)

def init_xavier(layer_dims):
    """Xavier/Glorot initialisation — recommended for tanh hidden layers."""
    params = {}
    for l in range(1, len(layer_dims)):
        n_prev = layer_dims[l-1]
        n_curr = layer_dims[l]
        params[f'W{l}'] = np.random.randn(n_curr, n_prev) * np.sqrt(2 / (n_prev + n_curr))
        params[f'b{l}'] = np.zeros((n_curr, 1))
    return params

params_xav = init_xavier(layer_dims)
print("Xavier init W1:")
n_p, n_c = layer_dims[0], layer_dims[1]
print(f"  Scale = sqrt(2 / ({n_p}+{n_c})) = {np.sqrt(2/(n_p+n_c)):.4f}")
print(f"  Std   : {params_xav['W1'].std():.4f}\n")

print("Summary — which init to use:")
print("  ReLU activation  → He initialisation   (scale = sqrt(2/n_prev))")
print("  tanh activation  → Xavier              (scale = sqrt(2/(n_prev+n_curr)))")
print("  sigmoid output   → random * 0.01       (small, doesn't matter as much)\n")


# =============================================================================
# 6. COMPARE INITIALISATIONS — effect on first forward pass
# =============================================================================
print("=" * 55)
print("6. Compare: activation variance across layers")
print("=" * 55)

relu = lambda z: np.maximum(0, z)

X_test = np.random.randn(3, 1000)   # 1000 examples, 3 features
dims   = [3, 10, 10, 10, 1]

for name, init_fn in [("zeros",  lambda d: init_zeros(d)),
                      ("random*10", lambda d: init_random(d, scale=10)),
                      ("random*0.01", lambda d: init_random(d, scale=0.01)),
                      ("He",     lambda d: init_he(d))]:
    p = init_fn(dims)
    A = X_test
    for l in range(1, len(dims)-1):
        Z = p[f'W{l}'] @ A + p[f'b{l}']
        A = relu(Z)
    print(f"  {name:15s} → Layer 3 activation std: {A.std():.4f}  mean: {A.mean():.4f}")

print("\nHe init keeps std ~1 across layers — others explode or vanish\n")
