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


print("=" * 55)
print("7. L2 Regularisation")
print("=" * 55)

def l2_cost_term(params, lam, m):
    """Extra cost from L2 regularisation."""
    L   = len(params) // 2
    reg = sum(np.sum(params[f'W{l}']**2) for l in range(1, L+1))
    return (lam / (2*m)) * reg

def l2_grad_term(W, lam, m):
    """Extra gradient from L2 regularisation."""
    return (lam / m) * W

lam = 0.1
m   = 100
p   = init_he([3, 5, 1])
print(f"L2 cost term (λ={lam}): {l2_cost_term(p, lam, m):.4f}")
print(f"L2 grad extra for W1 : adds (λ/m)*W1 = {lam/m:.4f} * W1 to each gradient\n")


def dropout_forward(A, keep_prob):
    """
    Apply dropout to activation A.
    Returns A_drop, mask (mask saved for backprop).
    """
    mask  = (np.random.rand(*A.shape) < keep_prob)   # True = keep
    A_drop = (A * mask) / keep_prob                   # scale up survivors
    return A_drop, mask

def dropout_backward(dA, mask, keep_prob):
    """Backprop through dropout layer."""
    return (dA * mask) / keep_prob

A_example  = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
keep_prob  = 0.8
A_dropped, mask = dropout_forward(A_example, keep_prob)

print(f"Original A    : {A_example}")
print(f"Dropout mask  : {mask.astype(int)}  (1=keep, 0=drop)")
print(f"After dropout : {A_dropped.round(3)}")
print(f"Scaling by 1/{keep_prob} = {1/keep_prob:.2f} keeps expected value same\n")
print("IMPORTANT: dropout only during TRAINING, NOT during inference!\n")


# =============================================================================
# 9. GRADIENT CHECKING
# =============================================================================
# Used to verify your backprop is correct.
# Numerical gradient (finite differences):
#   dJ/dθ ≈ (J(θ+ε) - J(θ-ε)) / (2ε)
#
# Compare with analytical gradient from backprop.
# If difference < 1e-5 → backprop is correct.

print("=" * 55)
print("9. Gradient Checking")
print("=" * 55)

def dict_to_vec(params):
    """Flatten all parameters into one long vector."""
    vec = []
    for key in sorted(params.keys()):
        vec.append(params[key].flatten())
    return np.concatenate(vec)

# Simple 1-layer network to demonstrate grad check
def simple_cost(W, X, Y):
    """Simple linear + sigmoid cost for 1 output."""
    Z  = W @ X
    AL = 1 / (1 + np.exp(-Z))
    AL = np.clip(AL, 1e-15, 1-1e-15)
    return -np.mean(Y * np.log(AL) + (1-Y) * np.log(1-AL))

W_gc = np.array([[0.5, -0.3, 0.1]])
X_gc = np.random.randn(3, 5)
Y_gc = np.array([[1, 0, 1, 0, 1]], dtype=float)

# Analytical gradient
AL = 1 / (1 + np.exp(-(W_gc @ X_gc)))
dW_analytical = ((AL - Y_gc) @ X_gc.T) / 5

# Numerical gradient
eps = 1e-5
dW_numerical = np.zeros_like(W_gc)
for i in range(W_gc.shape[1]):
    W_p = W_gc.copy(); W_p[0,i] += eps
    W_m = W_gc.copy(); W_m[0,i] -= eps
    dW_numerical[0,i] = (simple_cost(W_p, X_gc, Y_gc) -
                          simple_cost(W_m, X_gc, Y_gc)) / (2*eps)

diff = np.linalg.norm(dW_analytical - dW_numerical) / (
       np.linalg.norm(dW_analytical) + np.linalg.norm(dW_numerical))

print(f"Analytical gradient : {dW_analytical.round(6)}")
print(f"Numerical  gradient : {dW_numerical.round(6)}")
print(f"Relative difference : {diff:.2e}  (< 1e-5 = PASS ✓)\n")
