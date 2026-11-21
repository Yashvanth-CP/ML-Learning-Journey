import numpy as np

np.random.seed(42)
sigmoid = lambda z: 1/(1 + np.exp(-z))
relu = lambda z:np.maximum(0,z)

def init_he(dims):
    p = {}
    for l in range(1, len(dims)):
        p[f'W{l}'] = np.random.rand(dims[1], dims[l-1]) * np.sqrt(2/dims[l-1])
        p[f'b{1}'] = np.zeros((dims[l],1))

    return p

def forward(X, params):
    caches, A, L = [], X, len(params)//2
    for l in range(1, L):
        Z = params[f'W{l}'] @ A + params[f'b{l}']
        A = relu(Z)
        caches.append((A, params[f'W{l}'], params[f'b{l}'], Z))

    Z  = params[f'W{L}'] @ A + params[f'b{L}']
    AL = sigmoid(Z)
    caches.append((AL, params[f'W{L}'], params[f'b{L}'],Z))
    return AL, caches

def update(params, grads, lr):
    for l in range(1, len(params)//2 + 1):
        params[f'W{l}'] -= lr * grads[f'dW{l}']
        params[f'b{l}'] -= lr * grads[f'db{l}']
    return params

# L2 regularization 

def cost_L2(AL, Y, params, lam):
    """Binary cross-entropy + L2 penalty."""
    m   = Y.shape[1]
    AL  = np.clip(AL, 1e-15, 1-1e-15)
    bce = -np.mean(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    L   = len(params) // 2
    reg = (lam/(2*m)) * sum(np.sum(params[f'W{l}']**2) for l in range(1,L+1))
    return bce + reg

def backward_L2(X, AL, Y, caches, params, lam):
    """Backprop with L2 regularization."""
    grads, L = {}, len(caches)
    m        = Y.shape[1]
    dA       = -(Y/np.clip(AL,1e-15,1-1e-15) - (1-Y)/np.clip(1-AL,1e-15,1-1e-15))

    for l in reversed(range(1, L+1)):
        _, W, b, Z = caches[l-1]
        A_prev     = caches[l-2][0] if l > 1 else X
        dZ = dA * (Z > 0) if l < L else dA * AL * (1 - AL)
        # KEY LINE: add (λ/m)*W to the weight gradient
        grads[f'dW{l}'] = dZ @ A_prev.T / m + (lam/m) * W
        grads[f'db{l}'] = np.mean(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ
    return grads

# 2. L1 REGULARIZATION

def cost_L1(AL, Y, params, lam):
    """Binary cross-entropy + L1 penalty."""
    m   = Y.shape[1]
    AL  = np.clip(AL, 1e-15, 1-1e-15)
    bce = -np.mean(Y * np.log(AL) + (1-Y) * np.log(1-AL))
    L   = len(params) // 2
    reg = (lam/m) * sum(np.sum(np.abs(params[f'W{l}'])) for l in range(1,L+1))
    return bce + reg

def backward_L1(X, AL, Y, caches, params, lam):
    """Backprop with L1 regularization."""
    grads, L = {}, len(caches)
    m        = Y.shape[1]
    dA       = -(Y/np.clip(AL,1e-15,1-1e-15) - (1-Y)/np.clip(1-AL,1e-15,1-1e-15))

    for l in reversed(range(1, L+1)):
        _, W, b, Z = caches[l-1]
        A_prev     = caches[l-2][0] if l > 1 else X
        dZ = dA * (Z > 0) if l < L else dA * AL * (1 - AL)
        # KEY LINE: add (λ/m)*sign(W) — gradient of |W| is sign(W)
        grads[f'dW{l}'] = dZ @ A_prev.T / m + (lam/m) * np.sign(W)
        grads[f'db{l}'] = np.mean(dZ, axis=1, keepdims=True)
        dA = W.T @ dZ
    return grads

# 3. TRAIN FUNCTION (works for both L1 and L2)
def train(X, Y, dims, lr=0.1, epochs=2000, lam=0.1, reg='L2'):
    params = init_he(dims)
    cost_fn = cost_L2 if reg == 'L2' else cost_L1
    back_fn = backward_L2 if reg == 'L2' else backward_L1

    for i in range(epochs):
        AL, caches = forward(X, params)
        cost       = cost_fn(AL, Y, params, lam)
        grads      = back_fn(X, AL, Y, caches, params, lam)
        params     = update(params, grads, lr)
        if i % 500 == 0:
            print(f"  [{reg}] Epoch {i:4d}  cost={cost:.4f}")
    return params

def accuracy(X, Y, params):
    AL, _ = forward(X, params)
    return ((AL >= 0.5).astype(int) == Y).mean() * 100

print("=" * 50)
print("Dataset: 4 features, 300 examples")
print("=" * 50)

X_tr = np.random.randn(4, 300)
Y_tr = ((X_tr[0] + X_tr[1] - X_tr[2]) > 0).astype(float).reshape(1, 300)
X_te = np.random.randn(4, 100)
Y_te = ((X_te[0] + X_te[1] - X_te[2]) > 0).astype(float).reshape(1, 100)

dims = [4, 16, 8, 1]

print("\nNo Regularization (λ=0):")
p0 = train(X_tr, Y_tr, dims, lr=0.05, epochs=2000, lam=0.0, reg='L2')
print(f"  Train acc: {accuracy(X_tr, Y_tr, p0):.1f}%  Test acc: {accuracy(X_te, Y_te, p0):.1f}%")

print("\nL2 Regularization (λ=0.1):")
p2 = train(X_tr, Y_tr, dims, lr=0.05, epochs=2000, lam=0.1, reg='L2')
print(f"  Train acc: {accuracy(X_tr, Y_tr, p2):.1f}%  Test acc: {accuracy(X_te, Y_te, p2):.1f}%")
w2_mean = np.mean([np.abs(p2[f'W{l}']).mean() for l in range(1,4)])
print(f"  Mean |W|  : {w2_mean:.4f}  (smaller than no-reg)")

print("\nL1 Regularization (λ=0.1):")
p1 = train(X_tr, Y_tr, dims, lr=0.05, epochs=2000, lam=0.1, reg='L1')
print(f"  Train acc: {accuracy(X_tr, Y_tr, p1):.1f}%  Test acc: {accuracy(X_te, Y_te, p1):.1f}%")
zeros_pct = np.mean([np.mean(np.abs(p1[f'W{l}']) < 0.01) for l in range(1,4)]) * 100
print(f"  Weights near zero: {zeros_pct:.1f}%  (L1 sparsity effect)")

