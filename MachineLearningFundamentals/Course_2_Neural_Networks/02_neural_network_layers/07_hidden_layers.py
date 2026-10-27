import numpy as np
import matplotlib.pyplot as plt

"""
HIDDEN LAYERS
"""
print("=" * 72)
print("PART 1: XOR solved BY HAND with 2 hidden ReLU neurons")
print("=" * 72)
 
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])
W1 = np.array([[1.0, 1.0], [1.0, 1.0]])
b1 = np.array([0.0, -1.0])
W2 = np.array([1.0, -2.0])
 
Z1 = X_xor @ W1 + b1
H = np.maximum(0, Z1)
out = H @ W2
print("Hidden neuron 1: h1 = ReLU(x1 + x2)")
print("Hidden neuron 2: h2 = ReLU(x1 + x2 - 1)")
print("Output         : out = h1 - 2*h2\n")
print(f"{'x1':>3} {'x2':>3} | {'h1':>4} {'h2':>4} | {'out':>4} | target")
for x, h, o, t in zip(X_xor, H, out, y_xor):
    print(f"{x[0]:3.0f} {x[1]:3.0f} | {h[0]:4.0f} {h[1]:4.0f} | {o:4.0f} | {t}")
print(f"\nMatches XOR for all 4 points: {np.allclose(out, y_xor)}")
print("No single straight line can separate XOR - but 2 hidden neurons can.")
 
# ---------------------------------------------------------------------------
# PARAMETER COUNTING
# ---------------------------------------------------------------------------
def count_params(sizes):
    return sum(sizes[i] * sizes[i + 1] + sizes[i + 1] for i in range(len(sizes) - 1))
 
print("\n" + "=" * 72)
print("PART 2: COUNTING PARAMETERS")
print("=" * 72)
print("Layer with n_in inputs and n_out neurons has  n_in * n_out  weights + n_out biases.\n")
mn = [784, 128, 64, 10]
total = 0
for i in range(len(mn) - 1):
    p = mn[i] * mn[i + 1] + mn[i + 1]
    total += p
    print(f"  Layer {i+1}: {mn[i]:>3} -> {mn[i+1]:>3}   {mn[i]}*{mn[i+1]} + {mn[i+1]} = {p:,}")
print(f"  MNIST network {mn}: total = {total:,} parameters   (check: {count_params(mn):,})")
 
# ---------------------------------------------------------------------------
# NETWORK CODE (generic depth, ReLU hidden, softmax output)
# ---------------------------------------------------------------------------
def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    p = {}
    for l in range(1, len(sizes)):
        p[f"W{l}"] = rng.standard_normal((sizes[l - 1], sizes[l])) * np.sqrt(2.0 / sizes[l - 1])
        p[f"b{l}"] = np.zeros((1, sizes[l]))
    return p
 
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)
 
def forward(X, params, L):
    cache = {"A0": X}
    A = X
    for l in range(1, L + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        A = softmax(Z) if l == L else np.maximum(0, Z)
        cache[f"Z{l}"], cache[f"A{l}"] = Z, A
    return A, cache
 
def backward(Y, params, cache, L):
    m = Y.shape[0]
    grads = {}
    dZ = cache[f"A{L}"] - Y
    for l in range(L, 0, -1):
        grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ / m
        grads[f"db{l}"] = dZ.sum(axis=0, keepdims=True) / m
        if l > 1:
            dZ = (dZ @ params[f"W{l}"].T) * (cache[f"Z{l-1}"] > 0)
    return grads
 
def adam_update(params, grads, m_, v_, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        g = grads["d" + k]
        m_[k] = b1 * m_[k] + (1 - b1) * g
        v_[k] = b2 * v_[k] + (1 - b2) * g ** 2
        params[k] -= lr * (m_[k] / (1 - b1 ** t)) / (np.sqrt(v_[k] / (1 - b2 ** t)) + eps)
 
def train(sizes, X, Y, lr=0.01, epochs=200, batch=32, seed=0):
    L = len(sizes) - 1
    params = init_params(sizes, seed)
    m_ = {k: np.zeros_like(v) for k, v in params.items()}
    v_ = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(seed)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(X))                 # shuffle every epoch
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            _, cache = forward(X[idx], params, L)
            grads = backward(Y[idx], params, cache, L)
            t += 1
            adam_update(params, grads, m_, v_, t, lr)
    return params
 
def accuracy(params, sizes, X, y):
    return float(np.mean(np.argmax(forward(X, params, len(sizes) - 1)[0], axis=1) == y))
 
# ---------------------------------------------------------------------------
# DATA: 3-class spiral (needs a curvy boundary)
# ---------------------------------------------------------------------------
K = 3
def make_spiral(n_per_class=200, noise=0.2, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * K, 2)); y = np.zeros(n_per_class * K, dtype=int)
    for k in range(K):
        idx = range(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.standard_normal(n_per_class) * noise
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = k
    return X, y
 
X_pool, y_pool = make_spiral()
Y_pool = np.eye(K)[y_pool]
 
def split(seed):
    perm = np.random.default_rng(seed).permutation(len(X_pool))   # SHUFFLE before splitting
    tr, te = perm[:300], perm[300:]
    return X_pool[tr], Y_pool[tr], y_pool[tr], X_pool[te], y_pool[te]
 
SEEDS = [0, 1, 2, 3, 4]
def run(sizes):
    tr_acc, te_acc = [], []
    for s in SEEDS:
        Xtr, Ytr, ytr, Xte, yte = split(s)
        p = train(sizes, Xtr, Ytr, seed=s)
        tr_acc.append(accuracy(p, sizes, Xtr, ytr))
        te_acc.append(accuracy(p, sizes, Xte, yte))
    return np.mean(tr_acc), np.std(tr_acc), np.mean(te_acc), np.std(te_acc)
 
# ---------------------------------------------------------------------------
# EXPERIMENT 1: WIDTH (how many neurons in ONE hidden layer?)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPERIMENT 1: WIDTH - one hidden layer, vary the number of neurons")
print("=" * 72)
print("Spiral data, 300 train / 300 test, 200 epochs, mean of 5 seeds.\n")
 
width_cfgs = [("no hidden layer", [2, K])] + [(f"{h} neuron{'s' if h > 1 else ''}", [2, h, K]) for h in [1, 2, 4, 8, 16, 64]]
width_res = []
print(f"{'Model':<16} | {'sizes':<12} | {'params':>6} | {'train acc':>10} | {'test acc':>12}")
print("-" * 72)
for name, sz in width_cfgs:
    trm, trs, tem, tes = run(sz)
    width_res.append((name, sz, tem, tes, trm))
    print(f"{name:<16} | {str(sz):<12} | {count_params(sz):>6} | {trm*100:9.1f}% | {tem*100:6.1f}+-{tes*100:4.1f}%")
 
# ---------------------------------------------------------------------------
# EXPERIMENT 2: DEPTH vs WIDTH at (almost) the same parameter budget
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPERIMENT 2: DEPTH vs WIDTH with ~390 parameters each")
print("=" * 72)
depth_cfgs = [("1 hidden (wide)", [2, 64, K]),
              ("2 hidden", [2, 16, 16, K]),
              ("3 hidden", [2, 12, 12, 12, K]),
              ("4 hidden", [2, 10, 10, 10, 10, K])]
depth_res = []
print(f"{'Model':<16} | {'sizes':<24} | {'params':>6} | {'train acc':>10} | {'test acc':>12}")
print("-" * 82)
for name, sz in depth_cfgs:
    trm, trs, tem, tes = run(sz)
    depth_res.append((name, sz, tem, tes, trm))
    print(f"{name:<16} | {str(sz):<24} | {count_params(sz):>6} | {trm*100:9.1f}% | {tem*100:6.1f}+-{tes*100:4.1f}%")
 
print("\nHONEST NOTE: all four differ by less than ~1%, which is INSIDE the +- noise.")
print("On this simple 2-D spiral, depth gives no measurable benefit over width.")
print("Depth really pays off on hierarchical data (images, text), not on tiny 2-D toys.")
 
# ---------------------------------------------------------------------------
# DEMO 3: WHAT DOES EACH HIDDEN NEURON DO? (circle vs ring data)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("DEMO 3: what each hidden neuron learns (center-vs-ring data, [2, 4, 2])")
print("=" * 72)
rng = np.random.default_rng(42)
th0 = rng.uniform(0, 2 * np.pi, 100); r0 = rng.uniform(0, 1.0, 100)
th1 = rng.uniform(0, 2 * np.pi, 100); r1 = rng.uniform(1.5, 2.5, 100)
Xc = np.vstack([np.c_[r0 * np.cos(th0), r0 * np.sin(th0)], np.c_[r1 * np.cos(th1), r1 * np.sin(th1)]])
yc = np.r_[np.zeros(100, dtype=int), np.ones(100, dtype=int)]
perm = rng.permutation(len(Xc))                      # shuffle BEFORE split (fixes the all-class-1 test set)
Xc, yc = Xc[perm], yc[perm]
Xc_tr, yc_tr, Xc_te, yc_te = Xc[:160], yc[:160], Xc[160:], yc[160:]
print(f"Test set class counts: class0={np.sum(yc_te==0)}, class1={np.sum(yc_te==1)}  (both present now)")
 
csizes = [2, 4, 2]
cp = train(csizes, Xc_tr, np.eye(2)[yc_tr], lr=0.02, epochs=300, batch=16, seed=3)
print(f"Train acc: {accuracy(cp, csizes, Xc_tr, yc_tr)*100:.1f}%   Test acc: {accuracy(cp, csizes, Xc_te, yc_te)*100:.1f}%  (only 40 test points)")
alive = (forward(Xc, cp, 2)[1]["A1"] > 0).any(axis=0)
print(f"Hidden neurons that are active on at least one sample: {alive.sum()} of 4")
print("Each neuron = one ReLU 'half-plane'. The output layer combines them into a closed shape.")
 
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(20, 8.5))
gs = fig.add_gridspec(2, 5, hspace=0.45, wspace=0.3)
 
ax = fig.add_subplot(gs[0, 0:2])
labels = ["linear\n(none)"] + ["1", "2", "4", "8", "16", "64"]
tems = [r[2] for r in width_res]; tess = [r[3] for r in width_res]
ax.bar(range(len(width_res)), tems, yerr=tess, capsize=3, color="tab:orange")
ax.set_xticks(range(len(width_res))); ax.set_xticklabels(labels)
ax.set_xlabel("neurons in the hidden layer"); ax.set_ylabel("test accuracy")
ax.set_ylim(0, 1.05); ax.set_title("Width: too few neurons = underfit", fontweight="bold"); ax.grid(alpha=0.3, axis="y")
 
ax = fig.add_subplot(gs[0, 2:5])
dn = [f"{r[0]}\n{count_params(r[1])} params" for r in depth_res]
ax.bar(range(len(depth_res)), [r[2] for r in depth_res], yerr=[r[3] for r in depth_res], capsize=3, color="tab:green")
ax.set_xticks(range(len(depth_res))); ax.set_xticklabels(dn)
ax.set_ylim(0.5, 1.05); ax.set_ylabel("test accuracy")
ax.set_title("Depth vs width at ~same parameter count", fontweight="bold"); ax.grid(alpha=0.3, axis="y")
 
gx, gy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid = np.c_[gx.ravel(), gy.ravel()]
_, gc = forward(grid, cp, 2)
for j in range(4):
    ax = fig.add_subplot(gs[1, j])
    c = ax.contourf(gx, gy, gc["A1"][:, j].reshape(gx.shape), levels=20, cmap="viridis")
    ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="coolwarm", s=6, edgecolors="none")
    ax.set_title(f"Hidden neuron {j+1}\n(brighter = more active)", fontsize=10, fontweight="bold")
    ax.set_aspect("equal")
ax = fig.add_subplot(gs[1, 4])
pred = np.argmax(gc["A2"], axis=1).reshape(gx.shape)
ax.contourf(gx, gy, pred, alpha=0.35, cmap="coolwarm", levels=[-0.5, 0.5, 1.5])
ax.scatter(Xc[:, 0], Xc[:, 1], c=yc, cmap="coolwarm", s=10, edgecolors="k", linewidths=0.2)
ax.set_title("Output layer combines them\n(final decision regions)", fontsize=10, fontweight="bold")
ax.set_aspect("equal")
 
plt.savefig("hidden_layers_results.png", dpi=200, bbox_inches="tight")
print("\nSaved: hidden_layers_results.png")
plt.show()
 
