"""
REGULARIZATION
"""


import copy
import numpy as np
import matplotlib.pyplot as plt


w_a = np.array([3.0, 0.0, 0.0])
w_b = np.array([1.0, 1.0, 1.0])
print("Two weight vectors that give a similar output (sum = 3):")
print(f"  w_a = {w_a}   L1 = {np.abs(w_a).sum():.0f}   L2 (sum w^2) = {(w_a**2).sum():.0f}")
print(f"  w_b = {w_b}   L1 = {np.abs(w_b).sum():.0f}   L2 (sum w^2) = {(w_b**2).sum():.0f}")
print("  -> L2 punishes w_a 3x more: it prefers weights SPREAD OUT and small.")
print("  -> L1 sees them as equal in size, but its constant pull drives")
print("     unimportant weights all the way to ~0 (sparse model).")
 
print("\nGradient effect (the part that goes into backprop):")
print("  L2: dW += lambda * W          (pull proportional to size -> shrink)")
print("  L1: dW += lambda * sign(W)    (constant pull -> reach zero)")
 
A = np.array([2.0, 4.0, 6.0, 8.0])
keep = 0.5
mask = np.array([1, 0, 1, 0]) / keep
print(f"\nInverted dropout example (keep_prob = {keep}):")
print(f"  activations A        = {A}")
print(f"  mask (0/1)/keep      = {mask}")
print(f"  A * mask             = {A * mask}")
print("  Survivors are scaled UP by 1/keep so the expected value stays the same.")
print("  At TEST time: no dropout, no scaling - use the full network.")
 
# ---------------------------------------------------------------------------
# NETWORK (same building blocks as Topic 8, plus dropout + L1/L2)
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
 
def forward(X, params, L, keep=1.0, train=False, rng=None):
    cache = {"A0": X}
    A = X
    for l in range(1, L + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        if l == L:
            A = softmax(Z)
        else:
            A = np.maximum(0, Z)
            if train and keep < 1.0:                       # dropout ONLY while training
                D = (rng.random(A.shape) < keep) / keep    # inverted dropout mask
                A = A * D
                cache[f"D{l}"] = D
        cache[f"Z{l}"], cache[f"A{l}"] = Z, A
    return A, cache
 
def data_loss(A_out, Y):
    return -np.sum(Y * np.log(A_out + 1e-12)) / Y.shape[0]
 
def total_loss(params, X, Y, L, l2=0.0, l1=0.0):
    A_out, _ = forward(X, params, L)
    Ws = [params[f"W{l}"] for l in range(1, L + 1)]
    return (data_loss(A_out, Y)
            + 0.5 * l2 * sum(np.sum(W ** 2) for W in Ws)
            + l1 * sum(np.sum(np.abs(W)) for W in Ws))
 
def backward(Y, params, cache, L, l2=0.0, l1=0.0):
    m = Y.shape[0]
    grads = {}
    dZ = cache[f"A{L}"] - Y
    for l in range(L, 0, -1):
        W = params[f"W{l}"]
        grads[f"dW{l}"] = cache[f"A{l-1}"].T @ dZ / m + l2 * W + l1 * np.sign(W)   # penalty gradient
        grads[f"db{l}"] = dZ.sum(axis=0, keepdims=True) / m                        # biases NOT regularized
        if l > 1:
            dA = dZ @ W.T
            if f"D{l-1}" in cache:
                dA = dA * cache[f"D{l-1}"]              # same mask as forward
            dZ = dA * (cache[f"Z{l-1}"] > 0)
    return grads
 
def init_adam(params):
    return ({k: np.zeros_like(v) for k, v in params.items()},
            {k: np.zeros_like(v) for k, v in params.items()})
 
def adam_update(params, grads, m_, v_, t, lr=0.01, b1=0.9, b2=0.999, eps=1e-8):
    for k in params:
        g = grads["d" + k]
        m_[k] = b1 * m_[k] + (1 - b1) * g
        v_[k] = b2 * v_[k] + (1 - b2) * g ** 2
        params[k] -= lr * (m_[k] / (1 - b1 ** t)) / (np.sqrt(v_[k] / (1 - b2 ** t)) + eps)
 
# ---------------------------------------------------------------------------
# GRADIENT CHECK (does my penalty gradient match calculus?)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("GRADIENT CHECK for L2 + L1 terms (dropout off)")
print("=" * 72)
 
def grad_check(params, X, Y, L, l2, l1, n_checks=5, h=1e-5):
    _, cache = forward(X, params, L)
    grads = backward(Y, params, cache, L, l2, l1)
    rng = np.random.default_rng(1)
    worst = 0.0
    for key in params:
        p = params[key]
        for _ in range(n_checks):
            i = tuple(rng.integers(0, s) for s in p.shape)
            old = p[i]
            p[i] = old + h; lp = total_loss(params, X, Y, L, l2, l1)
            p[i] = old - h; lm = total_loss(params, X, Y, L, l2, l1)
            p[i] = old
            num = (lp - lm) / (2 * h)
            ana = grads["d" + key][i]
            worst = max(worst, abs(num - ana) / max(1e-8, abs(num) + abs(ana)))
    return worst
 
# ---------------------------------------------------------------------------
# DATA: noisy 3-class spiral, VERY few training samples (easy to memorize)
# ---------------------------------------------------------------------------
K = 3
def make_pool(n_per_class=400, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((n_per_class * K, 2)); y = np.zeros(n_per_class * K, dtype=int)
    for k in range(K):
        idx = range(n_per_class * k, n_per_class * (k + 1))
        r = np.linspace(0.0, 1.0, n_per_class)
        t = np.linspace(k * 4, (k + 1) * 4, n_per_class) + rng.standard_normal(n_per_class) * noise
        X[idx] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[idx] = k
    return X, y
 
X_pool, y_pool = make_pool()
Y_pool = np.eye(K)[y_pool]
 
def split(seed):
    perm = np.random.default_rng(seed).permutation(len(X_pool))
    tr, va, te = perm[:75], perm[75:300], perm[300:]
    return dict(Xtr=X_pool[tr], Ytr=Y_pool[tr], ytr=y_pool[tr],
                Xva=X_pool[va], Yva=Y_pool[va], yva=y_pool[va],
                Xte=X_pool[te], Yte=Y_pool[te], yte=y_pool[te])
 
d0 = split(0)
print(f"\nSplit -> train: {len(d0['Xtr'])}  validation: {len(d0['Xva'])}  test: {len(d0['Xte'])}")
print("(75 train samples vs a network with ~17,000 weights = easy to memorize)")
 
sizes = [2, 128, 128, K]
L = len(sizes) - 1
err = grad_check(init_params(sizes), d0["Xtr"][:20], d0["Ytr"][:20], L, l2=0.1, l1=0.01)
print(f"\nWorst relative gradient error: {err:.2e}  ->  "
      f"{'PASS' if err < 1e-5 else 'FAIL - bug in backward()'}")
 
# ---------------------------------------------------------------------------
# TRAINING FUNCTION
# ---------------------------------------------------------------------------
def acc(A_out, y):
    return float(np.mean(np.argmax(A_out, axis=1) == y))
 
def train(d, lr=0.01, epochs=300, batch=25, l2=0.0, l1=0.0, keep=1.0, patience=None, seed=0):
    params = init_params(sizes, seed)
    m_, v_ = init_adam(params)
    rng = np.random.default_rng(seed)
    n = len(d["Xtr"])
    hist = {"tr_loss": [], "va_loss": []}
    best_loss, best_params, best_epoch, wait, t = np.inf, None, 0, 0, 0
    for epoch in range(1, epochs + 1):
        order = rng.permutation(n)
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            _, cache = forward(d["Xtr"][idx], params, L, keep, True, rng)
            grads = backward(d["Ytr"][idx], params, cache, L, l2, l1)
            t += 1
            adam_update(params, grads, m_, v_, t, lr)
        tr_l = data_loss(forward(d["Xtr"], params, L)[0], d["Ytr"])   # monitor DATA loss only
        va_l = data_loss(forward(d["Xva"], params, L)[0], d["Yva"])
        hist["tr_loss"].append(tr_l); hist["va_loss"].append(va_l)
        if patience:
            if va_l < best_loss:
                best_loss, best_params, best_epoch, wait = va_l, copy.deepcopy(params), epoch, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if patience and best_params is not None:
        params = best_params                     # go back to the best epoch
    used = best_epoch if patience else epochs
    return params, hist, used
 
def evaluate(params, d):
    out = {}
    for s, name in [("tr", "tr"), ("va", "va"), ("te", "te")]:
        A_out = forward(d["X" + s], params, L)[0]
        out[name + "_acc"] = acc(A_out, d["y" + s])
        out[name + "_loss"] = data_loss(A_out, d["Y" + s])
    return out
 
SEEDS = [0, 1, 2, 3, 4]
def run(name, **cfg):
    """Train with 5 different seeds (different data split + init) and average."""
    runs = []
    for s in SEEDS:
        d = split(s)
        params, hist, used = train(d, seed=s, **cfg)
        r = evaluate(params, d); r.update(params=params, hist=hist, used=used)
        runs.append(r)
    agg = {k: (np.mean([r[k] for r in runs]), np.std([r[k] for r in runs]))
           for k in ["tr_acc", "va_acc", "te_acc", "va_loss", "used"]}
    return dict(name=name, runs=runs, agg=agg, cfg=cfg)
 
# ---------------------------------------------------------------------------
# EXPERIMENT 1: baseline (no regularization)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPERIMENT 1: NO regularization (300 epochs) - watch it overfit")
print("=" * 72)
base = run("Baseline (none)")
h = base["runs"][0]["hist"]
print(f"Seed 0 validation loss: lowest = {min(h['va_loss']):.3f} at epoch {int(np.argmin(h['va_loss']))+1}, "
      f"final = {h['va_loss'][-1]:.3f}")
print(f"Seed 0 train loss final = {h['tr_loss'][-1]:.4f}   (train keeps falling, val goes back UP = overfitting)")
 
# ---------------------------------------------------------------------------
# EXPERIMENT 2: choose lambda for L2 / L1 using VALIDATION loss (not test!)
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPERIMENT 2: lambda sweep (mean of 5 seeds)")
print("=" * 72)
print("Rule: pick hyperparameters using VALIDATION data. Test data is touched only to report.\n")
 
l2_grid = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
l2_runs = {}
print(f"{'L2 lambda':>10} | {'train acc':>9} | {'val loss':>8} | {'test acc':>8}")
for lam in l2_grid:
    r = run(f"L2 {lam}", l2=lam)
    l2_runs[lam] = r
    print(f"{lam:>10g} | {r['agg']['tr_acc'][0]*100:8.1f}% | {r['agg']['va_loss'][0]:8.3f} | {r['agg']['te_acc'][0]*100:7.1f}%")
best_l2 = min(l2_grid, key=lambda k: l2_runs[k]["agg"]["va_loss"][0])
print(f"-> best L2 lambda by validation loss: {best_l2:g}")
 
l1_grid = [1e-4, 1e-3, 1e-2]
l1_runs = {}
print(f"\n{'L1 lambda':>10} | {'train acc':>9} | {'val loss':>8} | {'test acc':>8}")
for lam in l1_grid:
    r = run(f"L1 {lam}", l1=lam)
    l1_runs[lam] = r
    print(f"{lam:>10g} | {r['agg']['tr_acc'][0]*100:8.1f}% | {r['agg']['va_loss'][0]:8.3f} | {r['agg']['te_acc'][0]*100:7.1f}%")
best_l1 = min(l1_grid, key=lambda k: l1_runs[k]["agg"]["va_loss"][0])
print(f"-> best L1 lambda by validation loss: {best_l1:g}")
 
# ---------------------------------------------------------------------------
# EXPERIMENT 3: dropout, early stopping, combination
# ---------------------------------------------------------------------------
print("\n" + "=" * 72)
print("EXPERIMENT 3: Dropout, Early stopping, and a combination")
print("=" * 72)
drop = {kp: run(f"Dropout keep={kp}", keep=kp) for kp in [0.8, 0.5]}
best_kp = min(drop, key=lambda k: drop[k]["agg"]["va_loss"][0])
early = run("Early stopping (patience 20)", patience=20)
combo = run(f"L2 {best_l2:g} + Dropout {best_kp}", l2=best_l2, keep=best_kp)
 
# ---------------------------------------------------------------------------
# RESULTS TABLE
# ---------------------------------------------------------------------------
final = [base, l2_runs[best_l2], l1_runs[best_l1], drop[best_kp], early, combo]
final[1]["name"] = f"L2 (lambda={best_l2:g})"
final[2]["name"] = f"L1 (lambda={best_l1:g})"
final[3]["name"] = f"Dropout (keep={best_kp})"
 
print("\n" + "=" * 72)
print("FINAL COMPARISON (mean +- std over 5 seeds, 900 test samples each)")
print("=" * 72)
print(f"{'Method':<28} | {'train acc':>12} | {'test acc':>12} | {'val loss':>8} | {'epochs':>6}")
print("-" * 80)
for r in final:
    a = r["agg"]
    print(f"{r['name']:<28} | {a['tr_acc'][0]*100:5.1f}+-{a['tr_acc'][1]*100:4.1f}% | "
          f"{a['te_acc'][0]*100:5.1f}+-{a['te_acc'][1]*100:4.1f}% | {a['va_loss'][0]:8.3f} | {a['used'][0]:6.0f}")
print("\nHONEST NOTE: test-accuracy differences of ~1% are INSIDE the +- std. On this small\n"
      "problem regularization mainly fixes the rising VALIDATION LOSS (plot 1), not accuracy.\n"
      "(Early stopping picks its epoch using val data, so its val loss is biased low.)")
print("\nTrain-vs-test GAP is the overfitting signal. A good regularizer shrinks the gap")
print("and/or raises test accuracy. The spiral has real noise, so 100% test is impossible.")
 
# ---------------------------------------------------------------------------
# L1 SPARSITY CHECK
# ---------------------------------------------------------------------------
def frac_small(params, thr=0.01):
    w = np.concatenate([params[f"W{l}"].ravel() for l in range(1, L + 1)])
    return float(np.mean(np.abs(w) < thr)), float(np.mean(np.abs(w)))
 
print("\n" + "=" * 72)
print("DO L1 AND L2 BEHAVE AS THEORY SAYS? (seed 0 models)")
print("=" * 72)
print(f"{'Model':<24} | {'mean |w|':>9} | {'weights with |w| < 0.01':>24}")
for r in [base, l2_runs[best_l2], l1_runs[best_l1]]:
    fs, mw = frac_small(r["runs"][0]["params"])
    print(f"{r['name']:<24} | {mw:9.4f} | {fs*100:23.1f}%")
print("(Note: with Adam, L1 weights jitter around 0 instead of sitting exactly on 0.)")
 
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
 
ax = axes[0, 0]
hb = base["runs"][0]["hist"]
ax.plot(hb["tr_loss"], label="train loss", linewidth=2)
ax.plot(hb["va_loss"], label="validation loss", linewidth=2, linestyle="--")
ax.axvline(int(np.argmin(hb["va_loss"])), color="red", alpha=0.5, linestyle=":", label="best val epoch")
ax.set_title("Baseline: train keeps falling, val goes back up", fontweight="bold")
ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy"); ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[0, 1]
ax.semilogx(l2_grid, [l2_runs[k]["agg"]["tr_acc"][0] for k in l2_grid], "o-", label="train acc", linewidth=2)
ax.semilogx(l2_grid, [l2_runs[k]["agg"]["te_acc"][0] for k in l2_grid], "s--", label="test acc", linewidth=2)
ax.axhline(base["agg"]["te_acc"][0], color="gray", linestyle=":", label="baseline test acc")
ax.set_title("L2 lambda: too small = overfit, too big = underfit", fontweight="bold")
ax.set_xlabel("lambda (log scale)"); ax.set_ylabel("accuracy"); ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[0, 2]
names = ["Baseline", "L2", "L1", "Dropout", "Early\nstopping", "L2 +\nDropout"]
x = np.arange(len(final))
ax.bar(x - 0.2, [r["agg"]["tr_acc"][0] for r in final], 0.4, label="train")
ax.bar(x + 0.2, [r["agg"]["te_acc"][0] for r in final], 0.4, label="test",
       yerr=[r["agg"]["te_acc"][1] for r in final], capsize=3)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0.5, 1.1); ax.set_title("Train vs test accuracy (mean of 5 seeds)", fontweight="bold")
ax.legend(loc="upper center", ncol=2); ax.grid(alpha=0.3, axis="y")
 
gx, gy = np.meshgrid(np.linspace(-1.2, 1.2, 250), np.linspace(-1.2, 1.2, 250))
grid = np.c_[gx.ravel(), gy.ravel()]
for ax, r, title in [(axes[1, 0], base, "Baseline (no regularization)"),
                     (axes[1, 1], l2_runs[best_l2], f"L2 lambda={best_l2:g}"),
                     (axes[1, 2], combo, combo["name"])]:
    p = r["runs"][0]["params"]
    pred = np.argmax(forward(grid, p, L)[0], axis=1).reshape(gx.shape)
    ax.contourf(gx, gy, pred, alpha=0.3, cmap="brg")
    ax.scatter(d0["Xtr"][:, 0], d0["Xtr"][:, 1], c=d0["ytr"], cmap="brg", s=25, edgecolors="k", linewidths=0.5)
    te = r["runs"][0]["te_acc"]
    ax.set_title(f"{title}\nseed-0 test acc = {te*100:.1f}%", fontsize=10, fontweight="bold")
 
plt.tight_layout()
plt.savefig("regularization_results.png", dpi=200, bbox_inches="tight")
print("\nSaved: regularization_results.png")
plt.show()
 
