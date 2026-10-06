"""
ERROR ANALYSIS
"""

import numpy as np
import matplotlib.pyplot as plt

TP, FP, FN, TN = 40, 10, 20, 930
P_ex, R_ex = TP / (TP + FP), TP / (TP + FN)
print("NUMERIC EXAMPLE:  TP=40  FP=10  FN=20  TN=930  (1000 samples, 60 real positives)")
print(f"  accuracy  = (TP+TN)/total = {(TP+TN)/1000:.3f}")
print(f"  precision = 40/(40+10)    = {P_ex:.3f}")
print(f"  recall    = 40/(40+20)    = {R_ex:.3f}")
print(f"  F1        = 2*{P_ex:.3f}*{R_ex:.3f}/({P_ex:.3f}+{R_ex:.3f}) = {2*P_ex*R_ex/(P_ex+R_ex):.3f}")
print("  -> 97% accuracy but we still missed a third of the positives.")

 # METRICS FROM SCRATCH

def confusion(y_true, y_pred, K):
    cm = np.zeros((K, K), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)                 # rows = actual, cols = predicted
    return cm
 
def prf(cm):
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    denom = np.maximum(precision + recall, 1e-12)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / denom, 0.0)
    return precision, recall, f1
 
def curves(scores, y):
    """ROC + Precision-Recall curves for a binary problem (y in {0,1})."""
    order = np.argsort(-scores, kind="mergesort")
    s, yy = scores[order], y[order]
    tps, fps = np.cumsum(yy), np.cumsum(1 - yy)
    idx = np.r_[np.where(np.diff(s))[0], len(yy) - 1]          # last index of every distinct score
    tps, fps = tps[idx], fps[idx]
    Pn, Nn = yy.sum(), len(yy) - yy.sum()
    tpr, fpr = np.r_[0.0, tps / Pn], np.r_[0.0, fps / Nn]
    precision, recall = tps / (tps + fps), tps / Pn
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))   # trapezoid
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))             # average precision
    return fpr, tpr, auc, precision, recall, ap
 
# ---------------------------------------------------------------------------
# NETWORK (same building blocks as earlier topics)
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
    A = X
    caches = [X]
    Zs = []
    for l in range(1, L + 1):
        Z = A @ params[f"W{l}"] + params[f"b{l}"]
        A = softmax(Z) if l == L else np.maximum(0, Z)
        Zs.append(Z); caches.append(A)
    return A, (caches, Zs)
 
def backward(Y, params, cache, L):
    caches, Zs = cache
    m = Y.shape[0]
    grads = {}
    dZ = caches[L] - Y
    for l in range(L, 0, -1):
        grads[f"dW{l}"] = caches[l - 1].T @ dZ / m
        grads[f"db{l}"] = dZ.sum(axis=0, keepdims=True) / m
        if l > 1:
            dZ = (dZ @ params[f"W{l}"].T) * (Zs[l - 2] > 0)
    return grads
 
def train(sizes, X, Y, lr=0.01, epochs=100, batch=64, seed=0):
    L = len(sizes) - 1
    params = init_params(sizes, seed)
    m_ = {k: np.zeros_like(v) for k, v in params.items()}
    v_ = {k: np.zeros_like(v) for k, v in params.items()}
    rng = np.random.default_rng(seed)
    t = 0
    for _ in range(epochs):
        order = rng.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            _, cache = forward(X[idx], params, L)
            g = backward(Y[idx], params, cache, L)
            t += 1
            for k in params:
                m_[k] = 0.9 * m_[k] + 0.1 * g["d" + k]
                v_[k] = 0.999 * v_[k] + 0.001 * g["d" + k] ** 2
                params[k] -= lr * (m_[k] / (1 - 0.9 ** t)) / (np.sqrt(v_[k] / (1 - 0.999 ** t)) + 1e-8)
    return params
 
# ---------------------------------------------------------------------------
# PART 1: IMBALANCED BINARY PROBLEM
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 1: imbalanced binary problem (about 5% positives, overlapping classes)")
print("=" * 74)
rng = np.random.default_rng(0)
n = 6000
y_all = (rng.random(n) < 0.05).astype(int)
X_all = rng.standard_normal((n, 2)) + y_all[:, None] * np.array([2.0, 2.0])      # positives shifted
perm = rng.permutation(n)
X_all, y_all = X_all[perm], y_all[perm]
tr, va, te = slice(0, 3000), slice(3000, 4500), slice(4500, 6000)
Xtr, ytr, Xva, yva, Xte, yte = X_all[tr], y_all[tr], X_all[va], y_all[va], X_all[te], y_all[te]
print(f"Positives -> train: {ytr.sum()}/{len(ytr)}   val: {yva.sum()}/{len(yva)}   test: {yte.sum()}/{len(yte)}")
 
dumb = np.zeros_like(yte)
cm_d = confusion(yte, dumb, 2); pd_, rd_, fd_ = prf(cm_d)
print(f"\nDumb baseline 'always negative': accuracy = {np.mean(dumb == yte)*100:.1f}%   "
      f"precision = {pd_[1]:.2f}  recall = {rd_[1]:.2f}  F1 = {fd_[1]:.2f}")
 
sizes_b = [2, 16, 16, 2]
params_b = train(sizes_b, Xtr, np.eye(2)[ytr], epochs=100)
score_va = forward(Xva, params_b, 3)[0][:, 1]
score_te = forward(Xte, params_b, 3)[0][:, 1]
 
def report(name, y, score, thr):
    pred = (score >= thr).astype(int)
    cm = confusion(y, pred, 2)
    p, r, f = prf(cm)
    print(f"{name:<34} acc {np.mean(pred == y)*100:5.1f}% | precision {p[1]:.3f} | recall {r[1]:.3f} | F1 {f[1]:.3f}")
    return cm, p[1], r[1], f[1]
 
print("\nTEST set results:")
cm_05, *_ = report("Network, threshold = 0.5 (default)", yte, score_te, 0.5)
 
# choose threshold on VALIDATION data (never on test!)
thrs = np.linspace(0.02, 0.98, 97)
val_f1 = []
val_p, val_r = [], []
for t_ in thrs:
    cm_ = confusion(yva, (score_va >= t_).astype(int), 2)
    p_, r_, f_ = prf(cm_)
    val_p.append(p_[1]); val_r.append(r_[1]); val_f1.append(f_[1])
best_thr = float(thrs[int(np.argmax(val_f1))])
cm_best, *_ = report(f"Network, tuned threshold = {best_thr:.2f}", yte, score_te, best_thr)
print(f"(threshold {best_thr:.2f} was chosen to maximize F1 on the VALIDATION set)")
 
fpr, tpr, auc, prec_c, rec_c, ap = curves(score_te, yte)
base_rate = yte.mean()
print(f"\nROC-AUC (test) = {auc:.3f}   |   Average precision (test) = {ap:.3f}   (random guessing AP = {base_rate:.3f})")
print("Rule: with heavy imbalance, trust the PR curve / AP more than ROC-AUC - ROC can look rosy.")
 
# ---------------------------------------------------------------------------
# VERIFY MY METRICS AGAINST SKLEARN
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("CHECK: are my from-scratch metrics correct? (compared with sklearn)")
print("=" * 74)
from sklearn.metrics import (confusion_matrix as sk_cm, precision_recall_fscore_support as sk_prf,
                             roc_auc_score, average_precision_score)
pred_best = (score_te >= best_thr).astype(int)
ok_cm = np.array_equal(sk_cm(yte, pred_best), cm_best)
sp, sr, sf, _ = sk_prf(yte, pred_best, zero_division=0)
mp, mr, mf = prf(cm_best)
ok_prf = np.allclose([sp, sr, sf], [mp, mr, mf])
ok_auc = abs(roc_auc_score(yte, score_te) - auc) < 1e-9
ok_ap = abs(average_precision_score(yte, score_te) - ap) < 1e-9
print(f"confusion matrix equal: {ok_cm} | precision/recall/F1 equal: {ok_prf} | "
      f"ROC-AUC equal: {ok_auc} | avg precision equal: {ok_ap}")
 
# ---------------------------------------------------------------------------
# PART 2: MULTI-CLASS FAILURE ANALYSIS
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 2: multi-class error analysis (noisy 3-class spiral)")
print("=" * 74)
K = 3
def make_spiral(n_per_class, noise, seed):
    r_ = np.random.default_rng(seed)
    X = np.zeros((n_per_class * K, 2)); y = np.zeros(n_per_class * K, dtype=int)
    for k in range(K):
        idx = range(n_per_class * k, n_per_class * (k + 1))
        rad = np.linspace(0.0, 1.0, n_per_class)
        th = np.linspace(k * 4, (k + 1) * 4, n_per_class) + r_.standard_normal(n_per_class) * noise
        X[idx] = np.c_[rad * np.sin(th), rad * np.cos(th)]
        y[idx] = k
    pm = r_.permutation(len(X))
    return X[pm], y[pm]
 
Xs, ys = make_spiral(300, noise=0.5, seed=3)
Xs_tr, ys_tr, Xs_te, ys_te = Xs[:300], ys[:300], Xs[300:], ys[300:]
params_s = train([2, 64, 64, K], Xs_tr, np.eye(K)[ys_tr], epochs=150, batch=32)
probs = forward(Xs_te, params_s, 3)[0]
pred_s = probs.argmax(axis=1)
cm_s = confusion(ys_te, pred_s, K)
ps, rs, fs = prf(cm_s)
print(f"Test accuracy: {np.mean(pred_s == ys_te)*100:.1f}%   ({np.sum(pred_s != ys_te)} mistakes out of {len(ys_te)})\n")
print("Confusion matrix (rows = actual, columns = predicted):")
print(cm_s)
print(f"\n{'class':>5} | {'precision':>9} | {'recall':>7} | {'F1':>6}")
for k in range(K):
    print(f"{k:>5} | {ps[k]:9.3f} | {rs[k]:7.3f} | {fs[k]:6.3f}")
print(f"{'macro':>5} | {ps.mean():9.3f} | {rs.mean():7.3f} | {fs.mean():6.3f}   (simple average over classes)")
 
wrong = np.where(pred_s != ys_te)[0]
conf = probs.max(axis=1)
worst = wrong[np.argsort(-conf[wrong])][:5]
print("\nTop 5 MOST CONFIDENT mistakes (look at these first - they are the most informative):")
print(f"{'#':>4} | {'x':>6} {'y':>6} | {'true':>4} {'pred':>4} | confidence")
for i in worst:
    print(f"{i:>4} | {Xs_te[i,0]:6.2f} {Xs_te[i,1]:6.2f} | {ys_te[i]:>4} {pred_s[i]:>4} | {conf[i]*100:6.1f}%")
 
radius = np.linalg.norm(Xs_te, axis=1)
edges = [0.0, 0.25, 0.5, 0.75, 1.01]
print("\nERROR RATE BY REGION (distance from the center):")
print(f"{'radius bin':>12} | {'samples':>7} | {'mistakes':>8} | error rate")
rate_bins = []
for a, b in zip(edges[:-1], edges[1:]):
    msk = (radius >= a) & (radius < b)
    nerr = int(np.sum(pred_s[msk] != ys_te[msk]))
    rate = nerr / max(msk.sum(), 1)
    rate_bins.append(rate)
    print(f"{a:5.2f}-{min(b,1.0):4.2f}  | {msk.sum():7d} | {nerr:8d} | {rate*100:6.1f}%")
print("Error analysis step: find WHERE errors cluster, then ask 'is it noise, missing data, or a model limit?'")
 
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def draw_cm(ax, cm, title, labels):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
 
fig, axes = plt.subplots(2, 4, figsize=(21, 10))
draw_cm(axes[0, 0], cm_05, "Binary, threshold 0.5 (test)", ["neg", "pos"])
draw_cm(axes[0, 1], cm_best, f"Binary, tuned threshold {best_thr:.2f} (test)", ["neg", "pos"])
draw_cm(axes[0, 2], cm_s, "3-class spiral (test)", ["0", "1", "2"])
 
ax = axes[0, 3]
ok = pred_s == ys_te
ax.scatter(Xs_te[ok, 0], Xs_te[ok, 1], c=ys_te[ok], cmap="brg", s=14, alpha=0.6)
ax.scatter(Xs_te[~ok, 0], Xs_te[~ok, 1], facecolors="none", edgecolors="k", s=90, linewidths=1.5, label="mistake")
ax.set_title("Where are the mistakes? (circled)", fontsize=10, fontweight="bold"); ax.legend(); ax.set_aspect("equal")
 
ax = axes[1, 0]
ax.plot(fpr, tpr, linewidth=2, label=f"ROC (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], "k:", label="random")
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (recall)")
ax.set_title("ROC curve", fontsize=10, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[1, 1]
ax.plot(rec_c, prec_c, linewidth=2, label=f"PR (AP={ap:.3f})")
ax.axhline(base_rate, color="k", linestyle=":", label=f"random ({base_rate:.3f})")
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("Precision-Recall curve", fontsize=10, fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[1, 2]
ax.plot(thrs, val_p, label="precision", linewidth=2)
ax.plot(thrs, val_r, label="recall", linewidth=2)
ax.plot(thrs, val_f1, label="F1", linewidth=2, linestyle="--")
ax.axvline(best_thr, color="red", alpha=0.5, linestyle=":", label=f"best F1 @ {best_thr:.2f}")
ax.set_xlabel("decision threshold"); ax.set_title("Threshold trade-off (validation set)", fontsize=10, fontweight="bold")
ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[1, 3]
labels_r = [f"{a:.2f}-{min(b,1.0):.2f}" for a, b in zip(edges[:-1], edges[1:])]
ax.bar(labels_r, np.array(rate_bins) * 100, color="tab:red")
ax.set_xlabel("distance from center"); ax.set_ylabel("error rate (%)")
ax.set_title("Error rate by region", fontsize=10, fontweight="bold"); ax.grid(alpha=0.3, axis="y")
 
plt.tight_layout()
plt.savefig("error_analysis.png", dpi=200, bbox_inches="tight")
print("\nSaved: error_analysis.png")
plt.show()
 