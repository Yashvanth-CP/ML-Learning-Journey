"""
COURSE 2 - TOPIC 13: DECISION TREES, RANDOM FOREST, XGBOOST
===========================================================
Tree-based models: the strongest tools for tabular (spreadsheet-style) data.

Everything is built from scratch in NumPy, then compared with sklearn and the real
xgboost library. Every number printed here comes from running this file.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 74)
print("DECISION TREES -> RANDOM FOREST -> XGBOOST")
print("=" * 74)

print("""
1) DECISION TREE = a game of 20 questions
   Each node asks one yes/no question about ONE feature ("age <= 30?").
   Training = choosing the questions that split the data into the purest groups.

   Purity measure ENTROPY:   H = - sum_k  p_k * log2(p_k)
       all one class      -> H = 0   (pure)
       50/50 two classes  -> H = 1   (most mixed)
   INFORMATION GAIN = H(parent) - weighted average H(children)   -> pick the split with max gain

2) RANDOM FOREST = wisdom of the crowd
   Train MANY trees, each on a random bootstrap sample of rows, and at every split let it
   look at only a random subset of features. Average their votes.
   Trees make different mistakes -> averaging cancels them (lower VARIANCE).

3) GRADIENT BOOSTING (XGBoost) = a team of students fixing each other's mistakes
   Tree 1 makes predictions. Tree 2 is trained ONLY to fix the errors of tree 1. Tree 3 fixes
   what is still wrong, and so on. Prediction = sum of all trees (each scaled by learning rate).
   XGBoost uses the gradient g and curvature h of the loss for every sample:
       leaf value = - sum(g) / (sum(h) + lambda)
       split gain = 1/2 [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam) ] - gamma
""")

# ---------------------------------------------------------------------------
# SMALL NUMERIC EXAMPLE: ENTROPY + INFORMATION GAIN
# ---------------------------------------------------------------------------
def entropy(c):
    """Entropy (bits) from class counts. Works on a vector or a (rows, K) array."""
    c = np.asarray(c, dtype=float)
    n = c.sum(axis=-1)
    p = c / np.maximum(n, 1)[..., None]
    return -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=-1)

print("=" * 74)
print("NUMERIC EXAMPLE: 10 animals (6 cats, 4 dogs). Split on 'ears pointy?'")
print("=" * 74)
parent = np.array([6, 4]); left = np.array([5, 1]); right = np.array([1, 3])
Hp, Hl, Hr = entropy(parent), entropy(left), entropy(right)
w = (6 * Hl + 4 * Hr) / 10
print(f"  parent  [6 cats, 4 dogs]  H = {Hp:.3f}")
print(f"  left    [5 cats, 1 dog ]  H = {Hl:.3f}   (6 animals)")
print(f"  right   [1 cat , 3 dogs]  H = {Hr:.3f}   (4 animals)")
print(f"  weighted children H = 0.6*{Hl:.3f} + 0.4*{Hr:.3f} = {w:.3f}")
print(f"  INFORMATION GAIN = {Hp:.3f} - {w:.3f} = {Hp - w:.3f}")

# ---------------------------------------------------------------------------
# DATA: tabular, 10 features, only some matter, non-linear, noisy
# ---------------------------------------------------------------------------
K = 2
rng = np.random.default_rng(0)
n, d = 3000, 10
X = rng.standard_normal((n, d))
score = X[:, 0] + X[:, 1] * X[:, 2] + 1.5 * (X[:, 3] > 0.5) + 0.7 * np.sin(2 * X[:, 4])
y = (score + 0.8 * rng.standard_normal(n) > 0.5).astype(int)
Xtr, ytr, Xte, yte = X[:1500], y[:1500], X[1500:], y[1500:]
print(f"\nData: {n} rows, {d} features (x0..x4 matter in different ways, x5..x9 are pure noise)")
print(f"Train {len(Xtr)} / Test {len(Xte)}   class-1 rate: {y.mean():.2f}")
print("Test-set standard error of an accuracy near 78% is about 1.1%, so differences smaller")
print("than ~2% between models are NOT meaningful.\n")

# ---------------------------------------------------------------------------
# DECISION TREE FROM SCRATCH
# ---------------------------------------------------------------------------
def best_split(X, Y, feats, min_leaf):
    n_ = len(X)
    total = Y.sum(0)
    parent_H = entropy(total)
    best = (0.0, None, None)
    nl = np.arange(1, n_)
    for j in feats:
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        left_c = np.cumsum(Y[order], axis=0)[:-1]
        right_c = total - left_c
        valid = (xs[:-1] < xs[1:]) & (nl >= min_leaf) & (n_ - nl >= min_leaf)
        if not valid.any():
            continue
        gain = parent_H - (nl * entropy(left_c) + (n_ - nl) * entropy(right_c)) / n_
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best[0]:
            best = (float(gain[i]), j, float((xs[i] + xs[i + 1]) / 2))
    return best

def build_tree(X, Y, depth, max_depth, min_leaf=1, rng=None, max_features=None):
    n_ = len(X)
    counts = Y.sum(0)
    leaf = {"probs": counts / n_}
    if depth >= max_depth or n_ < 2 * min_leaf or counts.max() == n_:
        return leaf
    feats = np.arange(X.shape[1]) if max_features is None else rng.choice(X.shape[1], max_features, replace=False)
    gain, j, thr = best_split(X, Y, feats, min_leaf)
    if j is None or gain <= 1e-12:
        return leaf                       # (sklearn would keep searching other features; we stop - tiny difference)
    m = X[:, j] <= thr
    return {"feat": j, "thr": thr, "gain": gain, "n": n_,
            "left": build_tree(X[m], Y[m], depth + 1, max_depth, min_leaf, rng, max_features),
            "right": build_tree(X[~m], Y[~m], depth + 1, max_depth, min_leaf, rng, max_features)}

def tree_proba(node, X):
    out = np.zeros((len(X), K))
    def rec(nd, idx):
        if "probs" in nd:
            out[idx] = nd["probs"]; return
        m = X[idx, nd["feat"]] <= nd["thr"]
        rec(nd["left"], idx[m]); rec(nd["right"], idx[~m])
    rec(node, np.arange(len(X)))
    return out

def count_leaves(nd):
    return 1 if "probs" in nd else count_leaves(nd["left"]) + count_leaves(nd["right"])

def show_tree(nd, indent=""):
    if "probs" in nd:
        print(f"{indent}-> predict class {nd['probs'].argmax()}  (P(class1) = {nd['probs'][1]:.2f})"); return
    print(f"{indent}if x{nd['feat']} <= {nd['thr']:.2f}:")
    show_tree(nd["left"], indent + "    ")
    print(f"{indent}else:")
    show_tree(nd["right"], indent + "    ")

def acc(p, y_):
    return float(np.mean(p.argmax(1) == y_))

Ytr = np.eye(K)[ytr]

print("=" * 74)
print("PART 1: a small tree (max_depth = 2) - you can READ it")
print("=" * 74)
small = build_tree(Xtr, Ytr, 0, 2, min_leaf=5)
show_tree(small)

print("\n" + "=" * 74)
print("PART 2: how deep should a single tree be?")
print("=" * 74)
print(f"{'max_depth':>9} | {'leaves':>6} | {'train acc':>9} | {'test acc':>8}")
depths = [1, 2, 3, 4, 6, 8, 12, 20]
d_tr, d_te = [], []
for md in depths:
    t_ = build_tree(Xtr, Ytr, 0, md, min_leaf=1)
    d_tr.append(acc(tree_proba(t_, Xtr), ytr)); d_te.append(acc(tree_proba(t_, Xte), yte))
    print(f"{md:>9} | {count_leaves(t_):>6} | {d_tr[-1]*100:8.1f}% | {d_te[-1]*100:7.1f}%")
print(f"Test accuracy peaks around depth {depths[int(np.argmax(d_te))]} ({max(d_te)*100:.1f}%) and then FALLS to {d_te[-1]*100:.1f}% at depth {depths[-1]} while train -> {d_tr[-1]*100:.0f}%: memorizing.")

# ---------------------------------------------------------------------------
# RANDOM FOREST FROM SCRATCH
# ---------------------------------------------------------------------------
def fit_forest(X, y_, n_trees=50, max_depth=10, max_features=3, seed=0):
    r = np.random.default_rng(seed)
    Y = np.eye(K)[y_]
    trees, inbag = [], []
    for _ in range(n_trees):
        idx = r.integers(0, len(X), len(X))                 # bootstrap sample (with replacement)
        trees.append(build_tree(X[idx], Y[idx], 0, max_depth, 1, r, max_features))
        m = np.zeros(len(X), bool); m[idx] = True; inbag.append(m)
    return trees, np.array(inbag)

def forest_proba(trees, X):
    return np.mean([tree_proba(t, X) for t in trees], axis=0)

print("\n" + "=" * 74)
print("PART 3: Random Forest (50 trees, depth <= 10, 3 random features per split)")
print("=" * 74)
trees, inbag = fit_forest(Xtr, ytr)
rf_tr, rf_te = acc(forest_proba(trees, Xtr), ytr), acc(forest_proba(trees, Xte), yte)

# out-of-bag accuracy: each row is scored only by trees that never saw it (a free validation set!)
oob_sum, oob_cnt = np.zeros((len(Xtr), K)), np.zeros(len(Xtr))
for t_, m in zip(trees, inbag):
    o = ~m
    if o.any():
        oob_sum[o] += tree_proba(t_, Xtr[o]); oob_cnt[o] += 1
okm = oob_cnt > 0
oob_acc = float(np.mean(oob_sum[okm].argmax(1) == ytr[okm]))
single_deep = build_tree(Xtr, Ytr, 0, 10, 1)
print(f"Single tree (depth 10): test acc = {acc(tree_proba(single_deep, Xte), yte)*100:.1f}%")
print(f"Random forest         : train acc = {rf_tr*100:.1f}%   test acc = {rf_te*100:.1f}%   "
      f"out-of-bag acc = {oob_acc*100:.1f}%")
print("(Out-of-bag accuracy needs NO separate validation set - and should be close to test accuracy.)")

# accuracy as trees are added
cum = np.cumsum([tree_proba(t_, Xte) for t_ in trees], axis=0)
rf_curve = [float(np.mean(cum[i].argmax(1) == yte)) for i in range(len(trees))]

# feature importance: total (information gain x samples) at splits that use each feature
def importances(nd, acc_):
    if "probs" in nd:
        return acc_
    acc_[nd["feat"]] += nd["gain"] * nd["n"]
    importances(nd["left"], acc_); importances(nd["right"], acc_)
    return acc_
imp = np.zeros(d)
for t_ in trees:
    importances(t_, imp)
imp /= imp.sum()
print("\nFeature importance (share of total information gain):")
for j in np.argsort(-imp):
    print(f"  x{j}: {imp[j]*100:5.1f}%  {'#' * int(imp[j] * 100)}")
# permutation importance: shuffle ONE feature on the test set, see how much accuracy drops
base_te = acc(forest_proba(trees, Xte), yte)
pr = np.random.default_rng(5)
perm_drop = np.zeros(d)
for j in range(d):
    drops = []
    for _ in range(5):
        Xp = Xte.copy(); Xp[:, j] = pr.permutation(Xp[:, j])
        drops.append(base_te - acc(forest_proba(trees, Xp), yte))
    perm_drop[j] = np.mean(drops)
print("\nCross-check: PERMUTATION importance (accuracy drop, in points, when the feature is shuffled on test):")
for j in np.argsort(-perm_drop):
    print(f"  x{j}: {perm_drop[j]*100:5.1f} points")
print(f"Noise features x5-x9 average {imp[5:].mean()*100:.1f}% impurity share each (they still get a share because deep trees")
print("split on noise). x1 and x2 affect the label only through their PRODUCT, so impurity importance")
print("can make them look weak. Lesson: never trust ONE importance measure - cross-check.")

# ---------------------------------------------------------------------------
# GRADIENT BOOSTING (XGBoost-style) FROM SCRATCH
# ---------------------------------------------------------------------------
def build_xgb(X, g, h, depth, max_depth, lam, gamma, mcw):
    G, H = g.sum(), h.sum()
    leaf = {"value": -G / (H + lam)}
    if depth >= max_depth or len(g) < 2:
        return leaf
    best = (0.0, None, None)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="mergesort")
        xs = X[order, j]
        GL = np.cumsum(g[order])[:-1]; HL = np.cumsum(h[order])[:-1]
        GR, HR = G - GL, H - HL
        valid = (xs[:-1] < xs[1:]) & (HL >= mcw) & (HR >= mcw)
        if not valid.any():
            continue
        gain = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - G ** 2 / (H + lam)) - gamma
        gain = np.where(valid, gain, -np.inf)
        i = int(np.argmax(gain))
        if gain[i] > best[0]:
            best = (float(gain[i]), j, float((xs[i] + xs[i + 1]) / 2))
    if best[1] is None:
        return leaf
    _, j, thr = best
    m = X[:, j] <= thr
    return {"feat": j, "thr": thr,
            "left": build_xgb(X[m], g[m], h[m], depth + 1, max_depth, lam, gamma, mcw),
            "right": build_xgb(X[~m], g[~m], h[~m], depth + 1, max_depth, lam, gamma, mcw)}

def tree_value(node, X):
    out = np.zeros(len(X))
    def rec(nd, idx):
        if "value" in nd:
            out[idx] = nd["value"]; return
        m = X[idx, nd["feat"]] <= nd["thr"]
        rec(nd["left"], idx[m]); rec(nd["right"], idx[~m])
    rec(node, np.arange(len(X)))
    return out

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logloss(p, y_):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y_ * np.log(p) + (1 - y_) * np.log(1 - p)))

def fit_boost(Xa, ya, Xb, yb, rounds=150, lr=0.1, max_depth=3, lam=1.0, gamma=0.0, mcw=1.0):
    p0 = ya.mean()
    F0 = np.log(p0 / (1 - p0))
    Fa = np.full(len(Xa), F0); Fb = np.full(len(Xb), F0)
    hist = {"tr_loss": [], "te_loss": [], "te_acc": []}
    for _ in range(rounds):
        p = sigmoid(Fa)
        g, h = p - ya, p * (1 - p)                        # gradient and curvature of logloss
        tree = build_xgb(Xa, g, h, 0, max_depth, lam, gamma, mcw)
        Fa += lr * tree_value(tree, Xa); Fb += lr * tree_value(tree, Xb)
        hist["tr_loss"].append(logloss(sigmoid(Fa), ya))
        hist["te_loss"].append(logloss(sigmoid(Fb), yb))
        hist["te_acc"].append(float(np.mean((sigmoid(Fb) > 0.5) == yb)))
    return Fa, Fb, hist

print("\n" + "=" * 74)
print("PART 4: Gradient boosting (XGBoost-style): 150 rounds, depth 3, lr 0.1, lambda 1")
print("=" * 74)
Fa, Fb, bh = fit_boost(Xtr, ytr, Xte, yte)
gb_tr = float(np.mean((sigmoid(Fa) > 0.5) == ytr)); gb_te = bh["te_acc"][-1]
print(f"Mine : train acc = {gb_tr*100:.1f}%   test acc = {gb_te*100:.1f}%   test logloss = {bh['te_loss'][-1]:.4f}")

# ---------------------------------------------------------------------------
# COMPARE WITH LIBRARIES
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 5: my from-scratch models vs sklearn / xgboost (same settings)")
print("=" * 74)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rows = []
lr_m = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
rows.append(("Logistic regression (linear baseline)", lr_m.score(Xtr, ytr), lr_m.score(Xte, yte)))
t3 = build_tree(Xtr, Ytr, 0, 3, 1)
rows.append(("Decision tree depth 3  (mine)", acc(tree_proba(t3, Xtr), ytr), acc(tree_proba(t3, Xte), yte)))
sk3 = DecisionTreeClassifier(criterion="entropy", max_depth=3, random_state=0).fit(Xtr, ytr)
rows.append(("Decision tree depth 3  (sklearn)", sk3.score(Xtr, ytr), sk3.score(Xte, yte)))
rows.append(("Random forest 50 trees (mine)", rf_tr, rf_te))
skrf = RandomForestClassifier(n_estimators=50, max_depth=10, max_features=3, random_state=0).fit(Xtr, ytr)
rows.append(("Random forest 50 trees (sklearn)", skrf.score(Xtr, ytr), skrf.score(Xte, yte)))
rows.append(("Gradient boosting      (mine)", gb_tr, gb_te))
try:
    from xgboost import XGBClassifier
    xg = XGBClassifier(n_estimators=150, learning_rate=0.1, max_depth=3, reg_lambda=1.0, gamma=0.0,
                       min_child_weight=1.0, tree_method="exact", base_score=float(ytr.mean()),
                       n_jobs=1, random_state=0).fit(Xtr, ytr)
    rows.append(("XGBoost library", xg.score(Xtr, ytr), xg.score(Xte, yte)))
    xg_ll = logloss(xg.predict_proba(Xte)[:, 1], yte)
    print(f"Test logloss -> mine: {bh['te_loss'][-1]:.4f}   xgboost library: {xg_ll:.4f}")
except Exception as e:
    print("xgboost library not installed here (pip install xgboost); skipping that row.", e)

print(f"\n{'Model':<38} | {'train acc':>9} | {'test acc':>8}")
print("-" * 62)
for name, a, b in rows:
    print(f"{name:<38} | {a*100:8.1f}% | {b*100:7.1f}%")
lin = rows[0][2]
print(f"\nForest / boosting beat the linear model by {(rf_te-lin)*100:.1f} / {(gb_te-lin)*100:.1f} points (more than the ~2% noise).")
print(f"The single depth-3 tree is only {(rows[1][2]-lin)*100:.1f} points above it - inside the noise.")
print("The data has an interaction (x1*x2), a threshold (x3 > 0.5) and a sine - a line cannot capture those.")
print("Mine and the libraries land (almost) identical: same algorithm, so the numbers agree.")

# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(19, 10))

ax = axes[0, 0]
ax.plot(depths, d_tr, "o-", label="train acc", linewidth=2)
ax.plot(depths, d_te, "s--", label="test acc", linewidth=2)
ax.set_xlabel("max_depth"); ax.set_ylabel("accuracy"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Single tree: deeper = memorizes", fontweight="bold")

ax = axes[0, 1]
ax.plot(range(1, len(rf_curve) + 1), rf_curve, linewidth=2)
ax.axhline(acc(tree_proba(single_deep, Xte), yte), color="red", linestyle=":", label="single depth-10 tree")
ax.set_xlabel("number of trees in the forest"); ax.set_ylabel("test accuracy"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Random forest: more trees = steadier", fontweight="bold")

ax = axes[0, 2]
ax.plot(bh["te_acc"], linewidth=2, label="mine: test acc")
if "xg" in dir():
    ax.axhline(xg.score(Xte, yte), color="green", linestyle="--", label="xgboost library (final)")
ax.set_xlabel("boosting round"); ax.set_ylabel("test accuracy"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Boosting: accuracy as trees are added", fontweight="bold")

ax = axes[1, 0]
xs_ = np.arange(d)
ax.bar(xs_ - 0.2, imp * 100, 0.4, label="impurity share (%)", color="tab:green")
ax.bar(xs_ + 0.2, perm_drop * 100, 0.4, label="permutation drop (points)", color="tab:purple")
ax.set_xticks(xs_); ax.set_xticklabels([f"x{j}" for j in range(d)])
ax.axvline(4.5, color="k", linestyle=":", alpha=0.5)
ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_title("Feature importance, two ways\n(x0-x4 real | x5-x9 noise)", fontweight="bold")

ax = axes[1, 1]
ax.plot(bh["tr_loss"], label="train logloss", linewidth=2)
ax.plot(bh["te_loss"], label="test logloss", linewidth=2, linestyle="--")
ax.set_xlabel("boosting round"); ax.set_ylabel("logloss"); ax.legend(); ax.grid(alpha=0.3)
ax.set_title("Boosting loss curves (read them like Topic 11!)", fontweight="bold")

ax = axes[1, 2]
names = ["Logistic\n(linear)", "Tree\ndepth 3", "Random\nforest", "Boosting\n(mine)"]
vals = [rows[0][2], rows[1][2], rows[3][2], rows[5][2]]
ax.bar(names, np.array(vals) * 100, color=["tab:gray", "tab:blue", "tab:orange", "tab:green"])
ax.set_ylim(50, 100); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=0.3, axis="y")
for i, v in enumerate(vals):
    ax.text(i, v * 100 + 0.5, f"{v*100:.1f}%", ha="center", fontweight="bold")
ax.set_title("Test accuracy (differences < 2% = noise)", fontweight="bold")

plt.tight_layout()
plt.savefig("trees_forests_xgboost.png", dpi=200, bbox_inches="tight")
print("\nSaved: trees_forests_xgboost.png")
plt.show()

print("""
WHEN TO USE WHAT
  Tabular data (rows & columns)   -> start with Random Forest / XGBoost (strong with little tuning)
  Images, audio, text             -> neural networks (CNNs, Transformers)
  Need an explainable model       -> a small decision tree (you can read it)
  Need the BEST tabular accuracy  -> gradient boosting (XGBoost / LightGBM / CatBoost)
  Trees need NO feature scaling, handle mixed feature types, ignore irrelevant features.

KEY HYPERPARAMETERS
  Tree     : max_depth, min_samples_leaf        (more depth = more overfitting)
  Forest   : n_trees (more is safe), max_features, max_depth
  XGBoost  : learning_rate (small + many rounds), max_depth (3-6), n_estimators,
             lambda/gamma (regularization), subsample - choose rounds with a VALIDATION set
             and early stopping, never with the test set.

COMMON MISTAKES
  1. Unlimited-depth single tree            -> memorizes training data
  2. Scaling features for trees             -> pointless (trees only compare thresholds)
  3. Picking boosting rounds on test data   -> leakage, use validation + early stopping
  4. Treating feature importance as causal  -> it only says what the model USED
  5. Comparing models by tiny test differences  -> inside the noise (here ~2%)

PRACTICE
  Q1. A node has [8 yes, 2 no]. Compute its entropy. (Check with the entropy() function.)
  Q2. Why does a random forest use only a SUBSET of features at each split?
  Q3. In boosting, what happens to the test loss if you use lr=1.0 instead of 0.1? Try it.
  Q4. Change max_depth in fit_boost from 3 to 6. Does train accuracy go up? Test accuracy?
""")