"""
 K-MEANS CLUSTERING

Unsupervised learning: find groups in data that has NO labels.
"""

import numpy as np
import matplotlib.pyplot as plt
 
print("-*-" * 14)
print("K-MEANS CLUSTERING")
print("*-*" * 14)


"""
THE ALGORITHM
  Repeat until nothing changes:
    assign : c(i)  = argmin_k  || x(i) - mu_k ||^2        (nearest centroid)
    update : mu_k  = average of all points with c(i) = k
 
THE COST FUNCTION (distortion)
  J = (1/m) * sum_i || x(i) - mu_c(i) ||^2
  x(i)      = the i-th data point          mu_k = centroid k
  c(i)      = cluster index of point i     m    = number of points
  J = average squared distance from each point to ITS centroid. 
  Smaller = tighter clusters.
  Both steps can only LOWER J (or keep it the same), so K-means always stops.
  But it may stop at a LOCAL minimum - the starting positions matter!

"""


# NUMERIC EXAMPLE BY HAND (1-D)

x1 = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
mu = np.array([1.0, 2.0])
print(f"points = {x1}   starting centroids = {mu}")
for it in range(1, 4):
    d = np.abs(x1[:, None] - mu[None, :])
    c = d.argmin(axis=1)
    J = np.mean((x1 - mu[c]) ** 2)
    new_mu = np.array([x1[c == k].mean() for k in range(2)])
    print(f"\n iteration {it}")
    print(f"   assign: cluster of each point = {c}      (J = {J:.3f})")
    print(f"   update: new centroids = {new_mu}")
    if np.allclose(new_mu, mu):
        print("   centroids did not move -> DONE")
        break
    mu = new_mu

# K-MEANS FROM SCRATCH

def sq_dists(X, C):
    """Squared distance from every row of X to every row of C -> shape (m, K)."""
    d = (X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None, :]
    return np.maximum(d, 0)
 
def init_centroids(X, K, rng, method):
    if method == "random":                       # K distinct random data points
        return X[rng.choice(len(X), K, replace=False)].copy()
    # k-means++ : first centroid random, each next one far from the existing ones
    C = [X[rng.integers(len(X))]]
    for _ in range(1, K):
        d2 = sq_dists(X, np.array(C)).min(axis=1)
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    return np.array(C)
 
def kmeans(X, K, n_iter=100, seed=0, init="random"):
    rng = np.random.default_rng(seed)
    C = init_centroids(X, K, rng, init)
    costs, history = [], []
    m = len(X)
    for _ in range(n_iter):
        D = sq_dists(X, C)
        labels = D.argmin(axis=1)
        point_cost = D[np.arange(m), labels]
        costs.append(point_cost.mean())
        history.append((C.copy(), labels.copy()))
        newC = C.copy()
        for k in range(K):
            pts = X[labels == k]
            newC[k] = pts.mean(axis=0) if len(pts) else X[point_cost.argmax()]   # empty cluster -> move to worst-fit point
        if np.allclose(newC, C):
            break
        C = newC
    D = sq_dists(X, C)
    labels = D.argmin(axis=1)
    cost = D[np.arange(m), labels].mean()
    return C, labels, cost, costs, history
 
def kmeans_best(X, K, n_init=10, init="random", seed0=0):
    """Run K-means n_init times from different starts, keep the LOWEST cost."""
    best = None
    for s in range(n_init):
        res = kmeans(X, K, seed=seed0 + s, init=init)
        if best is None or res[2] < best[2]:
            best = res
    return best
 
def silhouette(X, labels):
    """Mean silhouette: (b - a) / max(a, b). a = avg distance to own cluster, b = avg distance to the nearest other cluster."""
    D = np.sqrt(sq_dists(X, X))
    K = labels.max() + 1
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = labels == labels[i]
        if same.sum() <= 1:
            continue
        a = D[i, same].sum() / (same.sum() - 1)
        b = min(D[i, labels == k].mean() for k in range(K) if k != labels[i] and (labels == k).any())
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


# DATA GENERATORS

def make_blobs(centers, n_each, std, seed):
    r = np.random.default_rng(seed)
    X = np.vstack([c + std * r.standard_normal((n_each, 2)) for c in centers])
    y = np.repeat(np.arange(len(centers)), n_each)
    return X, y
 
def make_moons(n, noise, seed):
    r = np.random.default_rng(seed)
    t = r.uniform(0, np.pi, n // 2)
    a = np.c_[np.cos(t), np.sin(t)]
    b = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    X = np.vstack([a, b]) + r.normal(0, noise, (n // 2 * 2, 2))
    return X, np.r_[np.zeros(n // 2), np.ones(n // 2)].astype(int)


# PART 1: K-MEANS IN ACTION + CORRECTNESS CHECKS

print("\n" + "=" * 74)
print("PART 1: K-means on 3 blobs - does it behave as the theory says?")
print("=" * 74)
X3, y3 = make_blobs([(-4, -3), (0, 4), (5, -2)], 100, 1.0, seed=0)
C3, lab3, J3, costs3, hist3 = kmeans(X3, 3, seed=1)
print(f"Converged in {len(costs3) - 1} iterations. Final cost J = {J3:.4f}")
print("Cost after each iteration:", np.round(costs3, 3))
mono = all(costs3[i + 1] <= costs3[i] + 1e-12 for i in range(len(costs3) - 1))
print(f"Cost never went UP (theory says it can't): {mono}")

# PART 2: INITIALIZATION MATTERS (local minima)


X5, y5 = make_blobs([(-6, -2), (-5, 3), (0, 0.5), (5, -3), (6, 2.5)], 100, 0.9, seed=2)
final_rand = np.array([kmeans(X5, 5, seed=s, init="random")[2] for s in range(100)])
final_pp = np.array([kmeans(X5, 5, seed=s, init="kmeans++")[2] for s in range(100)])
best_known = min(final_rand.min(), final_pp.min())
hit = lambda a: np.mean(a < best_known * 1.001) * 100
print(f"Best cost found anywhere: {best_known:.4f}")
print(f"  random init   : {hit(final_rand):5.1f}% of runs reach it | worst run J = {final_rand.max():.3f}")
print(f"  k-means++ init: {hit(final_pp):5.1f}% of runs reach it | worst run J = {final_pp.max():.3f}")
print("A bad start can split one real cluster in two and merge two others (a local minimum).")
print("FIX 1: run K-means 10-100 times, keep the lowest-cost result.   FIX 2: k-means++ starts.")
 
# verify against sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
mine = kmeans_best(X5, 5, n_init=10, init="kmeans++")
sk = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X5)
print(f"\nCHECK vs sklearn (best of 10 runs each): mine J = {mine[2]:.4f} | sklearn J = {sk.inertia_/len(X5):.4f}")
print(f"  agreement with the TRUE blobs (adjusted Rand index, 1 = perfect): mine {adjusted_rand_score(y5, mine[1]):.3f} | sklearn {adjusted_rand_score(y5, sk.labels_):.3f}")
 
# ---------------------------------------------------------------------------
# PART 3: CHOOSING K
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 3: how many clusters? (data really has 4)")
print("=" * 74)
X4, y4 = make_blobs([(-5, -5), (-5, 5), (5, -5), (5, 5)], 150, 1.2, seed=3)
Ks = list(range(1, 9))
elbow, sil = [], {}
for K in Ks:
    res = kmeans_best(X4, K, n_init=10, init="kmeans++")
    elbow.append(res[2])
    if K >= 2:
        sil[K] = silhouette(X4, res[1])
print(f"{'K':>2} | {'cost J':>8} | {'silhouette':>10}")
for K in Ks:
    print(f"{K:>2} | {elbow[K-1]:8.3f} | {sil[K]:10.3f}" if K >= 2 else f"{K:>2} | {elbow[K-1]:8.3f} | {'-':>10}")
bestK = max(sil, key=sil.get)
print(f"\nELBOW: cost drops sharply until K = 4, then barely improves -> 'elbow' at 4.")
print(f"SILHOUETTE (higher = better separated): best at K = {bestK}")
chk = silhouette_score(X4, kmeans_best(X4, 4, 10, "kmeans++")[1])
print(f"CHECK vs sklearn silhouette at K=4: mine {sil[4]:.4f} | sklearn {chk:.4f}")
print("Warning: real data often has NO clear elbow. Pick K from the PURPOSE (e.g. 'we can run 3 campaigns').")


# PART 4: WHEN K-MEANS FAILS

Xm, ym = make_moons(400, 0.08, seed=4)
resm = kmeans_best(Xm, 2, 10, "kmeans++")
print(f"1) Two moons (curved clusters), K = 2: adjusted Rand index = {adjusted_rand_score(ym, resm[1]):.3f}  (1 = perfect, 0 = random)")
print("   K-means draws straight boundaries around ROUND blobs - it cannot follow curved shapes.")
 
r = np.random.default_rng(5)
n2 = 300
age = np.r_[r.normal(25, 3, n2), r.normal(40, 3, n2)]            # the real signal (small numbers)
income = r.normal(60000, 15000, 2 * n2)                         # irrelevant, but HUGE numbers
Xs = np.c_[age, income]; ys = np.r_[np.zeros(n2), np.ones(n2)].astype(int)
raw = kmeans_best(Xs, 2, 10, "kmeans++")
Xz = (Xs - Xs.mean(0)) / Xs.std(0)
scaled = kmeans_best(Xz, 2, 10, "kmeans++")
print(f"2) Feature scale: true groups differ only in AGE; income (tens of thousands) is irrelevant noise.")
print(f"   raw features    : adjusted Rand index = {adjusted_rand_score(ys, raw[1]):.3f}   (income swamps the distances)")
print(f"   standardized    : adjusted Rand index = {adjusted_rand_score(ys, scaled[1]):.3f}")
print("   ALWAYS scale features first (subtract mean, divide by std).")
print("3) Also weak with: very different cluster sizes/densities, outliers (they drag centroids), many dimensions.")
 
# PART 5: IMAGE COMPRESSION

try:
    from sklearn.datasets import load_sample_image
    img = load_sample_image("china.jpg").astype(float) / 255.0
    src = "sample photo (china.jpg, bundled with scikit-learn)"
except Exception as e:
    yy, xx = np.mgrid[0:200, 0:300]
    img = np.zeros((200, 300, 3))
    img[..., 2] = 0.4 + 0.6 * (1 - yy / 200)
    img[yy > 130] = [0.2, 0.6, 0.2]
    img[((xx - 220) ** 2 + (yy - 50) ** 2) < 400] = [1.0, 0.9, 0.2]
    src = f"synthetic picture (sample photo unavailable: {type(e).__name__})"
H, W, _ = img.shape
pix = img.reshape(-1, 3)
print(f"Image: {src}, {H}x{W} = {H*W:,} pixels, each pixel = 3 numbers (R, G, B)")
rs = np.random.default_rng(0)
train_pix = pix[rs.choice(len(pix), 20000, replace=False)]      # fit on a sample, assign ALL pixels afterwards
ks_img = [2, 4, 8, 16, 32]
comp = {}
print(f"\n{'colors K':>8} | {'bits/pixel':>10} | {'size vs 24-bit':>14} | {'avg error (0-255 scale)':>24}")
for K in ks_img:
    Cc, _, _, _, _ = kmeans_best(train_pix, K, n_init=3, init="kmeans++")
    lab = sq_dists(pix, Cc).argmin(axis=1)
    out = Cc[lab].reshape(H, W, 3)
    comp[K] = out
    bits = int(np.ceil(np.log2(K)))
    total_bits = H * W * bits + K * 24
    err = np.sqrt(np.mean((out - img) ** 2)) * 255
    print(f"{K:>8} | {bits:>10} | {total_bits/(H*W*24)*100:12.1f}% | {err:24.1f}")
print("Store a palette of K colors + one small index per pixel instead of 3 full numbers per pixel.")

fig, axes = plt.subplots(3, 4, figsize=(20, 14))
cols = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
 
show_its = [0, 1, 2, len(hist3) - 1]
for ax, it in zip(axes[0], show_its):
    Cc, lab = hist3[it]
    for k in range(3):
        ax.scatter(X3[lab == k, 0], X3[lab == k, 1], s=12, color=cols[k], alpha=0.6)
    ax.scatter(Cc[:, 0], Cc[:, 1], marker="X", s=250, c="k", edgecolors="white", linewidths=1.5)
    ax.set_title(f"Iteration {it}" + ("  (final)" if it == len(hist3) - 1 else "") + f"   J = {costs3[it]:.2f}", fontweight="bold")
    ax.grid(alpha=0.3)
 
ax = axes[1, 0]
bins = np.linspace(min(final_rand.min(), final_pp.min()) * 0.98, max(final_rand.max(), final_pp.max()) * 1.02, 30)
ax.hist(final_rand, bins=bins, alpha=0.6, label="random init")
ax.hist(final_pp, bins=bins, alpha=0.6, label="k-means++ init")
ax.set_xlabel("final cost J of one run"); ax.set_ylabel("number of runs (out of 100)")
ax.set_title("Starting points matter (local minima)", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
 
ax = axes[1, 1]
ax.plot(Ks, elbow, "o-", linewidth=2)
ax.axvline(4, color="red", linestyle=":", alpha=0.6)
ax.set_xlabel("K"); ax.set_ylabel("cost J"); ax.set_title("Elbow method (true K = 4)", fontweight="bold"); ax.grid(alpha=0.3)
 
ax = axes[1, 2]
ax.plot(list(sil.keys()), list(sil.values()), "s-", color="tab:green", linewidth=2)
ax.set_xlabel("K"); ax.set_ylabel("mean silhouette"); ax.set_title(f"Silhouette (best K = {bestK})", fontweight="bold"); ax.grid(alpha=0.3)
 
ax = axes[1, 3]
for k in range(2):
    ax.scatter(Xm[resm[1] == k, 0], Xm[resm[1] == k, 1], s=12, color=cols[k], alpha=0.7)
ax.scatter(resm[0][:, 0], resm[0][:, 1], marker="X", s=250, c="k", edgecolors="white", linewidths=1.5)
ax.set_title(f"Failure: two moons, K=2 (ARI = {adjusted_rand_score(ym, resm[1]):.2f})", fontweight="bold"); ax.grid(alpha=0.3)
 
axes[2, 0].imshow(img); axes[2, 0].set_title(f"Original ({H*W:,} pixels, millions of colors)", fontweight="bold")
for ax, K in zip(axes[2, 1:], [2, 8, 32]):
    ax.imshow(np.clip(comp[K], 0, 1)); ax.set_title(f"{K} colors", fontweight="bold")
for ax in axes[2]:
    ax.axis("off")
 
plt.tight_layout()
plt.savefig("c3_kmeans_results.png", dpi=160, bbox_inches="tight")
print("\nSaved: c3_kmeans_results.png")
plt.show()