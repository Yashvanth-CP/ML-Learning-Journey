# Kmeans Clustering 


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_blobs
 
np.random.seed(42)

print("=" * 60)
print("LEVEL 1: Generate and visualise 2D data")
print("=" * 60)

X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0)
 
print(f"Data shape : {X.shape}")   # (300, 2) — 300 examples, 2 features
print(f"True labels: {np.unique(y_true)}")  # 0, 1, 2
 
# Plot raw data (no labels — this is what the algorithm sees)
plt.figure(figsize=(6, 4))
plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5)
plt.title("Raw data — no labels")
plt.tight_layout()
plt.savefig("learn_c3_01a_raw_data.png")
plt.close()
print("Saved: learn_c3_01a_raw_data.png\n")

# LEVEL 2 — K-MEANS FROM SCRATCH

def assign_clusters(X, centroids):

    sq_dists = ((X[:, None, :] - centroids[None, : , :]) ** 2).sum(axis=2)
    return sq_dists.argmin(axis=1)

def update_centroids(X, labels, K):
    """
    Move each centroid to the mean of its assigned points.
 
    Returns new_centroids : (K, n)
    """
    n = X.shape[1]
    new_centroids = np.zeros((K, n))
    for k in range(K):
        pts = X[labels == k]         # all points assigned to cluster k
        if len(pts) > 0:
            new_centroids[k] = pts.mean(axis=0)
    return new_centroids
 
 
def compute_cost(X, labels, centroids):
    """
    J = (1/m) Σ ||xᵢ − μ_{cᵢ}||²
    """
    m = X.shape[0]
    total = 0.0
    for i in range(m):
        diff = X[i] - centroids[labels[i]]
        total += np.dot(diff, diff)
    return total / m
 
 
def kmeans_scratch(X, K, max_iters=100, seed=0):
    """
    Full K-means from scratch.
 
    Returns:
        centroids : (K, n) final centroid positions
        labels    : (m,)   final cluster assignment
        cost      : float  final distortion J
        history   : list of J values per iteration (should be non-increasing)
    """
    rng = np.random.default_rng(seed)
    # Initialise: pick K random examples as starting centroids
    idx = rng.choice(len(X), size=K, replace=False)
    centroids = X[idx].copy()
 
    history = []
    for _ in range(max_iters):
        labels    = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, K)
        cost      = compute_cost(X, labels, centroids)
        history.append(cost)
 
        # Stop early if centroids stopped moving
        if len(history) > 1 and abs(history[-1] - history[-2]) < 1e-8:
            break
 
    return centroids, labels, cost, history
 
 
K = 3
centroids, labels, cost, history = kmeans_scratch(X, K)
 
print(f"Final cost (J) : {cost:.4f}")
print(f"Iterations run : {len(history)}")
print(f"Cluster sizes  : {[int((labels==k).sum()) for k in range(K)]}")
 
# Plot convergence
plt.figure(figsize=(6, 3))
plt.plot(history, marker='o', markersize=3)
plt.xlabel("Iteration")
plt.ylabel("Cost J")
plt.title("K-means cost convergence (should never increase)")
plt.tight_layout()
plt.savefig("learn_c3_01b_cost_convergence.png")
plt.close()
print("Saved: learn_c3_01b_cost_convergence.png")
 
# Plot final clusters
colors = ['red', 'blue', 'green']
plt.figure(figsize=(6, 4))
for k in range(K):
    pts = X[labels == k]
    plt.scatter(pts[:, 0], pts[:, 1], c=colors[k], alpha=0.5, label=f"Cluster {k}")
plt.scatter(centroids[:, 0], centroids[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Centroids')
plt.title("K-means result (scratch)")
plt.legend()
plt.tight_layout()
plt.savefig("learn_c3_01c_clusters.png")
plt.close()
print("Saved: learn_c3_01c_clusters.png\n")
 
 
# =============================================================================
# LEVEL 3 — MULTIPLE RESTARTS
# =============================================================================
# Problem: K-means can get stuck in a local minimum depending on random init.
# Fix: run K-means many times with different seeds, keep the best (lowest J).
 
print("=" * 60)
print("LEVEL 3a: Multiple restarts")
print("=" * 60)
 
best_cost = np.inf
best_labels = None
best_centroids = None
 
for seed in range(20):                     # 20 different random starts
    c, l, cost_i, _ = kmeans_scratch(X, K=3, seed=seed)
    if cost_i < best_cost:
        best_cost      = cost_i
        best_labels    = l
        best_centroids = c
 
print(f"Best cost across 20 restarts: {best_cost:.4f}")
 
 
# =============================================================================
# LEVEL 3b — CHOOSING K: ELBOW METHOD
# =============================================================================
# Try K = 1, 2, ..., 8 and plot J vs K.
# The "elbow" is the K where the curve bends — after that point, adding more
# clusters gives diminishing returns.
 
print("\n" + "=" * 60)
print("LEVEL 3b: Elbow method")
print("=" * 60)
 
costs_by_k = []
for k in range(1, 9):
    best_j = np.inf
    for seed in range(5):
        _, _, j, _ = kmeans_scratch(X, K=k, seed=seed)
        if j < best_j:
            best_j = j
    costs_by_k.append(best_j)
    print(f"  K={k}: J={best_j:.4f}")
 
plt.figure(figsize=(6, 3))
plt.plot(range(1, 9), costs_by_k, marker='o')
plt.xlabel("K (number of clusters)")
plt.ylabel("Cost J")
plt.title("Elbow method — look for the bend")
plt.tight_layout()
plt.savefig("learn_c3_01d_elbow.png")
plt.close()
print("Saved: learn_c3_01d_elbow.png")
 
 
# =============================================================================
# LEVEL 3c — SILHOUETTE SCORE
# =============================================================================
# More rigorous than elbow. For each point:
#   a = mean distance to points in its own cluster  (compactness)
#   b = mean distance to points in the nearest other cluster  (separation)
#   s = (b - a) / max(a, b)    range: [-1, 1]   higher is better.
# Average s over all points → silhouette score for that K.
 
print("\n" + "=" * 60)
print("LEVEL 3c: Silhouette score")
print("=" * 60)
 
sil_scores = []
for k in range(2, 9):               # silhouette needs at least K=2
    _, labels_k, _, _ = kmeans_scratch(X, K=k, seed=0)
    s = silhouette_score(X, labels_k)
    sil_scores.append(s)
    print(f"  K={k}: silhouette={s:.4f}")
 
best_k = np.argmax(sil_scores) + 2
print(f"\nBest K by silhouette: {best_k}")
 
plt.figure(figsize=(6, 3))
plt.plot(range(2, 9), sil_scores, marker='o', color='purple')
plt.xlabel("K")
plt.ylabel("Silhouette score")
plt.title("Silhouette score — higher is better")
plt.tight_layout()
plt.savefig("learn_c3_01e_silhouette.png")
plt.close()
print("Saved: learn_c3_01e_silhouette.png\n")
 
 
# =============================================================================
# LEVEL 4 — COMPARE SCRATCH vs SKLEARN
# =============================================================================
 
print("=" * 60)
print("LEVEL 4: Scratch vs sklearn")
print("=" * 60)
 
km_sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X)
 
# Inertia = sklearn's name for the distortion cost (without the 1/m factor)
sklearn_cost = km_sk.inertia_ / len(X)
print(f"Scratch cost : {best_cost:.4f}")
print(f"Sklearn cost : {sklearn_cost:.4f}")
print(f"Difference   : {abs(best_cost - sklearn_cost):.6f}  (should be tiny)")
 
