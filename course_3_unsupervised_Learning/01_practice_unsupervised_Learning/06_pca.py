

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression

np.random.seed(42)




print("=" * 60)
print("LEVEL 1: Intuition — 2D data projected to 1D")
print("=" * 60)

# Create 2D data with a strong dominant direction
angle = np.pi / 4          # 45 degrees
cov   = np.array([[3, 2.5], [2.5, 3]])
X_2d  = np.random.multivariate_normal([0, 0], cov, size=100)

# The dominant direction is [1/√2, 1/√2] — along the diagonal
print(f"Data shape      : {X_2d.shape}")
print(f"Variance along x: {X_2d[:, 0].var():.2f}")
print(f"Variance along y: {X_2d[:, 1].var():.2f}")
print(f"Covariance x,y  : {np.cov(X_2d, rowvar=False)[0,1]:.2f}")
print("(High covariance → correlated → PCA will find a better direction)\n")

plt.figure(figsize=(5, 5))
plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.5)
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title("2D data — correlated features")
plt.tight_layout()
plt.savefig("learn_c3_06a_2d_data.png")
plt.close()
print("Saved: learn_c3_06a_2d_data.png\n")




print("=" * 60)
print("LEVEL 2: PCA from scratch — step by step")
print("=" * 60)

class PCAFromScratch:
    """
    PCA implemented without sklearn.

    Attributes after fit():
        mean_        : (n,) feature means
        components_  : (k, n) top-k eigenvectors (rows)
        eigenvalues_ : (k,)  corresponding eigenvalues
        explained_variance_ratio_ : (k,) fraction of variance per component
    """
    def __init__(self, n_components):
        self.k = n_components

    def fit(self, X):
        m, n = X.shape

        # Step 1: Centre
        self.mean_ = X.mean(axis=0)               # (n,)
        X_c = X - self.mean_                       # (m, n)

        # Step 2: Covariance matrix
        Sigma = (X_c.T @ X_c) / (m - 1)           # (n, n)

        # Step 3: Eigen-decomposition
        # np.linalg.eigh: symmetric matrix → real eigenvalues, stable
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma)
        # eigenvectors[:, i] = i-th eigenvector (column)

        # Step 4: Sort largest first (eigh returns ascending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store top k
        self.eigenvalues_  = eigenvalues[:self.k]
        self.components_   = eigenvectors[:, :self.k].T  # (k, n) — rows are PCs
        self.explained_variance_ratio_ = (
            eigenvalues[:self.k] / eigenvalues.sum()
        )
        return self

    def transform(self, X):
        """Project X onto the top-k principal components."""
        X_c = X - self.mean_
        return X_c @ self.components_.T   # (m, k)

    def inverse_transform(self, Z):
        """Reconstruct from projected coordinates back to original space."""
        return Z @ self.components_ + self.mean_  # (m, n)


# Apply to 2D data
pca_2d = PCAFromScratch(n_components=2).fit(X_2d)

print("Eigenvalues (variance captured by each PC):")
for i, ev in enumerate(pca_2d.eigenvalues_):
    print(f"  PC{i+1}: {ev:.4f}  ({pca_2d.explained_variance_ratio_[i]*100:.1f}% of variance)")

print(f"\nPC1 direction: {pca_2d.components_[0].round(4)}")
print(f"PC2 direction: {pca_2d.components_[1].round(4)}")
print("(PC1 should be close to [0.707, 0.707] — diagonal = most variance)\n")

# Verify PCs are orthogonal
dot = float(pca_2d.components_[0] @ pca_2d.components_[1])
print(f"PC1 · PC2 = {abs(round(dot, 12))}  (should be 0 — orthogonal)\n")

# Project data
Z_2d = pca_2d.transform(X_2d)           # (100, 2) in PC space

plt.figure(figsize=(5, 5))
plt.scatter(Z_2d[:, 0], Z_2d[:, 1], alpha=0.5, color='purple')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.title("Data in PCA space (PC1 = most variance)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("learn_c3_06b_pca_space.png")
plt.close()
print("Saved: learn_c3_06b_pca_space.png\n")






# Use the digits dataset (64 features — 8×8 pixel images of digits)
digits      = load_digits()
X_dig       = digits.data.astype(float)        # (1797, 64)
scaler      = StandardScaler()
X_dig_scaled = scaler.fit_transform(X_dig)

# Fit PCA with all components to get full EVR curve
pca_full = PCAFromScratch(n_components=64).fit(X_dig_scaled)
cumulative_evr = np.cumsum(pca_full.explained_variance_ratio_)

print("Components needed to reach each variance threshold:")
for threshold in [0.80, 0.90, 0.95, 0.99]:
    k_needed = int(np.searchsorted(cumulative_evr, threshold)) + 1
    print(f"  {threshold*100:.0f}% variance → k = {k_needed} components  "
          f"(out of 64)")

plt.figure(figsize=(7, 4))
plt.plot(range(1, 65), cumulative_evr * 100, marker='.', markersize=3)
plt.axhline(90, color='orange', linestyle='--', label='90%')
plt.axhline(95, color='red',    linestyle='--', label='95%')
plt.xlabel("Number of components (k)")
plt.ylabel("Cumulative explained variance (%)")
plt.title("Digits dataset — how many PCs do you need?")
plt.legend()
plt.tight_layout()
plt.savefig("learn_c3_06c_evr_curve.png")
plt.close()
print("Saved: learn_c3_06c_evr_curve.png\n")


print("=" * 60)
print("LEVEL 4a: PCA before logistic regression")
print("=" * 60)

from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X_dig_scaled, digits.target, test_size=0.2, random_state=42
)

results = {}
for k in [5, 10, 20, 30, 40, 64]:
    pca_k = PCAFromScratch(n_components=k).fit(X_tr)
    Z_tr  = pca_k.transform(X_tr)
    Z_te  = pca_k.transform(X_te)

    clf   = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(Z_tr, y_tr)
    acc   = clf.score(Z_te, y_te)
    results[k] = acc
    print(f"  k={k:2d} → test accuracy = {acc:.4f}")

print(f"\nBest k: {max(results, key=results.get)}")
print("(You don't need all 64 features to get high accuracy)\n")


print("=" * 60)
print("LEVEL 4b: Reconstruction error vs k")
print("=" * 60)

for k in [1, 5, 10, 20, 30, 64]:
    pca_k  = PCAFromScratch(n_components=k).fit(X_tr)
    Z_tr_k = pca_k.transform(X_tr)
    X_rec  = pca_k.inverse_transform(Z_tr_k)
    err    = ((X_tr - X_rec) ** 2).mean()
    print(f"  k={k:2d} → reconstruction error = {err:.4f}")

print("(k=64 → error = 0, no information lost)\n")


print("=" * 60)
print("LEVEL 4c: Why scaling is critical for PCA")
print("=" * 60)

from sklearn.datasets import load_wine

wine = load_wine()
X_w = wine.data.astype(float)

# PCA WITHOUT scaling
pca_noscale = PCAFromScratch(n_components=3).fit(X_w)
print("Without scaling — EVR of top 3 PCs:")
for i, r in enumerate(pca_noscale.explained_variance_ratio_):
    print(f"  PC{i+1}: {r*100:.1f}%  (loading: feature with largest range dominates)")

# PCA WITH scaling
scaler_w     = StandardScaler()
X_w_scaled   = scaler_w.fit_transform(X_w)
pca_scaled   = PCAFromScratch(n_components=3).fit(X_w_scaled)
print("\nWith scaling — EVR of top 3 PCs:")
for i, r in enumerate(pca_scaled.explained_variance_ratio_):
    print(f"  PC{i+1}: {r*100:.1f}%  (balanced contribution from all features)")

print("\nConclusion: without scaling, one high-variance feature hijacks all PCs.")



print("\n" + "=" * 60)
print("Verify scratch vs sklearn PCA")
print("=" * 60)

k_check = 10
sk_pca  = SklearnPCA(n_components=k_check).fit(X_tr)
my_pca  = PCAFromScratch(n_components=k_check).fit(X_tr)

Z_sk = sk_pca.transform(X_tr)
Z_my = my_pca.transform(X_tr)

# Sign of eigenvectors is arbitrary — compare absolute values
diff = np.abs(np.abs(Z_sk) - np.abs(Z_my)).max()
print(f"Max projection difference (abs): {diff:.2e}  (should be < 1e-10)")


