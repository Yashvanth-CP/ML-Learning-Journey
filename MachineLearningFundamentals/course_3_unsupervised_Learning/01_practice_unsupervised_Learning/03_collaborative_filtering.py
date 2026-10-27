import numpy as np
 
np.random.seed(42)

Y_example = np.array([
    [5,   4,   np.nan, np.nan, 1],
    [np.nan, 5,   np.nan, 3,   np.nan],
    [4,   np.nan, np.nan, 2,   1],
    [np.nan, np.nan, 3,   np.nan, 4],
])
R_example = (~np.isnan(Y_example)).astype(float)
 
print("Y (ratings matrix — ? = not rated):")
for row in Y_example:
    print("  ", ["?" if np.isnan(v) else str(int(v)) for v in row])
 
print("\nR (which entries are known):")
print(R_example.astype(int))
print(f"\nTotal ratings known: {int(R_example.sum())} / {R_example.size}")

n_u = 4    # number of users
n_m = 5    # number of movies
n   = 3    # latent dimension (features per user/movie)
lam = 1.0  # regularisation strength λ
 
# Initialise parameters randomly
W = np.random.randn(n_u, n) * 0.1   # (n_u, n) — one row per user
b = np.zeros((n_u, 1))              # (n_u, 1) — one bias per user
X = np.random.randn(n_m, n) * 0.1   # (n_m, n) — one row per movie
 
# Replace NaN with 0 so we can do matrix math (R masks them out)
Y = np.where(np.isnan(Y_example), 0, Y_example)
R = R_example
 
 
def predict(W, X, b):
     return W @ X.T + b 


def cost(W, b, X, Y, R, lam):
    """
    J = ½ Σ_{R=1} (ŷ - y)² + (λ/2)(||W||² + ||X||²)
    """
    Y_hat = predict(W, X, b)              # (n_u, n_m)
    diff  = (Y_hat - Y) * R               # zero out unknown entries
    j     = 0.5 * (diff**2).sum()
    j    += (lam / 2) * (W**2).sum()
    j    += (lam / 2) * (X**2).sum()
    return j
 
j0 = cost(W, b, X, Y, R, lam)
print(f"Initial cost J: {j0:.4f}  (high — parameters are random)")

def gradients(W, b, X, Y, R, lam):
    Y_hat = predict(W, X, b)
    error = (Y_hat - Y) * R        # (n_u, n_m)
    dW    = error @ X + lam * W    # (n_u, n)
    db    = error.sum(axis=1, keepdims=True)  # (n_u, 1)
    dX    = error.T @ W + lam * X  # (n_m, n)
    return dW, db, dX


eps = 1e-5
dW, db, dX = gradients(W, b, X, Y, R, lam)
 
# Check one entry of dW
i, j_col = 0, 0
W_plus           = W.copy(); W_plus[i, j_col] += eps
W_minus          = W.copy(); W_minus[i, j_col] -= eps
num_grad_W       = (cost(W_plus, b, X, Y, R, lam) - cost(W_minus, b, X, Y, R, lam)) / (2*eps)
print(f"dW[0,0] analytical : {dW[i,j_col]:.6f}")
print(f"dW[0,0] numerical  : {num_grad_W:.6f}")
print(f"Difference         : {abs(dW[i,j_col] - num_grad_W):.2e}  (should be < 1e-5)\n")


n_u, n_m, n = 50, 80, 5
lam = 0.5
alpha = 0.01
 
# Generate ground-truth W*, X* and create noisy ratings with 30% missing
W_true = np.random.randn(n_u, n)
X_true = np.random.randn(n_m, n)
b_true = np.random.randn(n_u, 1)
Y_full = W_true @ X_true.T + b_true + np.random.randn(n_u, n_m) * 0.3
R_full = (np.random.rand(n_u, n_m) > 0.3).astype(float)   # 70% observed
 
# Mean-normalise
movie_means = (Y_full * R_full).sum(axis=0) / (R_full.sum(axis=0) + 1e-8)
Y_norm = (Y_full - movie_means[None, :]) * R_full   # (n_u, n_m)
 
# Initialise parameters
W = np.random.randn(n_u, n) * 0.1
b = np.zeros((n_u, 1))
X = np.random.randn(n_m, n) * 0.1
 
cost_history = []
for epoch in range(2000):
    j       = cost(W, b, X, Y_norm, R_full, lam)
    dW, db, dX = gradients(W, b, X, Y_norm, R_full, lam)
    W -= alpha * dW
    b -= alpha * db
    X -= alpha * dX
    cost_history.append(j)
 
print(f"Initial cost : {cost_history[0]:.4f}")
print(f"Final cost   : {cost_history[-1]:.4f}")
 
# Evaluate on known ratings (add mean back)
Y_hat_final = predict(W, X, b) + movie_means[None, :]
known_mask  = R_full.astype(bool)
rmse_train  = np.sqrt(((Y_hat_final[known_mask] - Y_full[known_mask])**2).mean())
baseline    = np.sqrt(((movie_means[None, :] * np.ones_like(Y_full))[known_mask] - Y_full[known_mask])**2).mean()
print(f"Train RMSE   : {rmse_train:.4f}  (baseline = predict-mean: {baseline:.4f})")
 
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 3))
plt.plot(cost_history)
plt.xlabel("Epoch")
plt.ylabel("Cost J")
plt.title("Collaborative filtering training curve")
plt.tight_layout()
plt.savefig("learn_c3_03a_training.png")
plt.show()
plt.close()
print("Saved: learn_c3_03a_training.png\n")
 
 
# =============================================================================
# LEVEL 4 — TOP-N RECOMMENDATIONS & COLD START
# =============================================================================
 
print("=" * 60)
print("LEVEL 4: Top-N recommendations and cold-start problem")
print("=" * 60)
 
# Top-5 movies for user 0 (exclude movies they already rated)
user = 0
scores = Y_hat_final[user]                    # predicted ratings for all movies
already_rated = R_full[user].astype(bool)     # which movies they rated
scores[already_rated] = -np.inf               # exclude already-seen
top5 = np.argsort(scores)[::-1][:5]
print(f"Top-5 recommended movies for user 0: {top5.tolist()}")
print(f"Their predicted ratings: {scores[top5].round(2)}")