# =============================================================================
# COURSE 1 — WEEK 3: LOGISTIC REGRESSION
# DeepLearning.AI — Andrew Ng
# Beginner → Intermediate
# =============================================================================
# WHAT YOU WILL LEARN:
#   Level 1 : Why not linear regression for classification?
#   Level 2 : Sigmoid function — the core of logistic regression
#   Level 3 : Model, decision boundary, prediction
#   Level 4 : Cost function — why NOT mean squared error
#   Level 5 : Gradient descent for logistic regression
#   Level 6 : Regularisation — overfitting fix
#   Level 7 : Full training loop + evaluation
#   Level 8 : Multi-class (Softmax)
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)


# =============================================================================
# LEVEL 1 — WHY NOT LINEAR REGRESSION FOR CLASSIFICATION?
# =============================================================================
# Classification: output y is 0 or 1  (e.g. spam / not spam)
#
# If you use linear regression:
#   ŷ = wx + b
#   This can give values like -0.3 or 2.7 — meaningless for 0/1 labels.
#   It is also very sensitive to outliers.
#
# We need a model that ALWAYS outputs a value between 0 and 1.
# That value = "probability that y = 1"
# Solution: pass the linear output through the SIGMOID function.

print("=" * 60)
print("LEVEL 1: Why linear regression fails for classification")
print("=" * 60)

# Example: predict tumour malignancy from size
sizes  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1,  1],  dtype=float)

# Fit a line
w_lin = np.polyfit(sizes, labels, 1)
y_lin = np.polyval(w_lin, sizes)

plt.figure(figsize=(8, 3))
plt.scatter(sizes, labels, color='red', zorder=5, label='True labels (0/1)')
plt.plot(sizes, y_lin, 'b--', label='Linear regression fit')
plt.axhline(0, color='gray', linewidth=0.5)
plt.axhline(1, color='gray', linewidth=0.5)
plt.ylim(-0.3, 1.3)
plt.title("Linear regression for classification — predicts outside [0,1]")
plt.legend()
plt.tight_layout()
plt.savefig("c1_w3_01_linear_fails.png")
plt.close()
print("Saved: c1_w3_01_linear_fails.png")
print(f"Linear regression predictions: {y_lin.round(2)}")
print("Some values outside [0,1] — meaningless as probabilities\n")


# =============================================================================
# LEVEL 2 — SIGMOID FUNCTION
# =============================================================================
# sigmoid(z) = 1 / (1 + e^(-z))
#
# Properties:
#   z → +∞  : sigmoid → 1
#   z → -∞  : sigmoid → 0
#   z = 0   : sigmoid = 0.5  (decision boundary)
#   Always outputs between 0 and 1 — perfect for probability!

print("=" * 60)
print("LEVEL 2: Sigmoid function")
print("=" * 60)

def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + e^(-z))

    z can be a scalar, vector, or matrix.
    Returns the same shape as input.
    """
    return 1.0 / (1.0 + np.exp(-z))

# Test values
test_z = np.array([-10, -3, -1, 0, 1, 3, 10])
print("z         :", test_z)
print("sigmoid(z):", sigmoid(test_z).round(4))
print()
print("Key values:")
print(f"  sigmoid(0)   = {sigmoid(0):.4f}  ← always 0.5 (decision boundary)")
print(f"  sigmoid(1)   = {sigmoid(1):.4f}")
print(f"  sigmoid(-1)  = {sigmoid(-1):.4f}")
print(f"  sigmoid(10)  = {sigmoid(10):.6f}  ← very close to 1")
print(f"  sigmoid(-10) = {sigmoid(-10):.6f}  ← very close to 0\n")

# Plot sigmoid
z_range = np.linspace(-8, 8, 300)
plt.figure(figsize=(7, 4))
plt.plot(z_range, sigmoid(z_range), 'b-', linewidth=2)
plt.axhline(0.5, color='orange', linestyle='--', label='threshold = 0.5')
plt.axhline(0,   color='gray',   linewidth=0.5)
plt.axhline(1,   color='gray',   linewidth=0.5)
plt.axvline(0,   color='gray',   linewidth=0.5)
plt.xlabel("z  (= wx + b)")
plt.ylabel("sigmoid(z)  = P(y=1)")
plt.title("Sigmoid function")
plt.legend()
plt.tight_layout()
plt.savefig("c1_w3_02_sigmoid.png")
plt.close()
print("Saved: c1_w3_02_sigmoid.png\n")


# =============================================================================
# LEVEL 3 — THE LOGISTIC REGRESSION MODEL
# =============================================================================
# Step 1: compute linear combination   z = w·x + b
# Step 2: apply sigmoid               f(x) = sigmoid(z) = σ(w·x + b)
#
# Output interpretation:
#   f(x) = P(y=1 | x)   "probability that the label is 1 given input x"
#
# Decision rule (threshold = 0.5 by default):
#   if f(x) >= 0.5  → predict ŷ = 1
#   if f(x) <  0.5  → predict ŷ = 0
#
# DECISION BOUNDARY:
#   f(x) = 0.5  when  z = 0  i.e.  w·x + b = 0
#   That equation defines the boundary between class 0 and class 1.

print("=" * 60)
print("LEVEL 3: Logistic regression model and decision boundary")
print("=" * 60)

# 1D example: tumour size → malignant?
w = 1.5
b = -6.0

x_range = np.linspace(0, 10, 300)
z_vals  = w * x_range + b
prob    = sigmoid(z_vals)

# Decision boundary: w*x + b = 0  → x = -b/w
boundary = -b / w
print(f"w = {w}, b = {b}")
print(f"Decision boundary: x = {boundary:.2f}  (predict 1 if size > {boundary:.2f})")

plt.figure(figsize=(8, 4))
plt.plot(x_range, prob, 'b-', linewidth=2, label='P(y=1|x) = sigmoid(wx+b)')
plt.axvline(boundary, color='red', linestyle='--', label=f'Decision boundary x={boundary:.1f}')
plt.axhline(0.5, color='orange', linestyle=':', alpha=0.7)
plt.scatter(sizes, labels, color='green', zorder=5, label='Training data')
plt.xlabel("Tumour size")
plt.ylabel("P(malignant)")
plt.title("Logistic Regression — model output")
plt.legend()
plt.tight_layout()
plt.savefig("c1_w3_03_model.png")
plt.close()
print("Saved: c1_w3_03_model.png\n")

# 2D example — decision boundary is a LINE in 2D feature space
print("2D decision boundary: w1*x1 + w2*x2 + b = 0")
w1, w2, b2 = 1.0, -2.0, 1.0
# Rearrange: x2 = (w1*x1 + b2) / (-w2)
x1_range = np.linspace(-3, 3, 100)
x2_bound = (w1 * x1_range + b2) / (-w2)

# Synthetic 2D data
X_pos = np.random.randn(30, 2) + np.array([1.5, -1.5])
X_neg = np.random.randn(30, 2) + np.array([-1.5, 1.5])
plt.figure(figsize=(5, 5))
plt.scatter(X_pos[:, 0], X_pos[:, 1], color='red',  label='y=1')
plt.scatter(X_neg[:, 0], X_neg[:, 1], color='blue', label='y=0')
plt.plot(x1_range, x2_bound, 'k-', linewidth=2, label='Decision boundary')
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("2D Decision Boundary: w·x + b = 0")
plt.legend(); plt.tight_layout()
plt.savefig("c1_w3_04_decision_boundary.png")
plt.close()
print("Saved: c1_w3_04_decision_boundary.png\n")


# =============================================================================
# LEVEL 4 — COST FUNCTION
# =============================================================================
# Why NOT mean squared error (MSE)?
#   MSE with sigmoid = non-convex (many local minima) → gradient descent gets stuck
#
# Correct cost: BINARY CROSS-ENTROPY (Log Loss)
#
# For one example:
#   loss = -y * log(f(x)) - (1-y) * log(1 - f(x))
#
# Intuition:
#   If y=1 and f(x)=1 → loss = -log(1) = 0           (perfect prediction)
#   If y=1 and f(x)=0 → loss = -log(0) = +∞          (terrible prediction)
#   If y=0 and f(x)=0 → loss = -log(1) = 0           (perfect)
#   If y=0 and f(x)=1 → loss = -log(0) = +∞          (terrible)
#
# Full cost (average over m examples):
#   J(w,b) = -(1/m) Σᵢ [ y⁽ⁱ⁾ log(f(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾) log(1 - f(x⁽ⁱ⁾)) ]
#
# This is CONVEX — gradient descent always finds the global minimum.

print("=" * 60)
print("LEVEL 4: Cost function — Binary Cross-Entropy")
print("=" * 60)

# Show why MSE is bad for logistic regression
print("Loss intuition:")
for y_true, f_x in [(1, 0.99), (1, 0.5), (1, 0.01), (0, 0.01), (0, 0.99)]:
    loss = -y_true * np.log(f_x + 1e-15) - (1 - y_true) * np.log(1 - f_x + 1e-15)
    print(f"  y={y_true}, f(x)={f_x:.2f} → loss = {loss:.3f}")


def compute_cost_logistic(X, y, w, b):
    """
    Binary cross-entropy cost.

    X : (m, n)  — features
    y : (m,)    — labels (0 or 1)
    w : (n,)    — weights
    b : float   — bias

    Returns: J (scalar)
    """
    m     = X.shape[0]
    z     = X @ w + b          # (m,)
    f     = sigmoid(z)          # (m,)  predicted probabilities

    # Clip to avoid log(0)
    f     = np.clip(f, 1e-15, 1 - 1e-15)

    cost  = -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))
    return cost

# Test
X_test = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
y_test = np.array([0, 0, 1], dtype=float)
w_test = np.array([0.1, -0.2])
b_test = 0.0
print(f"\nTest cost: {compute_cost_logistic(X_test, y_test, w_test, b_test):.4f}\n")


# =============================================================================
# LEVEL 5 — GRADIENT DESCENT
# =============================================================================
# The gradients of binary cross-entropy are (remarkably clean):
#
#   ∂J/∂w = (1/m) Xᵀ (f - y)
#   ∂J/∂b = (1/m) Σ (f - y)
#
# These look identical to linear regression gradients!
# The only difference: f = sigmoid(w·x + b) instead of w·x + b.
#
# Update rule (same as always):
#   w ← w - α * ∂J/∂w
#   b ← b - α * ∂J/∂b

print("=" * 60)
print("LEVEL 5: Gradient descent")
print("=" * 60)

def compute_gradient_logistic(X, y, w, b):
    """
    Gradients of binary cross-entropy w.r.t. w and b.

    Returns: dw (n,), db (scalar)
    """
    m   = X.shape[0]
    z   = X @ w + b        # (m,)
    f   = sigmoid(z)        # (m,)
    err = f - y             # (m,)  prediction error

    dw  = (1/m) * X.T @ err  # (n,)
    db  = (1/m) * err.sum()   # scalar
    return dw, db


def gradient_descent_logistic(X, y, w, b, alpha, n_iters):
    """
    Full gradient descent for logistic regression.

    Returns: w, b, cost_history
    """
    cost_history = []
    for i in range(n_iters):
        dw, db = compute_gradient_logistic(X, y, w, b)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            j = compute_cost_logistic(X, y, w, b)
            cost_history.append(j)
    return w, b, cost_history


# =============================================================================
# NUMERICAL GRADIENT CHECK — VERIFY GRADIENTS ARE CORRECT
# =============================================================================
print("Gradient check (analytical vs numerical):")
eps = 1e-5
dw_a, db_a = compute_gradient_logistic(X_test, y_test, w_test, b_test)

# Numerical gradient for w[0]
w_p = w_test.copy(); w_p[0] += eps
w_m = w_test.copy(); w_m[0] -= eps
num_dw0 = (compute_cost_logistic(X_test, y_test, w_p, b_test) -
           compute_cost_logistic(X_test, y_test, w_m, b_test)) / (2*eps)

print(f"  dw[0] analytical : {dw_a[0]:.6f}")
print(f"  dw[0] numerical  : {num_dw0:.6f}")
print(f"  Difference       : {abs(dw_a[0] - num_dw0):.2e}  (should be < 1e-5)\n")


# =============================================================================
# LEVEL 6 — REGULARISATION
# =============================================================================
# Problem: if w values become very large, the model OVERFITS.
#   - Fits training data perfectly but fails on new data.
#
# Fix: add a PENALTY for large weights to the cost function.
#
# Regularised cost:
#   J(w,b) = -(1/m) Σ[...] + (λ/2m) Σⱼ wⱼ²
#                                    ↑
#                            L2 (Ridge) penalty
#
# Regularised gradients:
#   ∂J/∂w = (1/m) Xᵀ(f-y) + (λ/m) w
#   ∂J/∂b = (1/m) Σ(f-y)       ← b is NOT regularised
#
# λ (lambda) = regularisation strength
#   λ = 0    → no regularisation (can overfit)
#   λ large  → weights forced to zero (can underfit)
#   λ ~0.01 to 10 is typical

print("=" * 60)
print("LEVEL 6: Regularised logistic regression")
print("=" * 60)

def compute_cost_reg(X, y, w, b, lam):
    m    = X.shape[0]
    base = compute_cost_logistic(X, y, w, b)
    reg  = (lam / (2*m)) * np.sum(w**2)
    return base + reg

def compute_gradient_reg(X, y, w, b, lam):
    m          = X.shape[0]
    dw, db     = compute_gradient_logistic(X, y, w, b)
    dw_reg     = dw + (lam/m) * w    # add regularisation term to dw
    return dw_reg, db                 # db unchanged

def gradient_descent_reg(X, y, w, b, alpha, n_iters, lam):
    cost_history = []
    for i in range(n_iters):
        dw, db = compute_gradient_reg(X, y, w, b, lam)
        w = w - alpha * dw
        b = b - alpha * db
        if i % 100 == 0:
            cost_history.append(compute_cost_reg(X, y, w, b, lam))
    return w, b, cost_history


# =============================================================================
# LEVEL 7 — FULL TRAINING LOOP + EVALUATION
# =============================================================================
print("=" * 60)
print("LEVEL 7: Full training loop on synthetic dataset")
print("=" * 60)

# Generate 2-class 2D dataset
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X_data, y_data = make_classification(
    n_samples=500, n_features=2, n_redundant=0,
    n_informative=2, random_state=42
)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Normalise features (zero mean, unit variance)
mu_tr    = X_tr.mean(axis=0)
sigma_tr = X_tr.std(axis=0)
X_tr_n   = (X_tr - mu_tr) / sigma_tr
X_te_n   = (X_te - mu_tr) / sigma_tr   # use TRAINING stats on test

# Train
n_features   = X_tr_n.shape[1]
w_init       = np.zeros(n_features)
b_init       = 0.0
alpha        = 0.1
n_iters      = 2000
lam          = 0.1

w_trained, b_trained, cost_hist = gradient_descent_reg(
    X_tr_n, y_tr, w_init, b_init, alpha, n_iters, lam
)

# Predict
def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)

y_pred_tr = predict(X_tr_n, w_trained, b_trained)
y_pred_te = predict(X_te_n, w_trained, b_trained)

acc_tr = (y_pred_tr == y_tr).mean()
acc_te = (y_pred_te == y_te).mean()

print(f"Training accuracy : {acc_tr*100:.2f}%")
print(f"Test accuracy     : {acc_te*100:.2f}%")
print(f"Final cost (train): {cost_hist[-1]:.4f}")
print(f"Learned w: {w_trained.round(4)}")
print(f"Learned b: {b_trained:.4f}\n")

# Plot training curve
plt.figure(figsize=(7, 3))
plt.plot(range(0, n_iters, 100), cost_hist)
plt.xlabel("Iteration (×100)")
plt.ylabel("Cost J")
plt.title("Logistic Regression Training Curve")
plt.tight_layout()
plt.savefig("c1_w3_05_training_curve.png")
plt.close()
print("Saved: c1_w3_05_training_curve.png")

# Plot decision boundary on test data
x1_range = np.linspace(X_te_n[:, 0].min()-1, X_te_n[:, 0].max()+1, 300)
x2_bound = -(w_trained[0] * x1_range + b_trained) / (w_trained[1] + 1e-9)

plt.figure(figsize=(6, 5))
plt.scatter(X_te_n[y_te==1, 0], X_te_n[y_te==1, 1], c='red',  label='y=1', alpha=0.6)
plt.scatter(X_te_n[y_te==0, 0], X_te_n[y_te==0, 1], c='blue', label='y=0', alpha=0.6)
plt.plot(x1_range, x2_bound, 'k-', linewidth=2, label='Decision boundary')
plt.title(f"Test set — accuracy {acc_te*100:.1f}%")
plt.legend(); plt.tight_layout()
plt.savefig("c1_w3_06_test_boundary.png")
plt.close()
print("Saved: c1_w3_06_test_boundary.png\n")


# =============================================================================
# LEVEL 8 — MULTI-CLASS: SOFTMAX
# =============================================================================
# Logistic regression is binary (2 classes).
# For K classes, use SOFTMAX:
#
#   softmax(z)ⱼ = e^zⱼ / Σₖ e^zₖ
#
# Output: a vector of K probabilities that sum to 1.
# Pick the class with the highest probability.
#
# Cost: Categorical Cross-Entropy
#   Loss = -Σₖ y_k * log(softmax(z)_k)
# Since y is one-hot (only one class is 1):
#   Loss = -log(softmax(z)_true_class)

print("=" * 60)
print("LEVEL 8: Softmax for multi-class classification")
print("=" * 60)

def softmax(z):
    """
    z : (K,) or (m, K)
    Returns probabilities that sum to 1 (same shape).
    Subtract max for numerical stability (avoids e^huge number).
    """
    z_shift = z - z.max(axis=-1, keepdims=True)
    e       = np.exp(z_shift)
    return e / e.sum(axis=-1, keepdims=True)

# Example: 3-class problem, 1 example
z_example = np.array([2.0, 1.0, 0.1])
probs      = softmax(z_example)
print(f"z (raw scores) : {z_example}")
print(f"softmax(z)     : {probs.round(4)}")
print(f"Sum of probs   : {probs.sum():.4f}  (should be 1.0)")
print(f"Predicted class: {probs.argmax()}\n")

# Batch example (3 examples, 4 classes)
Z_batch = np.array([
    [3.0, 1.0, 0.2, 0.1],
    [0.5, 2.5, 0.3, 1.0],
    [0.1, 0.2, 4.0, 0.3],
])
P_batch = softmax(Z_batch)
print("Batch softmax output (each row sums to 1):")
print(P_batch.round(4))
print(f"Predicted classes: {P_batch.argmax(axis=1)}\n")

# Categorical cross-entropy loss
y_onehot = np.array([0, 1, 2])   # true class indices
loss = -np.log(P_batch[np.arange(3), y_onehot] + 1e-15)
print(f"Cross-entropy loss per example: {loss.round(4)}")
print(f"Mean loss: {loss.mean():.4f}\n")


# =============================================================================
# VERIFY AGAINST SKLEARN
# =============================================================================
print("=" * 60)
print("Verify against sklearn LogisticRegression")
print("=" * 60)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=1/lam, max_iter=1000, random_state=42)
clf.fit(X_tr_n, y_tr)
sk_acc = clf.score(X_te_n, y_te)
print(f"Scratch accuracy : {acc_te*100:.2f}%")
print(f"Sklearn accuracy : {sk_acc*100:.2f}%")
print("(Should be close — small differences due to optimiser)\n")


# =============================================================================
# COMPLETE SUMMARY
# =============================================================================
print("=" * 60)
print("COMPLETE SUMMARY")
print("=" * 60)
print("""
LOGISTIC REGRESSION PIPELINE:

1. FORWARD PASS:
   z    = X @ w + b          (linear combination)
   f    = sigmoid(z)          (probability: 0 to 1)
   ŷ    = (f >= 0.5)          (class prediction)

2. COST (Binary Cross-Entropy):
   J = -(1/m) Σ [y*log(f) + (1-y)*log(1-f)]
   + (λ/2m) Σ w²             (regularisation term)

3. GRADIENTS:
   dw = (1/m) Xᵀ(f-y) + (λ/m)*w
   db = (1/m) Σ(f-y)

4. UPDATE:
   w ← w - α*dw
   b ← b - α*db

KEY DIFFERENCES from Linear Regression:
   Linear  : f(x) = wx + b          (output: any real number)
   Logistic: f(x) = sigmoid(wx + b) (output: 0 to 1)
   Cost    : MSE    → Cross-Entropy
   Gradient: same formula, different f
""")


