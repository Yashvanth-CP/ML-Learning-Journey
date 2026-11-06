"""Logistic Regression with Gradient Descent (NumPy only)
Flow: DATA -> MODEL (w, b) -> PREDICTION (sigmoid) -> COST -> GRADIENT -> UPDATE
"""
import numpy as np
import matplotlib.pyplot as plt

# 1. Data: two groups of points (class 0 and class 1)
np.random.seed(0)
m = 200                                                  # number of examples
X0 = np.random.randn(m // 2, 2) + np.array([-2, -2])     # class 0 points
X1 = np.random.randn(m // 2, 2) + np.array([2, 2])       # class 1 points
X = np.vstack([X0, X1])                                  # shape (200, 2)
y = np.hstack([np.zeros(m // 2), np.ones(m // 2)])       # shape (200,)
X = (X - X.mean(axis=0)) / X.std(axis=0)                 # scale features


# 2. Sigmoid: squeezes any number into (0, 1)  ->  1 / (1 + e^-z)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 3. Cost (log loss): J = -(1/m) * sum(y*log(f) + (1-y)*log(1-f))
def compute_cost(X, y, w, b):
    f = np.clip(sigmoid(X @ w + b), 1e-15, 1 - 1e-15)    # avoid log(0)
    return -np.mean(y * np.log(f) + (1 - y) * np.log(1 - f))


# 4. Gradients: dJ/dw = (1/m) * X^T (f - y),  dJ/db = (1/m) * sum(f - y)
def compute_gradient(X, y, w, b):
    error = sigmoid(X @ w + b) - y                       # f - y
    return (X.T @ error) / len(y), np.mean(error)


# 5. Gradient descent: w = w - alpha * dJ/dw,  b = b - alpha * dJ/db
def gradient_descent(X, y, w, b, alpha, iterations):
    history = []
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw                            # move against slope
        b = b - alpha * dj_db
        history.append(compute_cost(X, y, w, b))
        if i % 100 == 0:
            print(f"Iteration {i:4d} | Cost = {history[-1]:.4f}")
    return w, b, history


# 6. Predict class labels (0 or 1)
def predict(X, w, b, threshold=0.5):
    return (sigmoid(X @ w + b) >= threshold).astype(int)


# 7. Train
w, b, history = gradient_descent(X, y, np.zeros(2), 0.0, alpha=0.1, iterations=1000)
accuracy = np.mean(predict(X, w, b) == y) * 100
print(f"\nLearned w = {w}, b = {b:.4f}")
print(f"Training accuracy = {accuracy:.2f}%")

# 8. Plots: cost curve + decision boundary
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history)
ax[0].set_title("Cost vs Iterations")
ax[0].set_xlabel("Iteration")
ax[0].set_ylabel("Cost J(w, b)")

ax[1].scatter(X[y == 0, 0], X[y == 0, 1], label="Class 0")
ax[1].scatter(X[y == 1, 0], X[y == 1, 1], label="Class 1")
x1 = np.array([X[:, 0].min(), X[:, 0].max()])
ax[1].plot(x1, -(w[0] * x1 + b) / w[1], "k--", label="Decision boundary")
ax[1].set_title("Decision Boundary")
ax[1].legend()
plt.tight_layout()
plt.savefig("logistic_result.png")
plt.show()