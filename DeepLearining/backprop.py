"""
Backpropagation from scratch (NumPy only)
Network: 2 inputs -> 4 hidden neurons (sigmoid) -> 1 output (sigmoid)
Task: learn XOR
Loss: Mean Squared Error  L = (1/m) * sum((a2 - y)^2)
"""
import numpy as np
 
np.random.seed(42)
 
# ---------- DATA ----------
# X shape: (m, 2)   y shape: (m, 1)   m = number of examples
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
y = np.array([[0], [1], [1], [0]], dtype=float)
m = X.shape[0]
 
# ---------- PARAMETERS ----------
# W1: (2, 4)  b1: (1, 4)  -> input to hidden
# W2: (4, 1)  b2: (1, 1)  -> hidden to output
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))
 
learning_rate = 1.0
epochs = 5000
 
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
 
 
def sigmoid_derivative(a):
    # a = sigmoid(z) already, so derivative = a * (1 - a)
    return a * (1 - a)
 
 
for epoch in range(epochs):
    # ---------- FORWARD PROPAGATION ----------
    z1 = X @ W1 + b1          # (m, 4)
    a1 = sigmoid(z1)          # hidden activations
    z2 = a1 @ W2 + b2         # (m, 1)
    a2 = sigmoid(z2)          # prediction y_hat
 
    # ---------- COST ----------
    loss = np.mean((a2 - y) ** 2)
 
    # ---------- BACKWARD PROPAGATION (chain rule) ----------
    # dL/da2 = 2 * (a2 - y) / m
    dL_da2 = 2 * (a2 - y) / m
    # dL/dz2 = dL/da2 * da2/dz2
    dz2 = dL_da2 * sigmoid_derivative(a2)          # (m, 1)
 
    # Output layer gradients
    dW2 = a1.T @ dz2                               # (4, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True)       # (1, 1)
 
    # Send the error back to the hidden layer
    da1 = dz2 @ W2.T                               # (m, 4)
    dz1 = da1 * sigmoid_derivative(a1)             # (m, 4)
 
    # Hidden layer gradients
    dW1 = X.T @ dz1                                # (2, 4)
    db1 = np.sum(dz1, axis=0, keepdims=True)       # (1, 4)
 
    # ---------- GRADIENT DESCENT UPDATE ----------
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
 
    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d} | Loss: {loss:.6f}")
 
# ---------- RESULT ----------
print("\nFinal predictions:")
final = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
for inp, pred, target in zip(X, final, y):
    print(f"Input {inp} -> {pred[0]:.4f} (target {int(target[0])})")
 