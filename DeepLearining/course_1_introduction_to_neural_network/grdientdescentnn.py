import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])   # shape (4, 2)
y = np.array([[0], [1], [1], [0]])               # shape (4, 1)
m = X.shape[0]


np.random.seed(42)
W1 = np.random.randn(2, 4) * 0.5   # input -> hidden weights
b1 = np.zeros((1, 4))              # hidden bias
W2 = np.random.randn(4, 1) * 0.5   # hidden -> output weights
b2 = np.zeros((1, 1))              # output bias
 
learning_rate = 0.5
epochs = 5000
 
 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

for epoch in range(epochs):
 
    # ----- Forward pass: DATA -> MODEL -> PREDICTION -----
    z1 = X @ W1 + b1          # hidden layer, before activation
    a1 = np.tanh(z1)          # hidden layer, after activation
    z2 = a1 @ W2 + b2         # output layer, before activation
    a2 = sigmoid(z2)          # final prediction (0 to 1)
 
    # ----- Cost: how wrong are we? (binary cross-entropy) -----
    cost = -np.mean(y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))
 
    # ----- Backward pass: find the gradients -----
    dz2 = a2 - y                          # error at output
    dW2 = (a1.T @ dz2) / m                # gradient for W2
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
 
    da1 = dz2 @ W2.T                      # send error back to hidden layer
    dz1 = da1 * (1 - a1 ** 2)             # tanh derivative
    dW1 = (X.T @ dz1) / m                 # gradient for W1
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
 
    # ----- Gradient descent update: move opposite to the gradient -----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
 
    if epoch % 1000 == 0:
        print(f"Epoch {epoch:4d} | Cost: {cost:.4f}")
 
# ---------- 4. PREDICTION ----------
a1 = np.tanh(X @ W1 + b1)
pred = sigmoid(a1 @ W2 + b2)
print("\nPredictions (probabilities):")
print(np.round(pred, 3))
print("Predicted classes:", (pred > 0.5).astype(int).ravel())
print("Actual classes:   ", y.ravel())