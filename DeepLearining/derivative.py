import numpy as np

# ---------- Sigmoid ----------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    a = sigmoid(z)
    return a * (1 - a)          # g'(z) = a(1 - a)

# ---------- Tanh ----------
def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    a = np.tanh(z)
    return 1 - a ** 2           # g'(z) = 1 - a^2

# ---------- ReLU ----------
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)   # 1 where z > 0, else 0

# ---------- Leaky ReLU ----------
def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

def leaky_relu_derivative(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)

# ---------- Test ----------
z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
print("sigmoid'  :", sigmoid_derivative(z))
print("tanh'     :", tanh_derivative(z))
print("relu'     :", relu_derivative(z))
print("leaky'    :", leaky_relu_derivative(z))


z = X @ W + b                    # forward: linear part
a = sigmoid(z)                   # forward: activation

# backward: da is the gradient coming from the next layer
dz = da * sigmoid_derivative(z)  # chain rule: dL/dz = dL/da * g'(z)
dW = X.T @ dz / m                # gradient for weights
db = np.sum(dz, axis=0) / m      # gradient for bias

def numerical_derivative(g, z, eps=1e-5):
    return (g(z + eps) - g(z - eps)) / (2 * eps)

print(np.allclose(sigmoid_derivative(z), numerical_derivative(sigmoid, z)))  # True
