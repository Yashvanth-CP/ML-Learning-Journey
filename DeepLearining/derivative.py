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


"""Non-linear activation functions (NumPy) with a plot of each one."""
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):                          # output in (0, 1)
    return 1 / (1 + np.exp(-z))

def tanh(z):                             # output in (-1, 1), zero-centered
    return np.tanh(z)

def relu(z):                             # max(0, z)
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):           # small slope for negative z
    return np.where(z > 0, z, alpha * z)

def elu(z, alpha=1.0):                   # smooth curve for negative z
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

def softplus(z):                         # smooth version of ReLU
    return np.log(1 + np.exp(z))

def gelu(z):                             # used in Transformers (tanh approx.)
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))

def swish(z):                            # z * sigmoid(z)
    return z * sigmoid(z)

def softmax(z):                          # turns a vector into probabilities
    e = np.exp(z - np.max(z))            # subtract max for numerical stability
    return e / e.sum()


# ---------- Test ----------
z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
for f in [sigmoid, tanh, relu, leaky_relu, elu, softplus, gelu, swish]:
    print(f"{f.__name__:11s}:", np.round(f(z), 3))
print("softmax    :", np.round(softmax(z), 3), "| sum =", softmax(z).sum())

# ---------- Plot ----------
x = np.linspace(-5, 5, 400)
funcs = [sigmoid, tanh, relu, leaky_relu, elu, softplus, gelu, swish]
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, f in zip(axes.ravel(), funcs):
    ax.plot(x, f(x))
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_title(f.__name__)
    ax.set_ylim(-2, 5)
plt.tight_layout()
plt.savefig("/home/claude/activations.png")