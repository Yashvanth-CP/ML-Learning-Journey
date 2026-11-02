import numpy as np
import time

a = np.random.randn(1_000_000)
b = np.random.randn(1_000_000)

# SLOW — Python loop
start = time.time()
result = 0
for i in range(len(a)):
    result += a[i] * b[i]
print(f"Loop  : {time.time()-start:.3f}s  result={result:.4f}")

# FAST — vectorized
start = time.time()
result = np.dot(a, b)
print(f"NumPy : {time.time()-start:.3f}s  result={result:.4f}")
# NumPy is typically 100x–500x faster

# Shape (3,) + scalar → works
a = np.array([1, 2, 3])
print(a + 10)       # [11 12 13]

# Shape (3, 1) + shape (3,) → broadcasts
A = np.array([[1], [2], [3]])   # shape (3, 1)
b = np.array([10, 20, 30])      # shape (3,)
print(A + b)
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]

# Mean normalisation (used in ML preprocessing)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)

mean = X.mean(axis=0)    # mean of each COLUMN → shape (3,)
X_centred = X - mean     # broadcasts: subtracts mean from every row
print(X_centred)
# [[-3. -3. -3.]
#  [ 0.  0.  0.]
#  [ 3.  3.  3.]]

# DOT PRODUCT
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a, b))          # 1*4 + 2*5 + 3*6 = 32

# MATRIX MULTIPLY  @  operator
W = np.array([[1, 2],
              [3, 4],
              [5, 6]])   # shape (3, 2)
x = np.array([1, 1])    # shape (2,)

print(W @ x)             # [3, 7, 11] — shape (3,)

# In ML: predictions = W @ X.T + b
# Instead of looping over every example!
m = 5   # 5 training examples
n = 3   # 3 features
X = np.random.randn(m, n)   # (5, 3)
w = np.random.randn(n)      # (3,)
b = 0.5

y_hat = X @ w + b            # (5,) — all 5 predictions at once
print(y_hat.shape)           # (5,)

X = np.array([[1, 2, 3],
              [4, 5, 6]])   # shape (2, 3)

print(X.sum())              # 21  — sum of everything
print(X.sum(axis=0))        # [5 7 9]  — sum DOWN each column
print(X.sum(axis=1))        # [6 15]   — sum ACROSS each row

print(X.mean(axis=0))       # [2.5 3.5 4.5]
print(X.max(axis=1))        # [3 6]
print(X.argmax(axis=1))     # [2 2] — index of max in each row

# LINEAR REGRESSION COST (vectorized)
# J = (1/2m) * sum((y_hat - y)^2)

def cost_vectorized(X, y, w, b):
    m     = X.shape[0]
    y_hat = X @ w + b          # (m,) — all predictions at once
    diff  = y_hat - y           # (m,)
    J     = (1/(2*m)) * np.dot(diff, diff)
    return J

# GRADIENT (vectorized)
def gradient_vectorized(X, y, w, b):
    m     = X.shape[0]
    y_hat = X @ w + b
    diff  = y_hat - y
    dw    = (1/m) * X.T @ diff  # (n,) — gradient for each weight
    db    = (1/m) * diff.sum()  # scalar
    return dw, db

# SIGMOID (vectorized — applies to every element)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.array([-2, -1, 0, 1, 2])
print(sigmoid(z))   # [0.119 0.269 0.5 0.731 0.881]

# SOFTMAX (for multi-class — vectorized)
def softmax(z):
    e = np.exp(z - z.max())   # subtract max for numerical stability
    return e / e.sum()

print(softmax(np.array([1.0, 2.0, 3.0])))  # [0.09 0.245 0.665]