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