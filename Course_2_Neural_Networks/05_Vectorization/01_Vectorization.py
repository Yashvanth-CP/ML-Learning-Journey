
"""
COURSE 2 TOPIC 1: VECTORIZATION
==============================

Why vectorization is CRITICAL for neural networks
No loops = 100x faster!
"""

import numpy as np
import time

print("="*70)
print("VECTORIZATION: The Most Important Optimization")
print("="*70)


print("\n1. WHY VECTORIZATION?")
print("-"*70)

print("""
Problem: Naive approach (loops) is SLOW

Example: Forward pass with 1000 samples, 128 features

Non-vectorized (loops):
    for i in range(1000):  # for each sample
        for j in range(128):  # for each feature
            z[i] += X[i,j] * w[j]
    → VERY SLOW! Nested loops in Python

Vectorized (matrix multiplication):
    Z = np.dot(X, w)
    → INSTANT! Uses optimized C code

Speed difference: 100-1000x faster!
""")


print("\n2. BENCHMARK: LOOPS VS VECTORIZATION")
print("-"*70)

# Setup
m = 1000000  # 1 million samples
n = 100      # 100 features

X = np.random.randn(m, n)
w = np.random.randn(n, 1)

# Non-vectorized (slow - DON'T RUN, just show example)
print("\nNon-vectorized approach (pseudocode):")
print("""
def forward_nonvectorized(X, w):
    m = X.shape[0]
    z = np.zeros((m, 1))
    for i in range(m):
        for j in range(len(w)):
            z[i] += X[i,j] * w[j]
    return z

Time: ~10-20 seconds for 1M samples! ❌
""")

# Vectorized (fast)
print("\nVectorized approach:")
print("z = np.dot(X, w)")

start = time.time()
z = np.dot(X, w)
vectorized_time = time.time() - start

print(f"Time: {vectorized_time*1000:.2f} ms for {m:,} samples ✓")
print(f"Speedup: ~{10000 / (vectorized_time*1000):.0f}x faster!")


print("\n3. VECTORIZATION IN NEURAL NETWORKS")
print("-"*70)

print("""
Layer 1 Forward Pass:

Non-vectorized (WRONG):
    for i in range(m):  # each sample
        for j in range(n_hidden):  # each neuron
            z = 0
            for k in range(n_input):
                z += X[i,k] * W[k,j]
            z += b[j]
            A[i,j] = relu(z)

Vectorized (CORRECT):
    Z1 = np.dot(X, W1) + b1  # One line!
    A1 = np.maximum(0, Z1)   # Another line!

Dimensions:
    X: (m, n_input)
    W1: (n_input, n_hidden)
    Z1: (m, n_hidden)
    A1: (m, n_hidden)
""")


print("\n4. PYTHON LOOPS = SLOW")
print("-"*70)

print("""
Why are Python loops slow?

Python overhead per iteration:
    - Type checking
    - Memory allocation
    - Function call overhead
    - Interpreter overhead

NumPy C code:
    - No type checking (already typed)
    - Pre-allocated memory
    - Direct CPU operations
    - Optimized for CPU cache

Result: 100-1000x speedup!
""")


print("\n5. COMMON VECTORIZATION MISTAKES")
print("-"*70)

print("""
❌ Mistake 1: Using loops when not needed
    for i in range(m):
        z[i] = x[i] * w[i]
    
✓ Correct:
    z = x * w  (element-wise)

❌ Mistake 2: Reshaping unnecessarily
    for i in range(m):
        y[i] = model(X[i]).reshape(1, -1)
    
✓ Correct:
    Y = model(X)  # Batch processing

❌ Mistake 3: Not using matrix operations
    for i in range(m):
        pred[i] = np.dot(X[i], w) + b
    
✓ Correct:
    pred = np.dot(X, w) + b

❌ Mistake 4: Using Python lists
    z = []
    for i in range(m):
        z.append(x[i] * w[i])
    
✓ Correct:
    z = np.array(x) * np.array(w)
""")


print("\n6. VECTORIZATION PATTERNS")
print("-"*70)

print("""
Pattern 1: Element-wise operations
    y = x * w          (faster than loop)
    z = a + b          (faster than loop)

Pattern 2: Matrix multiplication
    Z = X @ W          (crucial for forward pass)
    dW = X.T @ dZ      (crucial for backprop)

Pattern 3: Reduction operations
    mean = np.mean(X, axis=0)
    sum_rows = np.sum(X, axis=1, keepdims=True)

Pattern 4: Broadcasting
    X + b              (b broadcasts to match X shape)
    X * 2              (scalar broadcasts)
""")


print("\n7. BROADCASTING: IMPLICIT VECTORIZATION")
print("-"*70)

print("\nBroadcasting rules:")
print("  (m, n) + (1, n) → (m, n)  # Bias addition")
print("  (m, n) + (n,)   → (m, n)  # Bias addition (auto reshape)")
print("  (m, 1) * (1, n) → (m, n)  # Outer product")

# Example
X = np.array([[1, 2], [3, 4], [5, 6]])  # (3, 2)
b = np.array([10, 20])                   # (2,)

result = X + b  # b broadcasts to (3, 2)

print(f"\nX shape: {X.shape}")
print(f"b shape: {b.shape}")
print(f"X + b shape: {result.shape}")
print(f"Result:\n{result}")


print("\n8. ACTIVATION FUNCTIONS: VECTORIZED")
print("-"*70)

print("""
Non-vectorized:
    for i in range(m):
        for j in range(n):
            a[i,j] = max(0, z[i,j])

Vectorized:
    A = np.maximum(0, Z)  ← This is fast!

Same for sigmoid:
    A = 1 / (1 + np.exp(-Z))

Same for softmax:
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
""")


print("\n9. GRADIENT COMPUTATION: VECTORIZED")
print("-"*70)

print("""
Loss gradient for m samples:

Non-vectorized (WRONG):
    for i in range(m):
        dw[j] = (y_pred[i] - y_true[i]) * x[i,j]

Vectorized (CORRECT):
    dW = (1/m) * X.T @ dZ
    
This ONE line replaces the double loop!

Dimensions:
    X: (m, n)
    dZ: (m, k)
    X.T @ dZ: (n, k) ← Your gradient!
""")


print("\n10. PRACTICAL IMPACT")
print("-"*70)

# Demonstrate speed difference
print("\nForward pass speed comparison:")

X_batch = np.random.randn(32, 784)  # MNIST batch
W = np.random.randn(784, 128)

# Vectorized
start = time.time()
for _ in range(1000):
    Z = np.dot(X_batch, W)
vec_time = time.time() - start

print(f"  Vectorized (1000 iterations): {vec_time*1000:.2f} ms")
print(f"  Per-forward: {vec_time*1000/1000:.3f} ms")

print(f"\nWith vectorization:")
print(f"  Train 60K MNIST samples: ~seconds")
print(f"  Without vectorization: ~minutes")


print("\n11. KEY TAKEAWAY")
print("-"*70)

print("""
✓ ALWAYS vectorize!
✓ Use matrix operations (np.dot, @)
✓ Use element-wise operations (*, +, -)
✓ Use NumPy/TensorFlow built-ins
✓ Avoid Python loops at all costs

Speed matters:
  - Local testing: difference between 1ms and 10ms
  - Training large models: difference between hours and days
  - Deployment: difference between fast inference and timeout

Rule: If you wrote a loop in a neural network, ask: 
  "Can I use a matrix operation instead?"
  
Answer: Usually YES!
""")

print("="*70)
print("✓ VECTORIZATION = FOUNDATION OF DEEP LEARNING")
print("="*70)