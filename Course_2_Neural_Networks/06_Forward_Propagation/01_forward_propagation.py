"""
COURSE 2: FORWARD PROPAGATION - COMPLETE DEEP DIVE
==================================================

How data flows through a neural network
From input → hidden layers → output
The FOUNDATION of everything!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*80)
print("FORWARD PROPAGATION: How Neural Networks Make Predictions")
print("="*80)

# ============================================================================
# PART 1: SIMPLE DEFINITION
# ============================================================================

print("\n" + "="*80)
print("PART 1: WHAT IS FORWARD PROPAGATION?")
print("="*80)

print("""
Simple Definition:
  Forward propagation = Process of passing input data through the network
                       to get output (prediction)

Analogy:
  Think of a factory assembly line:
  
  Raw material (Input X)
    ↓
  Machine 1 (Layer 1) processes it
    ↓
  Machine 2 (Layer 2) processes it
    ↓
  Final product (Output Y)

Why "forward"?
  Data flows forward: Input → Layer1 → Layer2 → ... → Output
  (Later we'll do backward propagation for learning)

Simple formula:
  Forward pass: Y = f(f(f(X)))
  
  Where each f() is: multiply by weights + add bias + apply activation
""")

# ============================================================================
# PART 2: REAL-WORLD EXAMPLE
# ============================================================================

print("\n" + "="*80)
print("PART 2: REAL-WORLD EXAMPLE - HOUSE PRICE PREDICTION")
print("="*80)

print("""
Imagine predicting house price from:
  - Size (sq ft)
  - Bedrooms
  - Age (years)

Neural Network:
  Input Layer:     [Size, Bedrooms, Age]  ← 3 features
  Hidden Layer 1:  [neuron1, neuron2]     ← 2 neurons
  Hidden Layer 2:  [neuron3]              ← 1 neuron
  Output Layer:    [Price]                ← 1 prediction

Forward Pass:
  Step 1: Inputs → Hidden Layer 1
          Each hidden neuron computes: weighted sum + bias
  
  Step 2: Hidden Layer 1 → Hidden Layer 2
          Takes previous outputs as inputs
          Computes again: weighted sum + bias
  
  Step 3: Hidden Layer 2 → Output
          Final prediction: price estimate
""")

# ============================================================================
# PART 3: SINGLE NEURON FORWARD PASS
# ============================================================================

print("\n" + "="*80)
print("PART 3: SINGLE NEURON - SIMPLEST FORWARD PASS")
print("="*80)

print("\n1. Single input, single neuron:")
print("-"*80)

print("""
Components:
  ┌─────────────────────────────────────────────┐
  │  Input: x = 2.0                             │
  │  Weight: w = 0.5                            │
  │  Bias: b = 0.3                              │
  │  Activation: ReLU                           │
  └─────────────────────────────────────────────┘

Step 1: Weighted sum (Z)
  z = x * w + b
  z = 2.0 * 0.5 + 0.3
  z = 1.0 + 0.3
  z = 1.3

Step 2: Apply activation function (ReLU)
  a = ReLU(z) = max(0, z)
  a = max(0, 1.3)
  a = 1.3  ← This is the output!

Intuition:
  - Input (x=2) gets "weighted" (multiplied by 0.5)
  - Then shifted (added 0.3)
  - Then passed through ReLU to ensure non-linearity
""")

# Code example
x = 2.0
w = 0.5
b = 0.3

z = x * w + b
a = max(0, z)

print(f"\nCode execution:")
print(f"  x = {x}")
print(f"  w = {w}")
print(f"  b = {b}")
print(f"  z = x*w + b = {x}*{w} + {b} = {z}")
print(f"  a = ReLU({z}) = {a}")
print(f"  Output: {a}")

# ============================================================================
# PART 4: MULTIPLE INPUTS, SINGLE NEURON
# ============================================================================

print("\n" + "="*80)
print("PART 4: MULTIPLE INPUTS → SINGLE NEURON")
print("="*80)

print("\n2. Multiple inputs (vectorized):")
print("-"*80)

print("""
Inputs (x1, x2, x3):
  x1 = 2.0   (size)
  x2 = 3.0   (bedrooms)
  x3 = 5.0   (age)

Weights (w1, w2, w3):
  w1 = 0.5   (weight for size)
  w2 = 0.2   (weight for bedrooms)
  w3 = 0.1   (weight for age)

Bias:
  b = 0.3

Computation (non-vectorized):
  z = x1*w1 + x2*w2 + x3*w3 + b
  z = 2.0*0.5 + 3.0*0.2 + 5.0*0.1 + 0.3
  z = 1.0 + 0.6 + 0.5 + 0.3
  z = 2.4

Vectorized (matrix form):
  x = [2.0, 3.0, 5.0]       ← Input vector (1×3)
  w = [0.5, 0.2, 0.1]       ← Weight vector (3×1)
  
  z = x · w + b              ← Dot product!
  z = 2.4 (same result!)

Activation:
  a = ReLU(2.4) = 2.4
""")

# Vectorized code
x = np.array([2.0, 3.0, 5.0])
w = np.array([0.5, 0.2, 0.1])
b = 0.3

z = np.dot(x, w) + b
a = np.maximum(0, z)

print(f"\nCode (vectorized):")
print(f"  x = {x}")
print(f"  w = {w}")
print(f"  b = {b}")
print(f"  z = np.dot(x, w) + b = {z}")
print(f"  a = np.maximum(0, {z}) = {a}")

print("\n✓ KEY INSIGHT: Vectorization makes it fast!")
print("  One dot product instead of three multiplications")

# ============================================================================
# PART 5: MULTIPLE NEURONS IN ONE LAYER
# ============================================================================

print("\n" + "="*80)
print("PART 5: MULTIPLE NEURONS IN ONE LAYER")
print("="*80)

print("\n3. Multiple inputs → Multiple neurons:")
print("-"*80)

print("""
Setup:
  3 Inputs: [size, bedrooms, age]
  2 Hidden Neurons: [neuron1, neuron2]

Neuron 1 computation:
  z1 = x1*w11 + x2*w21 + x3*w31 + b1
  z1 = 2*0.5 + 3*0.2 + 5*0.1 + 0.3 = 2.4
  a1 = ReLU(2.4) = 2.4

Neuron 2 computation:
  z2 = x1*w12 + x2*w22 + x3*w32 + b2
  z2 = 2*0.3 + 3*0.4 + 5*0.2 + 0.1 = 2.9
  a2 = ReLU(2.9) = 2.9

Output of layer: [2.4, 2.9]

Matrix form (VECTORIZED):
  X = [2.0, 3.0, 5.0]           ← (1, 3) one sample, 3 features
  
  W = [[0.5, 0.3],              ← (3, 2) weights
       [0.2, 0.4],
       [0.1, 0.2]]
  
  b = [0.3, 0.1]                ← (2,) bias
  
  Z = X @ W + b                 ← Matrix multiplication!
  Z = [[2.4, 2.9]]              ← (1, 2) output
  
  A = ReLU(Z)
  A = [[2.4, 2.9]]

Dimensions:
  (1, 3) @ (3, 2) = (1, 2) ✓
  
  General: (m, n) @ (n, k) = (m, k)
           samples × features @ features × neurons = samples × neurons
""")

# Code example
X = np.array([[2.0, 3.0, 5.0]])  # (1, 3) - one sample
W = np.array([
    [0.5, 0.3],   # weights for both neurons from feature 1
    [0.2, 0.4],   # weights for both neurons from feature 2
    [0.1, 0.2]    # weights for both neurons from feature 3
])
b = np.array([0.3, 0.1])

Z = np.dot(X, W) + b
A = np.maximum(0, Z)

print(f"\nCode:")
print(f"  X shape: {X.shape}")
print(f"  W shape: {W.shape}")
print(f"  b shape: {b.shape}")
print(f"  Z = X @ W + b")
print(f"  Z shape: {Z.shape}")
print(f"  Z values: {Z}")
print(f"  A = ReLU(Z)")
print(f"  A values: {A}")

# ============================================================================
# PART 6: MULTIPLE SAMPLES (BATCHES)
# ============================================================================

print("\n" + "="*80)
print("PART 6: MULTIPLE SAMPLES (BATCHES) - PRACTICAL")
print("="*80)

print("\n4. Batch processing (m samples):")
print("-"*80)

print("""
Real scenario: Training with 32 samples at once

Setup:
  Batch size: m = 32 (32 house samples)
  Features: n = 3 (size, bedrooms, age)
  Hidden neurons: 2

Data shape:
  X = (32, 3)   ← 32 samples, 3 features each
  W = (3, 2)    ← 3 features, 2 hidden neurons
  b = (2,)      ← 2 biases
  
Computation:
  Z = X @ W + b        ← One line!
  Z = (32, 3) @ (3, 2) = (32, 2)
  
  Result: 32 × 2 matrix
  Each row: outputs for one sample
  Each column: output of one neuron

Example result:
  Z[0] = [2.4, 2.9]    ← Sample 1
  Z[1] = [1.8, 2.1]    ← Sample 2
  ...
  Z[31] = [3.2, 2.8]   ← Sample 32

Activation:
  A = ReLU(Z)   ← Applied element-wise!
  A = (32, 2)   ← Same shape as Z

Vectorization benefit:
  This processes 32 samples in parallel!
  Much faster than loop of 32 iterations.
""")

# Code example with batch
np.random.seed(42)
X_batch = np.random.randn(32, 3)  # 32 samples, 3 features
W_hidden = np.array([
    [0.5, 0.3],
    [0.2, 0.4],
    [0.1, 0.2]
])
b_hidden = np.array([0.3, 0.1])

Z_hidden = np.dot(X_batch, W_hidden) + b_hidden
A_hidden = np.maximum(0, Z_hidden)

print(f"\nCode:")
print(f"  X_batch shape: {X_batch.shape}  (32 samples, 3 features)")
print(f"  W_hidden shape: {W_hidden.shape}  (3 features, 2 neurons)")
print(f"  b_hidden shape: {b_hidden.shape}  (2 neurons)")
print(f"  Z_hidden = X_batch @ W_hidden + b_hidden")
print(f"  Z_hidden shape: {Z_hidden.shape}  (32 samples, 2 outputs)")
print(f"  A_hidden shape: {A_hidden.shape}")
print(f"\nFirst 3 samples Z values:\n{Z_hidden[:3]}")
print(f"\nFirst 3 samples A values (after ReLU):\n{A_hidden[:3]}")

# ============================================================================
# PART 7: COMPLETE NETWORK FORWARD PASS
# ============================================================================

print("\n" + "="*80)
print("PART 7: COMPLETE 3-LAYER NETWORK")
print("="*80)

print("\n5. Full forward pass through entire network:")
print("-"*80)

print("""
Architecture:
  Input:  28×28 MNIST image = 784 features
  
  Hidden Layer 1:  128 neurons, ReLU
  Hidden Layer 2:  64 neurons, ReLU
  Output Layer:    10 neurons (digits 0-9), Softmax

Forward Pass:

Step 1: Input → Hidden Layer 1
  Z1 = X @ W1 + b1
  A1 = ReLU(Z1)
  
  Dimensions:
    X:   (m, 784)
    W1:  (784, 128)
    b1:  (128,)
    Z1:  (m, 128)
    A1:  (m, 128)

Step 2: Hidden Layer 1 → Hidden Layer 2
  Z2 = A1 @ W2 + b2
  A2 = ReLU(Z2)
  
  Dimensions:
    A1:  (m, 128)    ← Output from Layer 1 becomes input
    W2:  (128, 64)
    b2:  (64,)
    Z2:  (m, 64)
    A2:  (m, 64)

Step 3: Hidden Layer 2 → Output Layer
  Z3 = A2 @ W3 + b3
  A3 = Softmax(Z3)
  
  Dimensions:
    A2:  (m, 64)
    W3:  (64, 10)
    b3:  (10,)
    Z3:  (m, 10)
    A3:  (m, 10)    ← Probabilities for each digit

Output: A3 ← Predicted probabilities for each sample!

Flow summary:
  X (784) → Z1 (128) → A1 (128) →
  Z2 (64) → A2 (64) → Z3 (10) → A3 (10)
""")

# Complete network forward pass
m = 32  # batch size
np.random.seed(42)

# Input
X = np.random.randn(m, 784)

# Layer 1: 784 → 128
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros(128)
Z1 = np.dot(X, W1) + b1
A1 = np.maximum(0, Z1)  # ReLU

# Layer 2: 128 → 64
W2 = np.random.randn(128, 64) * 0.01
b2 = np.zeros(64)
Z2 = np.dot(A1, W2) + b2
A2 = np.maximum(0, Z2)  # ReLU

# Layer 3: 64 → 10
W3 = np.random.randn(64, 10) * 0.01
b3 = np.zeros(10)
Z3 = np.dot(A2, W3) + b3

# Softmax
exp_Z3 = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
A3 = exp_Z3 / np.sum(exp_Z3, axis=1, keepdims=True)

print(f"\nCode execution:")
print(f"  Input X shape: {X.shape}")
print(f"\n  Layer 1:")
print(f"    W1 shape: {W1.shape}")
print(f"    Z1 shape: {Z1.shape}")
print(f"    A1 shape: {A1.shape}")
print(f"\n  Layer 2:")
print(f"    W2 shape: {W2.shape}")
print(f"    Z2 shape: {Z2.shape}")
print(f"    A2 shape: {A2.shape}")
print(f"\n  Layer 3 (Output):")
print(f"    W3 shape: {W3.shape}")
print(f"    Z3 shape: {Z3.shape}")
print(f"    A3 shape: {A3.shape}")
print(f"\nSample output (first image):")
print(f"  Probabilities: {A3[0]}")
print(f"  Predicted digit: {np.argmax(A3[0])}")
print(f"  Confidence: {np.max(A3[0]):.2%}")

# ============================================================================
# PART 8: ACTIVATION FUNCTIONS
# ============================================================================

print("\n" + "="*80)
print("PART 8: WHY ACTIVATION FUNCTIONS?")
print("="*80)

print("\n6. Understanding activation functions:")
print("-"*80)

print("""
Without activation (linear only):
  Z1 = X @ W1 + b1
  Z2 = Z1 @ W2 + b2
  Z3 = Z2 @ W3 + b3
  
  This is just: Z3 = X @ (W1 @ W2 @ W3) + ...
  
  Multiple linear layers = One linear layer!
  Problem: Can't learn non-linear patterns!

With activation (non-linear):
  Z1 = X @ W1 + b1
  A1 = ReLU(Z1)          ← Non-linear!
  Z2 = A1 @ W2 + b2
  A2 = ReLU(Z2)          ← Non-linear!
  Z3 = A2 @ W3 + b3
  
  Now we can learn curves, circles, complex patterns!

Common activation functions:

1. ReLU (Rectified Linear Unit)
   f(z) = max(0, z)
   Used in: Hidden layers
   Why: Fast, no vanishing gradient
   
2. Sigmoid
   f(z) = 1 / (1 + e^(-z))
   Output: 0-1
   Used in: Binary classification
   
3. Tanh
   f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
   Output: -1 to 1
   Used in: Hidden layers (less common now)
   
4. Softmax
   Used in: Output layer for multi-class
   Outputs: Probabilities (sum to 1)
""")

# Visualize activation functions
x_range = np.linspace(-5, 5, 100)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ReLU
ax = axes[0, 0]
y_relu = np.maximum(0, x_range)
ax.plot(x_range, y_relu, linewidth=3, color='red')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('ReLU: max(0, z)', fontweight='bold', fontsize=12)
ax.set_ylabel('Output')
ax.grid(True, alpha=0.3)

# Sigmoid
ax = axes[0, 1]
y_sigmoid = 1 / (1 + np.exp(-x_range))
ax.plot(x_range, y_sigmoid, linewidth=3, color='blue')
ax.axhline(y=0.5, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('Sigmoid: 1 / (1 + e^(-z))', fontweight='bold', fontsize=12)
ax.set_ylabel('Output')
ax.set_ylim([-0.1, 1.1])
ax.grid(True, alpha=0.3)

# Tanh
ax = axes[1, 0]
y_tanh = np.tanh(x_range)
ax.plot(x_range, y_tanh, linewidth=3, color='green')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('Tanh: (e^z - e^(-z)) / (e^z + e^(-z))', fontweight='bold', fontsize=12)
ax.set_xlabel('Input (z)')
ax.set_ylabel('Output')
ax.set_ylim([-1.1, 1.1])
ax.grid(True, alpha=0.3)

# Comparison
ax = axes[1, 1]
ax.plot(x_range, y_relu, linewidth=2, label='ReLU', color='red')
ax.plot(x_range, y_sigmoid, linewidth=2, label='Sigmoid', color='blue')
ax.plot(x_range, y_tanh, linewidth=2, label='Tanh', color='green')
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
ax.set_title('Activation Functions Comparison', fontweight='bold', fontsize=12)
ax.set_xlabel('Input (z)')
ax.set_ylabel('Output')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: activation_functions.png")
plt.show()

# ============================================================================
# PART 9: NUMERICAL EXAMPLE (TRACED STEP BY STEP)
# ============================================================================

print("\n" + "="*80)
print("PART 9: COMPLETE NUMERICAL TRACE (2-Layer Network)")
print("="*80)

print("\n7. Step-by-step numerical example:\n")

print("Input: x = [1.0, 2.0, 3.0]")
print("\nWeights Layer 1:")
print("  W1 = [[0.1, 0.2],")
print("        [0.3, 0.4],")
print("        [0.5, 0.6]]")
print("  b1 = [0.1, 0.2]")

print("\nWeights Layer 2:")
print("  W2 = [[0.5, 0.6, 0.7],")
print("        [0.8, 0.9, 1.0]]")
print("  b2 = [0.1, 0.2, 0.3]")

# Setup
x = np.array([1.0, 2.0, 3.0])
W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
b1 = np.array([0.1, 0.2])
W2 = np.array([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]])
b2 = np.array([0.1, 0.2, 0.3])

print("\n" + "-"*80)
print("FORWARD PASS:")
print("-"*80)

# Layer 1
print("\nLayer 1:")
print("-" * 40)
print("Step 1: Z1 = x @ W1 + b1")
print(f"  x = {x}")
print(f"  x @ W1 = {np.dot(x, W1)}")
Z1 = np.dot(x, W1) + b1
print(f"  Z1 = {Z1}")

print("\nStep 2: A1 = ReLU(Z1)")
A1 = np.maximum(0, Z1)
print(f"  A1 = max(0, {Z1}) = {A1}")

# Layer 2
print("\nLayer 2 (Output):")
print("-" * 40)
print("Step 1: Z2 = A1 @ W2 + b2")
print(f"  A1 = {A1}")
print(f"  A1 @ W2 = {np.dot(A1, W2)}")
Z2 = np.dot(A1, W2) + b2
print(f"  Z2 = {Z2}")

print("\nStep 2: A2 = Softmax(Z2)  [For classification]")
exp_Z2 = np.exp(Z2 - np.max(Z2))
A2 = exp_Z2 / np.sum(exp_Z2)
print(f"  exp(Z2) = {exp_Z2}")
print(f"  A2 (Softmax) = {A2}")

print("\n" + "="*80)
print("FINAL OUTPUT:")
print("="*80)
print(f"Input: {x}")
print(f"Output probabilities: {A2}")
print(f"Predicted class: {np.argmax(A2)}")
print(f"Confidence: {np.max(A2):.2%}")


# ============================================================================
# PART 12: PRACTICE PROBLEM
# ============================================================================

print("\n" + "="*80)
print("PART 12: PRACTICE PROBLEM - SOLVE THIS!")
print("="*80)

print("""
Given:
  Input: x = [2, 3]
  
  Layer 1 weights: W1 = [[0.5, 0.2],
                          [0.3, 0.1]]
  Layer 1 bias: b1 = [0.1, 0.2]
  
  Layer 2 weights: W2 = [[0.4, 0.6],
                          [0.7, 0.5]]
  Layer 2 bias: b2 = [0.2, 0.3]

Question:
  Compute forward pass through both layers
  Use ReLU for hidden layer
  Softmax for output layer
  
  What are the final output probabilities?
  Which class is predicted?

Take your time and try! 
Answers coming next...
""")

# ============================================================================
# PART 13: SOLUTION
# ============================================================================

print("\n" + "-"*80)
print("SOLUTION:")
print("-"*80)

x_practice = np.array([2.0, 3.0])
W1_practice = np.array([[0.5, 0.2], [0.3, 0.1]])
b1_practice = np.array([0.1, 0.2])
W2_practice = np.array([[0.4, 0.6], [0.7, 0.5]])
b2_practice = np.array([0.2, 0.3])

print("\nStep 1: Z1 = x @ W1 + b1")
Z1_practice = np.dot(x_practice, W1_practice) + b1_practice
print(f"  Z1 = {Z1_practice}")

print("\nStep 2: A1 = ReLU(Z1)")
A1_practice = np.maximum(0, Z1_practice)
print(f"  A1 = {A1_practice}")

print("\nStep 3: Z2 = A1 @ W2 + b2")
Z2_practice = np.dot(A1_practice, W2_practice) + b2_practice
print(f"  Z2 = {Z2_practice}")

print("\nStep 4: A2 = Softmax(Z2)")
exp_Z2_practice = np.exp(Z2_practice - np.max(Z2_practice))
A2_practice = exp_Z2_practice / np.sum(exp_Z2_practice)
print(f"  A2 (Probabilities) = {A2_practice}")

print("\nFinal Answer:")
print(f"  Class 0 probability: {A2_practice[0]:.4f}")
print(f"  Class 1 probability: {A2_practice[1]:.4f}")
print(f"  Predicted class: {np.argmax(A2_practice)}")

# ============================================================================
# PART 14: VISUALIZATION OF FORWARD PASS
# ============================================================================

print("\n" + "="*80)
print("PART 14: VISUALIZING THE FORWARD PASS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data flow through network
ax = axes[0, 0]
layers_names = ['Input\n(784)', 'Layer 1\n(128)', 'Layer 2\n(64)', 'Output\n(10)']
layer_sizes = [784, 128, 64, 10]
x_pos = np.arange(len(layers_names))

ax.bar(x_pos, layer_sizes, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(layers_names)
ax.set_ylabel('Neurons per layer')
ax.set_title('Network Architecture: Neuron Count', fontweight='bold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3, axis='y')

# Plot 2: Activation distributions
ax = axes[0, 1]
np.random.seed(42)
layers = ['Z1\n(pre-act)', 'A1\n(ReLU)', 'Z2\n(pre-act)', 'A2\n(ReLU)']
means = [0, 0.5, 0, 0.5]
stds = [1, 1, 1, 1]

positions = np.arange(len(layers))
data = []
for mean, std in zip(means, stds):
    data.append(np.random.normal(mean, std, 1000))

bp = ax.boxplot(data, labels=layers, patch_artist=True)
for patch, color in zip(bp['boxes'], ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']):
    patch.set_facecolor(color)
ax.set_ylabel('Activation Value')
ax.set_title('Distribution of Activations Through Network', fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Plot 3: Single sample forward pass
ax = axes[1, 0]
stages = ['Input\n(784→)', 'Layer 1\n(128→)', 'Layer 2\n(64→)', 'Output\n(10→)']
values = [784, 128, 64, 10]
colors_grad = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']

y_pos = np.arange(len(stages))
ax.barh(y_pos, values, color=colors_grad)
ax.set_yticks(y_pos)
ax.set_yticklabels(stages)
ax.set_xlabel('Dimension')
ax.set_title('Data Dimensions Through Forward Pass', fontweight='bold')
for i, v in enumerate(values):
    ax.text(v + 20, i, str(v), va='center')
ax.grid(True, alpha=0.3, axis='x')

# Plot 4: Computational flow
ax = axes[1, 1]
ax.text(0.5, 0.95, 'FORWARD PASS FLOWCHART', ha='center', fontsize=12, fontweight='bold', transform=ax.transAxes)

flow_text = """
Input X: (m, 784)
    ↓
@ W1 (784, 128) + b1 (128) → Z1 (m, 128)
    ↓
ReLU(Z1) → A1 (m, 128)
    ↓
@ W2 (128, 64) + b2 (64) → Z2 (m, 64)
    ↓
ReLU(Z2) → A2 (m, 64)
    ↓
@ W3 (64, 10) + b3 (10) → Z3 (m, 10)
    ↓
Softmax(Z3) → A3 (m, 10)
    ↓
Output: Probabilities for 10 classes
"""

ax.text(0.1, 0.85, flow_text, fontsize=10, family='monospace', 
        transform=ax.transAxes, verticalalignment='top')
ax.axis('off')

plt.tight_layout()
plt.savefig('forward_propagation_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: forward_propagation_visualization.png")
plt.show()

