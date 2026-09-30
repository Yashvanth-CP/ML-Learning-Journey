"""
04. FORWARD PROPAGATION
Understand exactly how data flows through a neural network.
Forward pass = Input → Processing → Output
"""

import numpy as np
import matplotlib.pyplot as plt
 
print("**-**"*30)
print("FORWARD PROPAGATION: How Data Flows Through Network")
print("-"*30)

# STEP-BY-STEP 

""" 2-Layer Network Forward Pass:
 
INPUT LAYER
    ↓
LAYER 1 LINEAR: Z₁ = X·W₁ + b₁
    ↓
LAYER 1 ACTIVATION: A₁ = ReLU(Z₁) = max(0, Z₁)
    ↓
LAYER 2 LINEAR: Z₂ = A₁·W₂ + b₂
    ↓
LAYER 2 ACTIVATION: A₂ = Sigmoid(Z₂) = 1/(1+e^(-Z₂))
    ↓
OUTPUT
 
Each step transforms the data into new representations
 """

# Example 

np.random.seed(42)

# single sample 

X = np.array([[1.5, -0.5]]) # 1 sample, 2 feature

print(f" Input X shape : {X.shape}")
print(f" X = {X}")

# layer 1 

w1 = np.array([[0.5, 0.2, -0.3],
               [-0.1, 0.4, 0.6]])
b1 = np.array([[0.1, -0.2, 0.3]]) # 1x3 matrix

print(f"\n  Layer 1 weights (W1) shape: {w1.shape}")
print(f"  W1 =\n{w1}")
print(f"  Layer 1 bias (b1) shape: {b1.shape}")
print(f"  b1 = {b1}")

# Layer 2
w2 = np.array([[0.3],     # 3x1 matrix
               [-0.4],
               [0.5]])
b2 = np.array([[0.2]])    # 1x1 matrix
 
print(f"\n  Layer 2 weights (W2) shape: {w2.shape}")
print(f"  W2 =\n{w2}")
print(f"  Layer 2 bias (b2): {b2}")



# FORWARD PASS

print("\n Forward pass")

# Step 1 z1 = X.w1 + b1

print("\n step 1  : z = x*w1  + b1 (Linear combination in layer 1)")

Z1 = np.dot(X, w1) + b1

print(f"  X·w1 = {np.dot(X, w1)}")
print(f"  X·w1 + b1 = {Z1}")
print(f"  Z1 shape: {Z1.shape}")

# step 2 : A1 = ReLU 

print("\n step 2 : A = ReLU(z1) (Activation in layer 1)")


A1 = np.maximum(0, Z1)

print(f"  Before ReLU: {Z1}")
print(f"  After ReLU:  {A1}")
print(f"  A1 shape: {A1.shape}")
print(f"  (negative values became 0, positive stayed same)")

# step 3 : Z2 =A1*W2 + b2
print("\nStep 3: Z₂ = A₁·W₂ + b₂ (Linear combination in layer 2)")

Z2 = np.dot(A1, w2) + b2
print(f"  A1·W2 = {np.dot(A1, w2)}")
print(f"  A1·W2 + b2 = {Z2}")
print(f"  Z2 shape: {Z2.shape}")

# step 4 : Sigmoid(Z2)

print("\nStep 4: A₂ = Sigmoid(Z₂) (Activation in layer 2)")
A2 = 1/ (1 + np.exp(-Z2))

print(f"  Before Sigmoid: {Z2}")
print(f"  After Sigmoid:  {A2}")
print(f"  A2 shape: {A2.shape}")
print(f"  (Converted to probability between 0 and 1)")
 
print(f"\n✓ Final prediction: {A2[0, 0]:.4f}")
print(f"  Interpretation: {A2[0, 0]*100:.1f}% confidence for class 1")


# BATCH FORWARD PASS

print("\n\n Batch foward pass (multiple sample: ) : ")

print("\nWhen we have multiple samples, matrix multiplication handles all at once!")
 
# Create batch of samples
X_batch = np.array([
    [1.5, -0.5],
    [-1.0, 0.8],
    [0.5, 0.5],
    [-0.5, -1.0]
])

print(f"\nInput batch shape: {X_batch.shape} (4 samples, 2 features)")

Z1_batch = np.dot(X_batch, w1) + b1
A1_batch = np.maximum(0, Z1_batch)
Z2_batch = np.dot(A1_batch, w2) + b2
A2_batch = 1 / (1+ np.exp(-Z2_batch))

print(f"\nZ1 shape: {Z1_batch.shape} (4 samples, 3 neurons)")
print(f"A1 shape: {A1_batch.shape} (4 samples, 3 neurons)")
print(f"Z2 shape: {Z2_batch.shape} (4 samples, 1 output)")
print(f"A2 shape: {A2_batch.shape} (4 samples, 1 output)")


print(" Prediction for each sample : ")
for i, pre in enumerate(A2_batch):
    print(f" Sample {i + 1} : {pre[0]:.4f}")


# DIMENSIONALITY TRACKING
"""
Important: Keep track of dimensions!
 
Input shape: (m, input_size)      m=batch size, input_size=2
  ↓
Layer 1 weights: (input_size, hidden_size) = (2, 3)
  ↓
Z1 shape: (m, hidden_size) = (m, 3)
A1 shape: (m, hidden_size) = (m, 3)
  ↓
Layer 2 weights: (hidden_size, output_size) = (3, 1)
  ↓
Z2 shape: (m, output_size) = (m, 1)
A2 shape: (m, output_size) = (m, 1)
 
Rule: (m, a) · (a, b) = (m, b)
Matrix multiplication preserves batch dimension!
"""

#  ACTIVATION FUNCTIONS

print("\n 5. ACTIVATION FUNCTIONS FORWARD EXPLAINED ")

print("\n Layer 1 : ReLU(z)")
print(" formula : f(z) = max(0, z)")
print("  Effect: Keeps positive values, kills negative values")

Z_test = np.array([-2, -1, 0, 0.5, 1, 2])
relu_out = np.maximum(0, Z_test)
print(f"Example: {Z_test} -> {relu_out}")


 
print("\nLayer 2: Sigmoid(z)")
print("  Purpose: Convert to probability")
print("  Formula: σ(z) = 1 / (1 + e^(-z))")

sigmoid_out = 1 / ( 1 + np.exp(-Z_test))
print(f"  Example: {Z_test} → {np.round(sigmoid_out, 4)}")


# VECTORIZATION

"""
for sample in range(m):
    for neuron in range(hidden_size):
        z = 0
        for feature in range(input_size):
            z += X[sample, feature] * W1[feature, neuron]
        z += b1[neuron]
        a1[sample, neuron] = max(0, z)
→ 4 nested loops! Very slow!
"""
print(""" z1 = x* w1 + b1 """)

print("\n speed comparision : ")

X_large = np.random.rand(1000, 100)
W_large = np.random.rand(100, 50)

# vectorised 

import time 
start= time.time()

for _ in range(100):
    Z = np.dot(X_large, W_large)
vectorized_time = time.time() - start

print(f"  1000x100 @ 100x50 matrix (100 iterations):")
print(f"  Vectorized: {vectorized_time*1000:.2f} ms")
print(f"  →Always use matrix operations!")

print("\n\n7. VISUALIZING INFORMATION FLOW")
print("-"*70)
 
# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
# Sample data for visualization
np.random.seed(42)
X_vis = np.random.randn(100, 2)
W1_vis = np.random.randn(2, 3) * 0.5
b1_vis = np.zeros((1, 3))
W2_vis = np.random.randn(3, 1) * 0.5
b2_vis = np.array([[0.1]])
 
# Forward pass
Z1_vis = np.dot(X_vis, W1_vis) + b1_vis
A1_vis = np.maximum(0, Z1_vis)
Z2_vis = np.dot(A1_vis, W2_vis) + b2_vis
A2_vis = 1 / (1 + np.exp(-Z2_vis))
 
# Plot 1: Input distribution
ax = axes[0, 0]
ax.scatter(X_vis[:, 0], X_vis[:, 1], alpha=0.6, s=30)
ax.set_title('Step 0: Input (X)', fontsize=11, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.grid(True, alpha=0.3)
 
# Plot 2: After Layer 1 Linear
ax = axes[0, 1]
ax.scatter(Z1_vis[:, 0], Z1_vis[:, 1], alpha=0.6, s=30, c='orange')
ax.set_title('Step 1: After Z₁ = X·W₁ + b₁', fontsize=11, fontweight='bold')
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.grid(True, alpha=0.3)
 
# Plot 3: After Layer 1 ReLU
ax = axes[1, 0]
ax.scatter(A1_vis[:, 0], A1_vis[:, 1], alpha=0.6, s=30, c='green')
ax.set_title('Step 2: After A₁ = ReLU(Z₁)', fontsize=11, fontweight='bold')
ax.set_xlabel('Neuron 1')
ax.set_ylabel('Neuron 2')
ax.grid(True, alpha=0.3)
 
# Plot 4: Final predictions
ax = axes[1, 1]
scatter = ax.scatter(np.arange(len(A2_vis)), A2_vis, c=A2_vis, cmap='RdYlGn', 
                     alpha=0.7, s=50)
ax.set_title('Step 4: After A₂ = Sigmoid(Z₂)', fontsize=11, fontweight='bold')
ax.set_xlabel('Sample')
ax.set_ylabel('Prediction (Probability)')
ax.set_ylim([0, 1])
ax.axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
plt.colorbar(scatter, ax=ax)
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('forward_propagation_flow.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: forward_propagation_flow.png")
plt.show()
 
 
# ============ EDGE CASES ============
 
# COMMON MISTAKES IN FORWARD PASS
 
"""
Mistake 1: Wrong matrix dimensions
  ❌ X (10, 2) @ W1 (3, 5)  → ERROR! Can't multiply
  ✅ X (10, 2) @ W1 (2, 5)  → OK! Result: (10, 5)
 
Mistake 2: Forgetting bias
  ❌ Z = X @ W  (missing +b)
  ✅ Z = X @ W + b
 
Mistake 3: Wrong activation
  ❌ Sigmoid in hidden layer (slow gradients)
  ✅ ReLU in hidden layer (fast training)
 
Mistake 4: Not clipping sigmoid input
  ❌ sigmoid(1000) → Overflow/NaN
  ✅ sigmoid(np.clip(z, -500, 500)) → Safe
 
Mistake 5: Forgetting to save values
  ❌ Don't save Z1, A1 for backward pass
  ✅ Save all intermediate values
"""
 
 
# SUMMARY 
"""
Key Points:
 
1. EQUATION
   A₂ = Sigmoid(A₁·W₂ + b₂)
   where A₁ = ReLU(X·W₁ + b₁)
 
2. DIMENSIONS MATTER
   Track shapes through network!
   (m, a) @ (a, b) = (m, b)
 
3. ACTIVATIONS
   - Hidden: ReLU (non-linearity)
   - Output: Sigmoid (probability)
 
4. VECTORIZATION
   Use matrix operations, not loops!
 
5. SAVE INTERMEDIATE VALUES
   Z1, A1, Z2, A2 needed for backward pass
 
NEXT: Backward propagation (the learning algorithm!)
"""
