"""
 BACKWARD PROPAGATION (BACKPROP)
 """

import numpy as np
import matplotlib.pyplot as plt

print("BACKWARD PROPAGATION: How Networks Learn")

#THE CORE IDEA

"""
Question: How do weights in layer 1 affect the loss?
 
The chain rule answers this:
 
Loss = f(A2) where A2 = g(Z2) where Z2 = h(A1) where A1 = k(Z1)
 
∂Loss/∂W1 = ∂Loss/∂A2 · ∂A2/∂Z2 · ∂Z2/∂A1 · ∂A1/∂Z1 · ∂Z1/∂W1

                              Chain of derivatives!
 
This is automatic differentiation (backpropagation)
"""
np.random.seed(42)
 
# Forward pass first (need these values)
X = np.array([[1.0, 0.5]])
W1 = np.array([[0.5, 0.2], [-0.1, 0.4]])
b1 = np.array([[0.1, -0.2]])
W2 = np.array([[0.3], [-0.4]])
b2 = np.array([[0.2]])
y = np.array([[1.0]])
 
print("Setup:")
print(f"  Input X: {X}")
print(f"  Target y: {y}")
# Forward
Z1 = np.dot(X, W1) + b1
A1 = np.maximum(0, Z1)
Z2 = np.dot(A1, W2) + b2
A2 = 1 / (1 + np.exp(-Z2))
 
print(f"\nForward pass results:")
print(f"  Z1: {Z1}")
print(f"  A1: {A1}")
print(f"  Z2: {Z2}")
print(f"  A2: {A2} (prediction)")
 
# Loss
loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
print(f"  Loss: {loss:.4f}")
 
# ===== BACKWARD PASS =====
print(f"\nBackward pass:")
 
# Step 1: Layer 2 error
dZ2 = A2 - y
print(f"\n  Step 1: dZ₂ = A₂ - y = {A2} - {y} = {dZ2}")
 
# Step 2: Layer 2 gradients
m = 1  # batch size = 1
dW2 = (1/m) * np.dot(A1.T, dZ2)
db2 = (1/m) * np.sum(dZ2)
 
print(f"  Step 2: dW₂ = (1/m)·A₁ᵀ·dZ₂")
print(f"          = (1/{m})·{A1.T}·{dZ2}")
print(f"          = {dW2}")
print(f"          db₂ = {db2}")
 
# Step 3: Propagate error back to layer 1
dA1 = np.dot(dZ2, W2.T)
print(f"\n  Step 3: dA₁ = dZ₂·W₂ᵀ = {dZ2}·{W2.T} = {dA1}")
 
# Step 4: Apply ReLU derivative
ReLU_derivative = (Z1 > 0).astype(float)
dZ1 = dA1 * ReLU_derivative
 
print(f"  Step 4: ReLU'(Z₁) = (Z₁ > 0) = {ReLU_derivative}")
print(f"          dZ₁ = dA₁ * ReLU'(Z₁) = {dA1} * {ReLU_derivative} = {dZ1}")
 
# Step 5: Layer 1 gradients
dW1 = (1/m) * np.dot(X.T, dZ1)
db1 = (1/m) * np.sum(dZ1)
 
print(f"  Step 5: dW₁ = (1/m)·Xᵀ·dZ₁")
print(f"          = (1/{m})·{X.T}·{dZ1}")
print(f"          = {dW1}")
print(f"          db₁ = {db1}")
 
# ===== UPDATE WEIGHTS =====
 
print(f"\n  Step 6: Update weights (learning rate = 0.1)")
learning_rate = 0.1
 
W1_new = W1 - learning_rate * dW1
b1_new = b1 - learning_rate * db1
W2_new = W2 - learning_rate * dW2
b2_new = b2 - learning_rate * db2
 
print(f"          W₁_new = W₁ - 0.1·dW₁")
print(f"                 = {W1} - 0.1·{dW1}")
print(f"                 = {W1_new}")
 
 
# ============ WHY THIS WORKS ============
 

 
"""
Key insight: Gradients tell us which direction to adjust weights!
 
If ∂Loss/∂W = 0.5 (positive):
  Increasing W → increases Loss (BAD)
  So we should DECREASE W
  W_new = W - learning_rate * 0.5
 
If ∂Loss/∂W = -0.3 (negative):
  Decreasing W → increases Loss (BAD)
  So we should INCREASE W
  W_new = W - learning_rate * (-0.3) = W + learning_rate * 0.3
 
General rule: W_new = W - learning_rate * ∂Loss/∂W
 
This moves weights toward lower loss (gradient descent!)
"""
 
 
# ============ VANISHING GRADIENTS ============
 
"""
Problem with sigmoid in hidden layers:
 
Sigmoid derivative: σ'(z) = σ(z)·(1 - σ(z))
At z = ±10: σ'(z) ≈ 0.00000  (tiny!)
 
When gradients multiply through layers:
∂Loss/∂W₁ = ... · σ'(z₂) · ... · σ'(z₁) · ...
           ≈ ... · 0 · ... · 0 · ...
           = 0 (practically)
 
Result: Layer 1 weights barely update (can't learn!)
 
Solution: Use ReLU instead of sigmoid in hidden layers!
 
ReLU derivative: ReLU'(z) = 1 (if z > 0) or 0 (if z < 0)
Avoids multiplying by tiny numbers → gradients flow!
"""
 
 
# ============ MATRIX DIMENSIONS IN BACKPROP ============

 
"""
Key rule: Dimensions must line up for addition!
 
Forward shapes:
  X: (m, 2)
  W1: (2, 3)
  Z1: (m, 3) = X @ W1
  A1: (m, 3)
  W2: (3, 1)
  Z2: (m, 1) = A1 @ W2
  A2: (m, 1)
 
Backward shapes (must match for element-wise operations):
  dZ2: (m, 1) = (m, 1) - (m, 1)  ✓
  dW2: (3, 1) = (3, m) @ (m, 1)  ✓
  dA1: (m, 3) = (m, 1) @ (1, 3)  ✓
  dZ1: (m, 3) = (m, 3) * (m, 3)  ✓
  dW1: (2, 3) = (2, m) @ (m, 3)  ✓
 
Always check dimensions!
"""
 
 
# ============ FULL BACKPROP IMPLEMENTATION ============
 
print("\n8. COMPLETE BACKPROP FUNCTION")
print("-"*70)
 
class SimpleNetwork:
    """Network with full forward and backward"""
    
    def __init__(self, input_size=2, hidden_size=3, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros((1, 1))
    
    def forward(self, X):
        """Forward pass"""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-np.clip(self.Z2, -500, 500)))
        self.X = X
        return self.A2
    
    def backward(self, y):
        """Complete backward pass"""
        m = self.X.shape[0]
        
        # Layer 2 backward
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        # Layer 1 backward
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(self.X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        # Update
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
 
print("\n✓ Implementation created (see code)")
 
 
# ============ VISUALIZATION ============
 
print("\n9. VISUALIZING GRADIENT FLOW")
print("-"*70)
 
# Create example
np.random.seed(42)
X = np.random.randn(100, 2)
y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int).reshape(-1, 1)
 
network = SimpleNetwork(input_size=2, hidden_size=4, learning_rate=0.1)
 
losses = []
w1_changes = []
w2_changes = []
 
for epoch in range(100):
    # Forward
    A2 = network.forward(X)
    loss = -np.mean(y * np.log(A2 + 1e-8) + (1 - y) * np.log(1 - A2 + 1e-8))
    losses.append(loss)
    
    # Save weight magnitudes
    w1_changes.append(np.abs(network.W1).mean())
    w2_changes.append(np.abs(network.W2).mean())
    
    # Backward
    network.backward(y)
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
# Plot 1: Loss over time
ax = axes[0, 0]
ax.plot(losses, 'b-', linewidth=2)
ax.set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)
 
# Plot 2: Weight magnitude changes
ax = axes[0, 1]
ax.plot(w1_changes, 'r-', label='Layer 1 weights', linewidth=2)
ax.plot(w2_changes, 'b-', label='Layer 2 weights', linewidth=2)
ax.set_title('Weight Magnitudes Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Avg |Weight|')
ax.legend()
ax.grid(True, alpha=0.3)
 
# Plot 3: Gradient magnitude (simulated)
ax = axes[1, 0]
gradient_layer1 = np.diff([0] + w1_changes)
gradient_layer2 = np.diff([0] + w2_changes)
ax.plot(np.abs(gradient_layer1), 'r-', label='Layer 1 gradients', linewidth=2)
ax.plot(np.abs(gradient_layer2), 'b-', label='Layer 2 gradients', linewidth=2)
ax.set_title('Gradient Flow Through Network', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('|Gradient|')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)
 
# Plot 4: Learning comparison
ax = axes[1, 1]
ax.text(0.5, 0.7, 'Backpropagation Works!', ha='center', fontsize=16, 
        fontweight='bold', transform=ax.transAxes)
ax.text(0.5, 0.5, 
        f'Loss: {losses[0]:.3f} → {losses[-1]:.3f}\n'
        f'Reduction: {(1-losses[-1]/losses[0])*100:.1f}%',
        ha='center', fontsize=12, transform=ax.transAxes)
ax.axis('off')
 
plt.tight_layout()
plt.savefig('backward_propagation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: backward_propagation.png")
plt.show()
 
 
# ============ SUMMARY ============
 
print("\n" + "="*70)
print("BACKPROPAGATION SUMMARY")

"""
Key Points:
 
1. THE CHAIN RULE
   ∂Loss/∂W1 = chain of derivatives from loss back to weight
 
2. GRADIENT DESCENT
   Update: W_new = W - learning_rate · ∂Loss/∂W
 
3. THE DIRECTION
   Gradients point uphill (toward higher loss)
   We go opposite direction (downhill)
 
4. LAYER 2 BACKWARD
   dZ2 = A2 - y
   (Simple for cross-entropy!)
 
5. LAYER 1 BACKWARD
   - Propagate error: dA1 = dZ2 @ W2.T
   - Apply activation: dZ1 = dA1 * ReLU'(Z1)
   - Calculate gradients: dW1, db1
 
6. WHY ReLU MATTERS
   Sigmoid → tiny gradients (vanishing)
   ReLU → constant gradients (flow!)
 
7. VECTORIZATION AGAIN
   Use matrix operations, not loops!
 
NEXT: Put it all together in complete training loop!
"""