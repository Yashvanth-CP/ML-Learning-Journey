"""
10. DROPOUT LAYER
 Regularization technique: randomly disable neurons during training.
Prevents overfitting!
"""

import numpy as np
import matplotlib.pyplot as plt

print("_-"*70)
print("DROPOUT LAYER: Fighting Overfitting")
print("-_"*70)

class DropoutLayer :
    # DropLayer: Randomly disable neurons 

    def __init__(self, dropout_rate =0.5):
        # Args : dropout_rate = Fraction of neurons to disable (0.5 = 50%)

        self.dropout_rate = dropout_rate
        self.mask = None
        print(f"Dropout layer created (rate = {dropout_rate})")


    def forward(self,X, training=True):
        if not training:
            return X
        self.mask = np.random.binomial(1, 1 - self.dropout_rate)

        # Apply mask and scale 
        return (X * self.mask) / (1- self.dropout_rate)

    def backward(self, dA):

        # Backward pass 
        return (dA * self.mask) / (1 - self.dropout_rate)

print("\n DROPOUT INTUITION")
print("-"*70)
 
"""
Problem: Network memorizes training data → Overfits
Solution: Randomly "turn off" neurons during training
 
Dropout rate = 0.5:
  - 50% of neurons deactivated randomly each iteration
  - Forces network to learn redundant features
  - During testing: use all neurons (with scaling)
 
Why it works:
  - Like training multiple sub-networks
  - Averaging their predictions at test time
  - Prevents co-adaptation of neurons
"""
 
 
print("\n DROPOUT IN ACTION")
print("-"*70)
 
# Create layer
dropout = DropoutLayer(dropout_rate=0.5)
 
# Forward pass
X = np.ones((1, 10))  # 10 neurons
print(f"Input: {X}")
 
X_dropped = dropout.forward(X, training=True)
print(f"After dropout: {X_dropped}")
print(f"Active neurons: {np.sum(X_dropped > 0)}/10")
 
 
print("\nNETWORK WITH DROPOUT")
print("-"*70)
 
"""
Architecture:
  Dense(512) → ReLU → Dropout(0.5)
           → Dense(256) → ReLU → Dropout(0.5)
           → Dense(10) → Softmax
 
Dropout placement:
  ✓ After hidden layers (not input!)
  ✓ Not after output layer
  ✓ Rate = 0.5 (50%) is common
  ✓ Can be 0.2-0.5 typically
"""
 
 
print("\n DROPOUT RATES")
print("-"*70)
 
rates = [0.0, 0.2, 0.5, 0.7]
print("Common dropout rates:")
for rate in rates:
    print(f"  {rate}: {(1-rate)*100:.0f}% neurons active")
 
print("\nEffect on overfitting:")
print("  No dropout (0.0):   May overfit")
print("  Light dropout (0.2): Good for small networks")
print("  Medium dropout (0.5): Standard for large networks")
print("  Heavy dropout (0.7):  For very large networks")
 
 
print("\n VISUALIZATION")
print("-"*70)
 
# Simulate training and testing
np.random.seed(42)
 
# Without dropout
losses_no_dropout = np.array([0.8, 0.6, 0.4, 0.3, 0.25, 0.24, 0.24, 0.24])
test_losses_no_dropout = np.array([0.8, 0.65, 0.5, 0.5, 0.6, 0.8, 1.1, 1.5])
 
# With dropout
losses_dropout = np.array([0.9, 0.75, 0.6, 0.5, 0.45, 0.42, 0.40, 0.38])
test_losses_dropout = np.array([0.9, 0.76, 0.61, 0.52, 0.48, 0.45, 0.43, 0.41])
 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
 
# Plot 1: Loss curves
ax = axes[0]
epochs = range(len(losses_no_dropout))
ax.plot(epochs, losses_no_dropout, 'b-', label='Train (no dropout)', marker='o')
ax.plot(epochs, test_losses_no_dropout, 'r-', label='Test (no dropout)', marker='x')
ax.plot(epochs, losses_dropout, 'b--', label='Train (dropout)', marker='o', alpha=0.7)
ax.plot(epochs, test_losses_dropout, 'r--', label='Test (dropout)', marker='x', alpha=0.7)
ax.set_title('Dropout Effect on Overfitting', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
 
# Plot 2: Gap visualization
ax = axes[1]
gap_no_dropout = test_losses_no_dropout - losses_no_dropout
gap_dropout = test_losses_dropout - losses_dropout
 
ax.plot(epochs, gap_no_dropout, 'r-', label='Gap (no dropout)', marker='o', linewidth=2)
ax.plot(epochs, gap_dropout, 'g-', label='Gap (dropout)', marker='o', linewidth=2)
ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax.set_title('Overfitting Gap (Test - Train)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss Difference')
ax.legend()
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('dropout_effect.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: dropout_effect.png")
plt.show()
 

 
"""
✓ Dropout is a regularization technique
✓ Prevents overfitting by adding noise
✓ Only active during training
✓ Disabled during testing/inference
✓ Common rates: 0.2-0.5
✓ Must scale activations (divide by 1-dropout_rate)
 
Implementation in practice:
  model = Sequential([
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
  ])
"""
 