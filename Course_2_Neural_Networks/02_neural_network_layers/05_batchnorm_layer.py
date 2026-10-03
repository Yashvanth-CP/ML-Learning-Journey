"""
11. BATCH NORMALIZATION
=======================

Normalize inputs to each layer. Speeds up training, prevents vanishing gradients.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("BATCH NORMALIZATION: Normalizing Layer Inputs")
print("="*70)


class BatchNormLayer:
    """
    Batch normalization
    """
    
    def __init__(self, input_size, epsilon=1e-5, momentum=0.9):
        """
        Args:
            input_size: Size of input
            epsilon: Small value for numerical stability
            momentum: For running statistics
        """
        self.input_size = input_size
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Learnable parameters
        self.gamma = np.ones((1, input_size))
        self.beta = np.zeros((1, input_size))
        
        # Running statistics
        self.running_mean = np.zeros((1, input_size))
        self.running_var = np.ones((1, input_size))
        
        print(f"BatchNorm layer created (input_size={input_size})")
    
    def forward(self, X, training=True):
        """
        Normalize batch
        """
        if training:
            # Batch statistics
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running statistics
            batch_mean = self.running_mean
            batch_var = self.running_var
        
        # Normalize
        X_norm = (X - batch_mean) / np.sqrt(batch_var + self.epsilon)
        
        # Scale and shift
        output = self.gamma * X_norm + self.beta
        
        return output


print("\n1. BATCH NORMALIZATION INTUITION")
print("-"*70)

print("""
Internal Covariate Shift Problem:
  - As network trains, weights change
  - Distribution of layer inputs changes
  - Slower training, unstable gradients

Solution: Batch Normalization
  - Normalize each batch before passing to next layer
  - Keeps mean ≈ 0, std ≈ 1
  - Stabilizes training
  - Allows higher learning rates

Formula:
  1. Compute batch mean: μ = mean(X)
  2. Compute batch variance: σ² = var(X)
  3. Normalize: X̂ = (X - μ) / √(σ² + ε)
  4. Scale and shift: Y = γ·X̂ + β
     (γ and β are learned parameters)
""")


print("\n2. BATCHNORM IN PRACTICE")
print("-"*70)

# Create layer
bn = BatchNormLayer(input_size=10)

# Raw input (random, high variance)
X_raw = np.random.randn(32, 10) * 10 + 100

print(f"Before BatchNorm:")
print(f"  Mean: {X_raw.mean(axis=0).mean():.2f}")
print(f"  Std:  {X_raw.std(axis=0).mean():.2f}")

X_norm = bn.forward(X_raw, training=True)

print(f"\nAfter BatchNorm:")
print(f"  Mean: {X_norm.mean(axis=0).mean():.4f}")
print(f"  Std:  {X_norm.std(axis=0).mean():.4f}")


print("\n3. ARCHITECTURE WITH BATCHNORM")
print("-"*70)

print("""
Without BatchNorm:
  Dense(512) → ReLU
  Dense(256) → ReLU
  Dense(128) → ReLU
  Dense(10) → Softmax

With BatchNorm (Modern):
  Dense(512) → BatchNorm → ReLU
  Dense(256) → BatchNorm → ReLU
  Dense(128) → BatchNorm → ReLU
  Dense(10) → Softmax

Benefits:
  ✓ 2-10x faster training
  ✓ Higher learning rates possible
  ✓ Reduces sensitivity to weight initialization
  ✓ Acts as regularizer (can drop dropout!)
  ✓ Smoother loss landscape
""")


print("\n4. TRAINING VS TESTING")
print("-"*70)

print("""
During Training:
  - Use batch statistics (mean/var of current batch)
  - Noisy but updates quickly
  - Helps regularization

During Testing/Inference:
  - Use running statistics (averaged across batches)
  - Stable predictions
  - Computed with momentum during training
""")


print("\n5. BATCHNORM EFFECT")
print("-"*70)

# Simulate loss curves
epochs = np.arange(100)

# Without BatchNorm
loss_no_bn = 2 * np.exp(-0.05 * epochs) + 0.1 * np.random.randn(100) * np.exp(-0.03 * epochs)
loss_no_bn = np.maximum(loss_no_bn, 0.1)

# With BatchNorm
loss_with_bn = 2 * np.exp(-0.1 * epochs) + 0.05 * np.random.randn(100) * np.exp(-0.05 * epochs)
loss_with_bn = np.maximum(loss_with_bn, 0.1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Plot 1: Loss curves
ax = axes[0]
ax.plot(epochs, loss_no_bn, 'r-', label='Without BatchNorm', linewidth=2)
ax.plot(epochs, loss_with_bn, 'g-', label='With BatchNorm', linewidth=2)
ax.set_title('Training Speed: BatchNorm Effect', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

# Plot 2: Input distributions
ax = axes[1]

# Before normalization
x1 = np.random.randn(1000) * 5 + 50
x2 = np.random.randn(1000) * 5  # After normalization

ax.hist(x1, bins=30, alpha=0.5, label='Before BatchNorm', color='red')
ax.hist(x2, bins=30, alpha=0.5, label='After BatchNorm', color='green')
ax.axvline(x=x1.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={x1.mean():.1f}')
ax.axvline(x=x2.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean={x2.mean():.2f}')
ax.set_title('Input Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('batchnorm_effect.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: batchnorm_effect.png")
plt.show()


print("\n" + "="*70)
print("✓ BatchNorm: Modern deep learning essential!")
print("="*70)