"""
09. FLATTEN LAYER
=================

Reshape multi-dimensional data to 1D for dense layers.
"""

import numpy as np

print("="*70)
print("FLATTEN LAYER: Reshape for Dense Layers")
print("="*70)


class FlattenLayer:
    """Reshape layer"""
    
    def __init__(self):
        self.original_shape = None
        print("Flatten layer created")
    
    def forward(self, X):
        """Flatten to 2D: (batch, features)"""
        self.original_shape = X.shape
        batch_size = X.shape[0]
        return X.reshape(batch_size, -1)
    
    def backward(self, dA):
        """Reshape back to original"""
        return dA.reshape(self.original_shape)


print("\n1. FLATTEN BASICS")
print("-"*70)

# After conv layers
conv_output = np.random.randn(32, 32, 5, 5)  # batch=32, channels=32, 5x5
print(f"After Conv layer: {conv_output.shape}")
print(f"  batch_size: 32")
print(f"  channels: 32")
print(f"  height: 5")
print(f"  width: 5")

flatten = FlattenLayer()
flattened = flatten.forward(conv_output)
print(f"\nAfter Flatten: {flattened.shape}")
print(f"  (32, 800) ← 32×5×5 = 800 features")


print("\n2. CNN → DENSE PIPELINE")
print("-"*70)

"""
Typical pipeline:

Input (224×224×3)
  ↓
Conv + ReLU → (112×112×32)
  ↓
MaxPool → (56×56×32)
  ↓
Conv + ReLU → (28×28×64)
  ↓
MaxPool → (14×14×64)
  ↓
FLATTEN → (12,544)
  ↓
Dense(4096) + ReLU
  ↓
Dense(1000) + Softmax
  ↓
Output (1000 classes)
"""


print("\n3. SHAPE TRACKING")
print("-"*70)

shapes = [
    (32, 64, 28, 28),  # Conv output
    (32, 128, 14, 14), # After pool
    (32, 256, 7, 7),   # After conv
]

for shape in shapes:
    batch, channels, h, w = shape
    flattened_size = channels * h * w
    print(f"Conv output {shape} → Flattened ({batch}, {flattened_size})")


