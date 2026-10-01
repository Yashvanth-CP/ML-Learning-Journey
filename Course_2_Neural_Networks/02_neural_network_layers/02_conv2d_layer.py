"""
08. CONV2D LAYER (Convolutional Layer)
======================================

For image processing! Applies filters to detect features.
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("CONV2D LAYER: Images and Feature Detection")
print("="*70)


print("\n1. CONVOLUTION BASICS")
print("-"*70)

"""
Convolution: Slide a filter over input to detect patterns

Example: 3x3 filter on image
┌─────────────────┐
│ Image (5x5)     │
│  ┌───┐          │
│  │ * │ Filter   │
│  └───┘ (3x3)    │
└─────────────────┘
     ↓
Result: Applies filter at every position
"""

class Conv2DLayer:
    """
    2D Convolutional layer
    """
    
    def __init__(self, input_channels=3, output_channels=32, 
                 kernel_size=3, stride=1, padding=1):
        """
        Args:
            input_channels: Input channels (RGB=3)
            output_channels: Number of filters
            kernel_size: Size of filter (3x3, 5x5, etc)
            stride: Step size
            padding: Zero padding
        """
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize filters
        self.filters = np.random.randn(output_channels, input_channels, 
                                       kernel_size, kernel_size) * 0.01
        self.biases = np.zeros(output_channels)
        
        params = output_channels * input_channels * kernel_size * kernel_size
        
        print(f"Conv2D Layer created:")
        print(f"  Input channels: {input_channels}")
        print(f"  Output channels: {output_channels}")
        print(f"  Kernel size: {kernel_size}x{kernel_size}")
        print(f"  Stride: {stride}")
        print(f"  Padding: {padding}")
        print(f"  Total parameters: {params + output_channels:,}")
    
    def forward(self, X):
        """Forward pass (simplified)"""
        # This is simplified - real convolution is complex!
        print("  (Forward pass skipped in demo - implement with stride/padding)")
        return X


print(" CONV2D LAYER CREATION")
print("-"*70)

conv_layer = Conv2DLayer(input_channels=3, output_channels=32, 
                        kernel_size=3, stride=1, padding=1)


print("\n COMMON CNN ARCHITECTURES")
print("-"*70)

"""
LeNet (Classic CNN):
  Input (28x28)
  → Conv(6) → Pool → Conv(16) → Pool
  → Dense(120) → Dense(84) → Dense(10)

VGG (Modern CNN):
  Input (224x224)
  → Conv(64) → Conv(64) → Pool
  → Conv(128) → Conv(128) → Pool
  → Conv(256) → Conv(256) → Conv(256) → Pool
  → Dense(4096) → Dense(4096) → Dense(1000)

ResNet (Deep CNN):
  Similar to VGG but with skip connections
  → Allows much deeper networks
"""


print("\n4. FILTER VISUALIZATION")
print("-"*70)

# Visualize what filters do
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Edge detection filter
edge_filter = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

# Blur filter
blur_filter = np.array([[1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1]]) / 9

# Sharpen filter
sharpen_filter = np.array([[0, -1, 0],
                          [-1, 5, -1],
                          [0, -1, 0]])

filters = [edge_filter, blur_filter, sharpen_filter]
names = ['Edge Detection', 'Blur', 'Sharpen']

for idx, (f, name) in enumerate(zip(filters, names)):
    ax = axes[0, idx]
    im = ax.imshow(f, cmap='RdBu')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)
    
    # Show text values
    for i in range(3):
        for j in range(3):
            text = ax.text(j, i, f'{f[i, j]:.2f}',
                          ha="center", va="center", color="black")

# Random filters learned by network
for idx in range(3):
    ax = axes[1, idx]
    random_filter = np.random.randn(3, 3)
    im = ax.imshow(random_filter, cmap='RdBu')
    ax.set_title(f'Learned Filter {idx+1}', fontsize=12, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('conv2d_filters.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: conv2d_filters.png")
plt.show()


"""
Why CNN for images?

✓ Local connectivity: Filters only look at small regions
✓ Weight sharing: Same filter used across image
✓ Translation invariance: Detects features anywhere
✓ Efficient: Fewer parameters than dense layer
✓ Hierarchical: Learn edges → shapes → objects

Example comparison:
  Dense(3×224×224 image → 512):
    = 3 × 224 × 224 × 512 = 76 million parameters
  
  Conv(3 → 32, 3×3 kernel):
    = 32 × 3 × 3 × 3 = 864 parameters
    50,000x fewer!
"""


