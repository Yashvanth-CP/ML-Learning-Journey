"""
12. LAYER COMBINATIONS
=====================

Build complete neural networks by combining layers!
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("COMBINING LAYERS: Complete Networks")
print("="*70)


print("\n1. LAYER COMBINATION BASICS")
print("-"*70)

print("""
Neural Network = Stack of Layers

Each layer:
  - Takes input
  - Transforms it
  - Passes to next layer

Example:
  Input (784) → Dense(128) → ReLU → Dropout(0.5)
             → Dense(64) → ReLU → Dropout(0.5)
             → Dense(10) → Softmax
             → Output (10 classes)

Each arrow is a layer!
""")


class SimpleNetwork:
    """Complete network from scratch"""
    
    def __init__(self):
        print("Building network...\n")
        
        # Layer 1
        self.W1 = np.random.randn(784, 128) * 0.01
        self.b1 = np.zeros((1, 128))
        print("Layer 1: Dense(784 → 128)")
        
        # Layer 2
        self.W2 = np.random.randn(128, 64) * 0.01
        self.b2 = np.zeros((1, 64))
        print("Layer 2: Dense(128 → 64)")
        
        # Layer 3
        self.W3 = np.random.randn(64, 10) * 0.01
        self.b3 = np.zeros((1, 10))
        print("Layer 3: Dense(64 → 10)")
        
        print("\n✓ Network ready!")
    
    def forward(self, X):
        """Forward through all layers"""
        # Layer 1: Dense + ReLU
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = np.maximum(0, Z1)
        
        # Layer 2: Dense + ReLU
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = np.maximum(0, Z2)
        
        # Layer 3: Dense + Softmax
        Z3 = np.dot(A2, self.W3) + self.b3
        
        # Softmax
        exp_Z3 = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
        A3 = exp_Z3 / np.sum(exp_Z3, axis=1, keepdims=True)
        
        return A3


print("\n2. NETWORK CREATED")
print("-"*70)

network = SimpleNetwork()


print("\n3. ARCHITECTURE PATTERNS")
print("-"*70)

print("""
Pattern 1: Linear Stack (MLP)
  Input → Dense → ReLU → Dense → ReLU → Dense → Output
  Use for: Tabular data, classification

Pattern 2: CNN for Images
  Input → Conv → Pool → Conv → Pool → Flatten → Dense → Output
  Use for: Image classification

Pattern 3: Sequential with Dropout
  Input → Dense → ReLU → Dropout → Dense → ReLU → Dropout → Output
  Use for: Prevent overfitting on small datasets

Pattern 4: ResNet (Skip Connections)
  Input → Dense → ReLU ⤺
       └────────────→ Add → Dense → Output
  Use for: Very deep networks, stable gradients
""")


print("\n4. COMMON ARCHITECTURES")
print("-"*70)

architectures = {
    'MLP': {
        'layers': ['Input', 'Dense(512)', 'ReLU', 'Dense(256)', 'ReLU', 'Dense(10)', 'Softmax'],
        'use': 'Simple classification'
    },
    'MLP with Dropout': {
        'layers': ['Input', 'Dense(512)', 'ReLU', 'Dropout', 'Dense(256)', 'ReLU', 'Dropout', 'Dense(10)'],
        'use': 'Prevent overfitting'
    },
    'VGG': {
        'layers': ['Input', 'Conv(64)', 'Conv(64)', 'Pool', 'Conv(128)', 'Conv(128)', 'Pool', 'Flatten', 'Dense(1000)'],
        'use': 'Image classification'
    },
    'ResNet-like': {
        'layers': ['Input', 'Dense(256)', '↓', 'Dense(256)', '↑', 'Add', 'Dense(10)'],
        'use': 'Deep networks'
    }
}

for name, arch in architectures.items():
    print(f"\n{name}:")
    print(f"  Use: {arch['use']}")
    print(f"  Layers: {' → '.join(arch['layers'])}")


print("\n5. BUILDING BLOCKS")
print("-"*70)

print("""
Dense Block:
  Dense → BatchNorm → ReLU

Conv Block:
  Conv → BatchNorm → ReLU → Dropout

Residual Block:
         ↓
  Dense → ReLU
         ↓
  Dense ⤺
         ↓
       Add → ReLU
         ↑
  Original input (skip)
""")


print("\n6. PARAMETER SHARING PATTERNS")
print("-"*70)

print("""
Dense Layers: Each layer has unique parameters
  Dense(100 → 50): 100×50 + 50 = 5,050 parameters
  Dense(50 → 25):  50×25 + 25 = 1,275 parameters
  Dense(25 → 10):  25×10 + 10 = 260 parameters
  Total: 6,585 parameters

Conv Layers: Parameters shared across space
  Conv(3→32, 3×3): 3×3×3×32 + 32 = 896 parameters
  But applied to entire image!
  Much more efficient than Dense
""")


print("\n7. LAYER STACKING RULES")
print("-"*70)

print("""
✓ DO:
  - Use ReLU in hidden layers
  - Use BatchNorm before activation
  - Use Dropout for regularization
  - Use appropriate output activation
  - Stack similar-sized layers
  
✗ DON'T:
  - Use sigmoid in hidden layers (slow gradients)
  - Use Dropout after output
  - Stack Conv directly to Dense (use Flatten!)
  - Forget activation functions
  - Use output activation in hidden layers
""")


print("\n8. VISUALIZATION: NETWORK STRUCTURE")
print("-"*70)

fig, ax = plt.subplots(figsize=(12, 8))

# Network diagram
layers_info = [
    ('Input', 784, 'gray'),
    ('Dense', 128, 'blue'),
    ('ReLU', 128, 'lightblue'),
    ('Dropout', 128, 'yellow'),
    ('Dense', 64, 'blue'),
    ('ReLU', 64, 'lightblue'),
    ('Dropout', 64, 'yellow'),
    ('Dense', 10, 'red'),
    ('Softmax', 10, 'pink'),
]

x_pos = np.linspace(0, 9, len(layers_info))
layer_height = 100

for i, (name, size, color) in enumerate(layers_info):
    # Draw box
    rect = plt.Rectangle((i-0.4, size/2-50), 0.8, layer_height, 
                         color=color, alpha=0.6, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Label
    ax.text(i, size/2, f'{name}\n({size})', ha='center', va='center',
           fontsize=10, fontweight='bold')
    
    # Arrow to next layer
    if i < len(layers_info) - 1:
        ax.arrow(i+0.4, 0, 0.15, 0, head_width=10, head_length=0.1, fc='black', ec='black')

ax.set_xlim(-1, len(layers_info))
ax.set_ylim(-100, 200)
ax.axis('off')
ax.set_title('Neural Network Architecture', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('network_architecture.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: network_architecture.png")
plt.show()


print("\n9. KEY INSIGHTS")
print("-"*70)

print("""
✓ Networks are just stacked layers
✓ Each layer transforms data
✓ Activation functions add non-linearity
✓ Regularization prevents overfitting
✓ Order of layers matters!
✓ Match input/output dimensions
✓ Test incrementally (layer by layer)

Debugging tip:
  If something breaks:
  1. Check dimensions first!
  2. Then check activations
  3. Then check loss computation
  4. Finally, debug training loop
""")


print("\n" + "="*70)
print("✓ Now ready for real projects!")
print("="*70)