"""
CONVOLUTIONAL NEURAL NETWORKS (CNN)
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Problem: Image of 28×28 = 784 features
 
Dense Layer (Bad for images):
  Input: 784 neurons
  Hidden: 128 neurons
  Weights: 784 X 128 = 100K parameters!
  Problem: Doesn't understand spatial structure
          Treats pixel at (0,0) same as (27,27)
          Wastes parameters on position info
 
CNN (Good for images):
  Filter: 3X3 = 9 parameters
  Share filters across image
  32 such filters: 32*9 + 32 = 320 parameters (vs 784x128 = 100,352 for ONE dense layer)
  Benefit: Learns local features (edges, corners)
           Translation invariant
           Much more efficient!
"""

# Simple convolution example
input_img = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])
 
# Edge detection filter
edge_filter = np.array([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])


def convolve2d(img, filt, stride=1):
    h, w = img.shape
    fh, fw = filt.shape
    output_h = (h - fh) // stride + 1
    output_w = (w - fw) // stride + 1
    output = np.zeros((output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            patch = img[i * stride:i*stride + fh, j*stride:j*stride+fw]
            output[i, j] = np.sum(patch * filt)

    return output

output = convolve2d(input_img, edge_filter)

print(f"Input: {input_img.shape}")
print(f"Filter: {edge_filter.shape}")
print(f"Output: {output.shape}")
print(f"\nOutput:\n{output}")

"""
Filter (Kernel):
  Small matrix (3×3, 5×5)
  Learns to detect features (edges, textures, shapes)
  Applied to entire image (parameter sharing!)
 
Stride:
  How much to move filter each step
  Stride=1: Move 1 pixel
  Stride=2: Move 2 pixels (reduces output size)
 
Padding:
  Add zeros around image
  Keeps spatial dimensions
  "Same" padding: output same size as input
 
Activation Map (Feature Map):
  Output of convolution
  Passed through ReLU
  Next layer's input
 
Pooling:
  Max pooling: Take max in window
  Average pooling: Take average
  Reduces spatial size, keeps important features
"""

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
# Panel 1: parameters in ONE layer (computed)
ax = axes[0, 0]
dense_params = 784 * 128 + 128
conv_params = 3 * 3 * 1 * 32 + 32
ax.bar(['Dense layer\n784 -> 128', 'Conv layer\n32 filters 3x3'], [dense_params, conv_params], color=['tab:red', 'tab:green'])
for i, v in enumerate([dense_params, conv_params]):
    ax.text(i, v, f"{v:,}", ha='center', va='bottom', fontweight='bold')
ax.set_yscale('log'); ax.set_ylabel('parameters (log scale)')
ax.set_title('Parameters in ONE layer', fontweight='bold'); ax.grid(True, alpha=0.3, axis='y')
 
# Panels 2-4: a REAL convolution of a synthetic image with two edge filters
demo = np.zeros((16, 16)); demo[4:12, 4:12] = 1.0
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)   # reacts to VERTICAL edges
sobel_y = sobel_x.T                                                     # reacts to HORIZONTAL edges
out_v = convolve2d(demo, sobel_x)
out_h = convolve2d(demo, sobel_y)
 
ax = axes[0, 1]; ax.imshow(demo, cmap='gray'); ax.set_title('Input: 16x16 image with a bright square', fontweight='bold'); ax.axis('off')
ax = axes[1, 0]; ax.imshow(out_v, cmap='RdBu'); ax.set_title('Output of a VERTICAL-edge filter\n(left / right sides light up)', fontweight='bold'); ax.axis('off')
ax = axes[1, 1]; ax.imshow(out_h, cmap='RdBu'); ax.set_title('Output of a HORIZONTAL-edge filter\n(top / bottom sides light up)', fontweight='bold'); ax.axis('off')
 
plt.tight_layout()
plt.savefig('cnn_concepts.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cnn_concepts.png")
plt.show()
 
