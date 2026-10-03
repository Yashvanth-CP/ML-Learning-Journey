"""
SOFTMAX & MULTI-CLASS CLASSIFICATION

Convert network outputs to probabilities for >2 classes
Essential for MNIST, image classification, etc

Sigmoid (Binary):
  σ(z) = 1 / (1 + e^(-z))


Softmax (Multi-class):
  softmax(z_i) = e^(z_i) / Σ(e^(z_j))
  
Key difference:
  - Sigmoid: Independent outputs
  - Softmax: Outputs sum to 1 (true probability distribution)
"""
import numpy as np
import matplotlib.pyplot as plt

Z = np.array([2.0, 1.0, 0.1])

# softmax (numerically stable)

Z_adjusted = Z - np.max(Z)
exp_z = np.exp(Z_adjusted)
softmax = exp_z / np.sum(exp_z)

print(f"\n After softmax: ")
for i, prob in enumerate(softmax):
    print(f" Class {i}: {prob :.4f} ({prob*100:.1f}%)")

print(f"\n Sum of probabilities: {np.sum(softmax):.4f}(should be 1.0) ")

print("SOFTMAX VISUALIZATION")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
 
# Plot 1: vary ONE logit, keep the others fixed -> watch all 3 probabilities move
ax = axes[0]
z0 = np.linspace(-5, 5, 200)
logits = np.stack([z0, np.ones_like(z0), np.zeros_like(z0)], axis=1)   # class0 varies, class1 = 1, class2 = 0
e = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = e / e.sum(axis=1, keepdims=True)
for k in range(3):
    ax.plot(z0, probs[:, k], linewidth=2, label=f'Class {k}')
ax.set_xlabel('Logit of class 0 (class 1 fixed at 1, class 2 at 0)')
ax.set_ylabel('Probability')
ax.set_title('Raising one logit steals probability\nfrom the others (sum is always 1)', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
 
# Plot 2: all 3 class probabilities for 3 different logit vectors
ax = axes[1]
outputs = [np.array([1, 1, 1]), np.array([2, 1, 0]), np.array([5, 1, 0])]
labels = ['Equal\n[1,1,1]', 'Medium\n[2,1,0]', 'Big gap\n[5,1,0]']
width = 0.25
for k in range(3):
    vals = []
    for output in outputs:
        zz = output - np.max(output)
        vals.append(np.exp(zz[k]) / np.sum(np.exp(zz)))
    ax.bar(np.arange(3) + (k - 1) * width, vals, width, label=f'Class {k}')
ax.set_xticks(np.arange(3)); ax.set_xticklabels(labels)
ax.set_ylabel('Probability'); ax.set_ylim([0, 1])
ax.set_title('Bigger logit gaps = more confident softmax', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3, axis='y')
 
plt.tight_layout()
plt.savefig('softmax_visualization.png', dpi=300, bbox_inches='tight')
print("✓ Saved: softmax_visualization.png")
plt.show()

"""
Cross-Entropy Loss (Categorical):
  Loss = -Σ y_i * log(ŷ_i)

Perfect prediction (0.999):
  Loss = -log(0.999) ≈ 0.001
 
Wrong prediction (0.001):
  Loss = -log(0.001) ≈ 6.9

"""
def softmax(Z):
    Z_adj = Z - np.max(Z, axis=1, keepdims=True)
    exp_z = np.exp(Z_adj)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    m = len(y_true)
    loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
    return loss

# Example
z = np.array([[2.0, 1.0, 0.1],
              [1.0, 3.0, 0.5]])
 
y_pred = softmax(z)
print(f"Softmax output:\n{y_pred}\n")
 
y_true = np.array([[0, 0, 1],
                   [0, 1, 0]])
 
loss = cross_entropy_loss(y_true, y_pred)
print(f"Cross-entropy loss: {loss:.4f}")

"""
Softmax derivative (surprisingly simple!):
 
∂softmax_i / ∂z_j = softmax_i * (δ_ij - softmax_j)
 
"""