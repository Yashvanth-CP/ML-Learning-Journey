"""
01. ACTIVATION FUNCTIONS
"""
import numpy as np
import matplotlib
# ← Add this
import matplotlib.pyplot as plt
 
print("="*60)
print("ACTIVATION FUNCTIONS: Learning the Fundamentals")
print("="*60)

# Linear Stacking

"""
Imagine we stack linear layers: 
    Layer 1: y₁ = w₁·x + b₁")
    Layer 2: y₂ = w₂·y₁ + b₂")
    = w₂·(w₁·x + b₁) + b₂")
    = (w₂·w₁)·x + (w₂·b₁ + b₂)")
    = W·x + B  (still just linear!)")
 """

# ReLU 

print("\n\n2. RELU (Rectified Linear Unit)")

def relu(z):
    """ReLU: max(0, z)"""
    return np.maximum(0, z)
 
def relu_derivative(z):
    """Gradient of ReLU"""
    return (z > 0).astype(float)


"""
Formula: f(z) = max(0, z)
      - If z > 0: output = z
      - If z ≤ 0: output = 0
      """

z_test = np.array([-2, -1, 0, 1, 2])
print(f"  Input:  {z_test}")
print(f"  Output: {relu(z_test)}")
 
print("\nGradient (for backpropagation):")
print(f"  Input:    {z_test}")
print(f"  Gradient: {relu_derivative(z_test)}")
print("  (1 if positive, 0 if negative)")

# SIGMOID 

print("\n\n3. SIGMOID")
print("-"*60)
 
def sigmoid(z):
    """Sigmoid: 1 / (1 + e^(-z))"""
    z = np.clip(z, -500, 500)  # Prevent overflow
    return 1 / (1 + np.exp(-z))
 
def sigmoid_derivative(a):
    """Gradient of sigmoid (a is sigmoid output)"""
    return a * (1 - a)


"""
Formula: σ(z) = 1 / (1 + e^(-z))
"""

z_test = np.array([-5, -1, 0, 1, 5])
print(f"  Input:  {z_test}")
print(f"  Output: {np.round(sigmoid(z_test), 4)}")
 
print("\nProbability interpretation:")
print("  z=-5  → σ=0.0067  (almost 0%, definitely negative class)")
print("  z=0   → σ=0.5000  (50-50)")
print("  z=5   → σ=0.9933  (almost 100%, definitely positive class)")


# TANH


def tanh_custom(z):
    """Tanh: (e^z - e^(-z)) / (e^z + e^(-z))"""
    return np.tanh(z)
 
def tanh_derivative(a):
    """Gradient of tanh (a is tanh output)"""
    return 1- np.power(a, 2)

"""
Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
"""
z_test = np.array([-5, -1, 0, 1, 5])
print(f"  Input:  {z_test}")
print(f"  Output: {np.round(tanh_custom(z_test), 4)}")
 
print("\nComparison with sigmoid:")
print("  Sigmoid:  outputs 0 to 1")
print("  Tanh:     outputs -1 to 1 (zero-centered)")

# VISUALIZATION 

 
# Create range of inputs
z = np.linspace(-5, 5, 100)
 
# Calculate outputs
relu_out = relu(z)
sigmoid_out = sigmoid(z)
tanh_out = tanh_custom(z)
linear_out = z
 
# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 
# ReLU
axes[0, 0].plot(z, relu_out, 'b-', linewidth=2, label='ReLU')
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title('ReLU: f(z) = max(0, z)', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('z')
axes[0, 0].set_ylabel('f(z)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
 
# Sigmoid
axes[0, 1].plot(z, sigmoid_out, 'g-', linewidth=2, label='Sigmoid')
axes[0, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='y=0.5')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Sigmoid: σ(z) = 1/(1+e^-z)', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('z')
axes[0, 1].set_ylabel('f(z)')
axes[0, 1].set_ylim([-0.1, 1.1])
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
 
# Tanh
axes[1, 0].plot(z, tanh_out, 'r-', linewidth=2, label='Tanh')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=-1, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Tanh: tanh(z) = (e^z - e^-z)/(e^z + e^-z)', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('z')
axes[1, 0].set_ylabel('f(z)')
axes[1, 0].set_ylim([-1.1, 1.1])
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
 
# Comparison
axes[1, 1].plot(z, relu_out, 'b-', linewidth=2, label='ReLU')
axes[1, 1].plot(z, sigmoid_out, 'g-', linewidth=2, label='Sigmoid')
axes[1, 1].plot(z, tanh_out, 'r-', linewidth=2, label='Tanh')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Comparison of All Activations', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('z')
axes[1, 1].set_ylabel('f(z)')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
 
plt.tight_layout()
plt.savefig('activation_functions_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: activation_functions_visualization.png")
plt.show()



# PRACTICAL USAGE 

"""
Network Architecture Example:
 
Input (raw numbers)
       ↓
Hidden Layer 1 + ReLU (adds non-linearity)
       ↓
Hidden Layer 2 + ReLU (adds non-linearity)
       ↓
Output Layer + Sigmoid (converts to probability)
 
Why this order?
- ReLU in hidden layers: Fast, learns complex patterns
- Sigmoid in output: Converts to probability (0-1)
"""

# KEY INSIGHTS

"""
1. WHY ACTIVATION FUNCTIONS?
   - Without them: Multiple layers = still linear = can't learn complex patterns
   - With them: Can learn any pattern (universal approximation)
 
2. WHY RELU MOST POPULAR?
   - Simple: max(0, z)
   - Fast to compute
   - Works very well empirically
   - Solves vanishing gradient problem (compared to sigmoid)
 
3. GRADIENT FLOWS
   - Sigmoid gradient: Can vanish (→0) for extreme values
   - Tanh gradient: Similar issue but better
   - ReLU gradient: Constant (1 or 0) - doesn't vanish!
 
4. CHOOSING ACTIVATION
   - Hidden layers: ReLU (sometimes Tanh)
   - Output (binary classification): Sigmoid
   - Output (multi-class): Softmax
   - Output (regression): Linear (no activation)
"""

# DERIVATIVES
 
print("\n9. DERIVATIVES (For Backpropagation)")
print("-"*60)
 
z_test = np.array([-2, -1, 0, 1, 2])
 
print("\nReLU Derivative:")
print(f"  z:              {z_test}")
print(f"  ReLU(z):        {relu(z_test)}")
print(f"  ReLU'(z):       {relu_derivative(z_test)}")
print("  → 1 if z > 0, else 0")
 
print("\nSigmoid Derivative:")
sigmoid_z = sigmoid(z_test)
print(f"  z:              {z_test}")
print(f"  σ(z):           {np.round(sigmoid_z, 4)}")
print(f"  σ'(σ(z)):       {np.round(sigmoid_derivative(sigmoid_z), 4)}")
print("  → σ(z) * (1 - σ(z))")
 
print("\n✓ These gradients are crucial for backpropagation!")


#  SUMMARY 

"""
✓ Activation functions add non-linearity to neural networks
✓ Without them: stacking layers is useless (still linear)
✓ ReLU: f(z) = max(0, z) - most popular
✓ Sigmoid: σ(z) = 1/(1+e^-z) - for binary classification output
✓ Tanh: zero-centered version of sigmoid
✓ Derivatives needed for backpropagation
 
NEXT STEP: Learn how to use these in a single neuron!
"""