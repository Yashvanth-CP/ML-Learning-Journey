"""
01. ACTIVATION FUNCTIONS
"""

import numpy as np
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