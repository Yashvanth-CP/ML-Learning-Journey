"""
07. DENSE LAYER (Fully Connected Layer)
"""

import numpy as np
import matplotlib.pyplot as plt

print("_-"*20)
print("DENSE LAYER: The Core Building Block")
print("-_"*20)


class DenseLayer:
    # fully connected layer from scratch
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize dense layer
        
        Args:
            input_size: Number of input features
            output_size: Number of neurons
            activation: 'relu' or 'sigmoid'
        """
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        
        # Initialize weights and biases
        self.W = np.random.randn(input_size, output_size) * 0.01
        self.b = np.zeros((1, output_size))
        
        print(f"Dense Layer created:")
        print(f"  Input:  {input_size}")
        print(f"  Output: {output_size}")
        print(f"  Weights shape: {self.W.shape}")
        print(f"  Bias shape: {self.b.shape}")
        print(f"  Activation: {activation}")
    
    def forward(self, X):
        """Forward pass"""
        self.Z = np.dot(X, self.W) + self.b
        
        if self.activation_name == 'relu':
            self.A = np.maximum(0, self.Z)
        elif self.activation_name == 'sigmoid':
            self.A = 1 / (1 + np.exp(-np.clip(self.Z, -500, 500)))
        else:
            self.A = self.Z
        
        self.X = X
        return self.A
    
    def backward(self, dA, learning_rate=0.01):
        """Backward pass"""
        m = self.X.shape[0]
        
        # Activation derivative
        if self.activation_name == 'relu':
            dZ = dA * (self.Z > 0)
        elif self.activation_name == 'sigmoid':
            dZ = dA * self.A * (1 - self.A)
        else:
            dZ = dA
        
        # Gradients
        dW = (1/m) * np.dot(self.X.T, dZ)
        db = (1/m) * np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, self.W.T)
        
        # Update
        self.W -= learning_rate * dW
        self.b -= learning_rate * db
        
        return dX
 
 
print("\n1. DENSE LAYER BASICS")
print("-"*70)
 
# Create layer
layer = DenseLayer(input_size=5, output_size=3, activation='relu')
 
# Forward pass
X_sample = np.random.randn(2, 5)  # 2 samples, 5 features
print(f"\nInput shape: {X_sample.shape}")
 
output = layer.forward(X_sample)
print(f"Output shape: {output.shape}")
print(f"✓ Forward pass works!")
 
# Backward pass
dA_sample = np.random.randn(2, 3)
dX = layer.backward(dA_sample, learning_rate=0.01)
print(f"Gradient shape: {dX.shape}")
print(f"✓ Backward pass works!")
 
 
print("\n2. STACKING DENSE LAYERS")
print("-"*70)
 
# Build simple network
layer1 = DenseLayer(10, 8, 'relu')
layer2 = DenseLayer(8, 4, 'relu')
layer3 = DenseLayer(4, 1, 'sigmoid')
 
print("\nNetwork architecture:")
print("  Input (10) → Dense(8) → Dense(4) → Dense(1)")
 
# Forward
X = np.random.randn(5, 10)
a1 = layer1.forward(X)
a2 = layer2.forward(a1)
a3 = layer3.forward(a2)
 
print(f"\nForward pass dimensions:")
print(f"  Input:  {X.shape}")
print(f"  After layer 1: {a1.shape}")
print(f"  After layer 2: {a2.shape}")
print(f"  Output: {a3.shape}")
 
 
print("\n3. PARAMETER EFFICIENCY")
print("-"*70)
 
def count_params(input_size, output_size):
    """Count parameters in dense layer"""
    weights = input_size * output_size
    biases = output_size
    return weights + biases
 
print("\nParameter count for different layer sizes:")
configs = [
    (10, 5),
    (10, 10),
    (100, 50),
    (1000, 512),
]
 
for inp, out in configs:
    params = count_params(inp, out)
    print(f"  Dense({inp} → {out}): {params:,} parameters")
 
 
print("\n" + "-+-"*30)
print("✓ Dense layers are the foundation of neural networks!")
print("-_-"*30)
 
