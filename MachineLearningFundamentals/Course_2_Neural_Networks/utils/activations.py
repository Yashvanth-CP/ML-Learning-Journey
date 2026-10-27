"""
Activation Functions Utility
 
All the activation functions I need for neural networks.
Each has forward and backward (gradient) implementation.
"""


import numpy as np

# RELU 

def relu(z):
    return np.maximum(0, z)

"""
ReLU(Rectified Linear Unit)
Formula: f(z) = max(0,z)
why : Introduction non-linearity, most common activation
Args:
    z: Input (any shape)
    
Returns :
    Output after reLU(same shape as input)
 Example:
    >>> relu(np.array([-2,-1,0,1,2]))
    array([0,0,0,1,2])
    """

def relu_derivative(z):
    return (z > 0).astype(float)
"""
    Derivative of ReLU for backpropagation
    
    Formula: f'(z) = 1 if z > 0, else 0
    
    Args:
        z: Input (any shape)
    
    Returns:
        Gradient (1 if positive, 0 if negative)
    
    Example:
        >>> relu_derivative(np.array([-2, -1, 0, 1, 2]))
        array([0, 0, 0, 1, 1])
    """

# Sigmoid

def sigmoid(z):
    z = np.clip(z,-500,500)
    return 1/(1 + np.exp(-z))

"""
    Sigmoid function
    
    Formula: σ(z) = 1 / (1 + e^(-z))
    
    Why: Converts any number to probability (0-1)
    
    Args:
        z: Input (any shape)
    
    Returns:
        Output between 0 and 1
    
    Example:
        >>> sigmoid(np.array([-5, 0, 5]))
        array([0.0067, 0.5, 0.9933])
    """

def sigmoid_derivative(a):
    return a * (1 - a)

"""
    Derivative of sigmoid for backpropagation
    
    Formula: σ'(a) = a * (1 - a), where a = sigmoid(z)
    
    Args:
        a: Sigmoid output (already applied sigmoid)
    
    Returns:
        Gradient
    """


# tanh

def tanh(z):
    return np.tanh(z)

    """
    Tanh (Hyperbolic Tangent)
    
    Formula: tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    
    Why: Like sigmoid but outputs -1 to 1 (zero-centered)
    
    Args:
        z: Input (any shape)
    
    Returns:
        Output between -1 and 1
    
    Example:
        >>> tanh(np.array([-5, 0, 5]))
        array([-0.9999, 0, 0.9999])
    """  

def tanh_derivative(a):
    return 1 - np.power(a, 2)

"""
    Derivative of tanh
    
    Formula: tanh'(a) = 1 - a^2, where a = tanh(z)
    
    Args:
        a: Tanh output
    
    Returns:
        Gradient
    """

def linear(z):
    return z

    """
    Linear activation (no activation)
    
    Formula: f(z) = z
    
    Why: Used in output layer for regression
    
    Args:
        z: Input
    
    Returns:
        Same as input
    """

def linear_derivative(z):
    """
    Derivative of linear (always 1)
    """
    return np.ones_like(z)

# Softmax

def softmax(z):
     # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
"""
    Softmax (Multi-class probability)
    
    Formula: σ(z_i) = e^(z_i) / Σ(e^(z_j))
    
    Why: Convert logits to probabilities for multi-class
    
    Args:
        z: Input (can be 2D for batch)
    
    Returns:
        Probabilities that sum to 1
    
    Example:
        >>> softmax(np.array([[1, 2, 3]]))
        array([[0.09, 0.24, 0.67]])  # Sums to 1
    """

#  Leaky ReLU

def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)

"""
    Leaky ReLU (improved ReLU)
    
    Formula: f(z) = z if z > 0, else alpha*z
    
    Why: Allows small negative gradients (fixes dying ReLU)
    
    Args:
        z: Input
        alpha: Slope for negative values (default 0.01)
    
    Returns:
        Output
    """
def leaky_relu_derivative(z, alpha=0.01):
    """
    Derivative of Leaky ReLU
    """
    return np.where(z > 0, 1, alpha)

# TESTING
 
if __name__ == "__main__":
    print("="*50)
    print("Testing Activation Functions")
    print("="*50)
    
    # Test data
    z = np.array([-2, -1, 0, 1, 2])
    
    print(f"\nInput: {z}")
    print(f"ReLU:     {relu(z)}")
    print(f"Sigmoid:  {np.round(sigmoid(z), 4)}")
    print(f"Tanh:     {np.round(tanh(z), 4)}")
    
    print("\n" + "="*50)
    print("Testing Derivatives")
    print("="*50)
    
    print(f"\nInput: {z}")
    print(f"ReLU derivative:    {relu_derivative(z)}")
    print(f"Sigmoid derivative: {np.round(sigmoid_derivative(sigmoid(z)), 4)}")
    
    print("\n" + "="*50)
    print("Testing Softmax")
    print("="*50)
    
    z_multi = np.array([[1, 2, 3], [4, 5, 6]])
    softmax_out = softmax(z_multi)
    print(f"\nInput shape: {z_multi.shape}")
    print(f"Output:\n{np.round(softmax_out, 4)}")
    print(f"Sum per row: {np.sum(softmax_out, axis=1)}")  # Should be [1, 1]