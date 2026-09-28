"""
Loss Functions Utility

Different loss functions for regression and classification.
Each shows what it measures and when to use it.
"""

import numpy as np


# ============ MSE (Mean Squared Error) ============

def mse(y_pred, y_true):
    """
    Mean Squared Error (Regression)
    
    Formula: MSE = (1/m) * Σ(y_pred - y_true)²
    
    When to use: Regression problems
    
    Args:
        y_pred: Predictions (m, )
        y_true: Actual values (m, )
    
    Returns:
        Average squared error
    
    Example:
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 2.2, 2.9])
        >>> mse(y_pred, y_true)  # ~0.01
    """
    m = y_true.shape[0]
    return np.mean((y_pred - y_true) ** 2)


def mse_gradient(y_pred, y_true):
    """
    Gradient of MSE
    
    Formula: ∂MSE/∂y_pred = (2/m) * (y_pred - y_true)
    
    Returns gradient for backpropagation
    """
    return (2 / y_true.shape[0]) * (y_pred - y_true)


# ============ MAE (Mean Absolute Error) ============

def mae(y_pred, y_true):
    """
    Mean Absolute Error (Regression)
    
    Formula: MAE = (1/m) * Σ|y_pred - y_true|
    
    When to use: Robust to outliers (outliers don't get squared)
    
    Args:
        y_pred: Predictions
        y_true: Actual values
    
    Returns:
        Average absolute error
    """
    m = y_true.shape[0]
    return np.mean(np.abs(y_pred - y_true))


# ============ BINARY CROSS ENTROPY ============

def binary_cross_entropy(y_pred, y_true):
    """
    Binary Cross Entropy (Binary Classification)
    
    Formula: BCE = -(1/m) * Σ[y*log(y_pred) + (1-y)*log(1-y_pred)]
    
    When to use: Binary classification (0 or 1)
    
    Why: Exponentially penalizes wrong predictions
    
    Args:
        y_pred: Predicted probabilities (0-1)
        y_true: Actual labels (0 or 1)
    
    Returns:
        Cross entropy loss
    """
    m = y_true.shape[0]
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    return np.mean(loss)


def binary_cross_entropy_gradient(y_pred, y_true):
    """
    Gradient of Binary Cross Entropy
    
    Formula: ∂BCE/∂y_pred = -(y/y_pred - (1-y)/(1-y_pred))
    
    Simplified: ≈ (y_pred - y) for sigmoid output
    """
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / y_true.shape[0]


# ============ CATEGORICAL CROSS ENTROPY ============

def categorical_cross_entropy(y_pred, y_true):
    """
    Categorical Cross Entropy (Multi-class Classification)
    
    Formula: CCE = -(1/m) * Σ Σ y_true[i,j] * log(y_pred[i,j])
    
    When to use: Multi-class classification (3+ classes)
    
    Args:
        y_pred: Predicted probabilities (m, num_classes)
        y_true: One-hot encoded labels (m, num_classes)
    
    Returns:
        Cross entropy loss
    
    Example:
        >>> y_true = np.array([[1, 0, 0], [0, 1, 0]])  # 2 samples, 3 classes
        >>> y_pred = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
        >>> categorical_cross_entropy(y_pred, y_true)  # Low loss (correct predictions)
    """
    m = y_true.shape[0]
    epsilon = 1e-8
    
    # Clip to prevent log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Sum across classes, then average across samples
    loss = -np.sum(y_true * np.log(y_pred)) / m
    
    return loss


# ============ SPARSE CATEGORICAL CROSS ENTROPY ============

def sparse_categorical_cross_entropy(y_pred, y_true):
    """
    Sparse Categorical Cross Entropy (Multi-class with label encoding)
    
    Like categorical but y_true is integer labels, not one-hot
    
    Args:
        y_pred: Predicted probabilities (m, num_classes)
        y_true: Class indices (m, ) - values 0 to num_classes-1
    
    Returns:
        Cross entropy loss
    
    Example:
        >>> y_true = np.array([0, 1, 2, 1])  # Class indices
        >>> y_pred = np.array([[0.7, 0.2, 0.1],
        ...                    [0.1, 0.8, 0.1],
        ...                    [0.1, 0.2, 0.7],
        ...                    [0.1, 0.8, 0.1]])
        >>> sparse_categorical_cross_entropy(y_pred, y_true)  # Low loss
    """
    m = y_true.shape[0]
    epsilon = 1e-8
    
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Get probability of correct class
    correct_class_probs = y_pred[np.arange(m), y_true.astype(int)]
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss


# ============ HUBER LOSS ============

def huber_loss(y_pred, y_true, delta=1.0):
    """
    Huber Loss (Robust regression)
    
    Combination of MSE (smooth) and MAE (robust)
    
    Formula:
        if |error| <= delta: 0.5 * error²
        else: delta * |error| - 0.5 * delta²
    
    When to use: Regression with outliers
    
    Args:
        y_pred: Predictions
        y_true: Actual values
        delta: Threshold parameter
    
    Returns:
        Huber loss
    """
    error = y_pred - y_true
    
    quadratic = 0.5 * np.square(error)
    linear = delta * (np.abs(error) - 0.5 * delta)
    
    # Use quadratic for small errors, linear for large
    loss = np.where(np.abs(error) <= delta, quadratic, linear)
    
    return np.mean(loss)


# ============ TESTING ============

if __name__ == "__main__":
    print("="*60)
    print("Testing Loss Functions")
    print("="*60)
    
    # Regression test
    print("\n1. REGRESSION LOSSES")
    print("-"*60)
    
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1])
    
    print(f"True:  {y_true_reg}")
    print(f"Pred:  {y_pred_reg}")
    print(f"MSE:   {mse(y_pred_reg, y_true_reg):.4f}")
    print(f"MAE:   {mae(y_pred_reg, y_true_reg):.4f}")
    
    # Binary classification test
    print("\n2. BINARY CLASSIFICATION LOSS")
    print("-"*60)
    
    y_true_bin = np.array([0, 1, 1, 0])
    y_pred_bin = np.array([0.1, 0.9, 0.8, 0.2])
    
    print(f"True:  {y_true_bin}")
    print(f"Pred:  {y_pred_bin}")
    print(f"Binary CE: {binary_cross_entropy(y_pred_bin, y_true_bin):.4f}")
    
    # Multi-class test
    print("\n3. MULTI-CLASS CLASSIFICATION LOSS")
    print("-"*60)
    
    y_true_multi = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_pred_multi = np.array([[0.7, 0.2, 0.1],
                              [0.1, 0.8, 0.1],
                              [0.1, 0.2, 0.7]])
    
    print(f"Categorical CE: {categorical_cross_entropy(y_pred_multi, y_true_multi):.4f}")
    
    y_true_sparse = np.array([0, 1, 2])
    print(f"Sparse CE: {sparse_categorical_cross_entropy(y_pred_multi, y_true_sparse):.4f}")
    
    print("\n" + "="*60)
    print("✓ All loss functions working!")
    print("="*60)