"""
Evaluation Metrics Utility

How to measure if your model is working well.
Different metrics for different problems.
"""

import numpy as np


# ============ CONFUSION MATRIX ============

def confusion_matrix(y_true, y_pred):
    """
    Create confusion matrix
    
    Shows:
        TN (True Negative):  Correctly predicted 0
        FP (False Positive): Incorrectly predicted 1
        FN (False Negative): Incorrectly predicted 0
        TP (True Positive):  Correctly predicted 1
    
    Args:
        y_true: Actual labels (binary)
        y_pred: Predicted labels
    
    Returns:
        Confusion matrix
    
    Example:
        >>> y_true = [0, 1, 1, 0]
        >>> y_pred = [0, 1, 0, 0]
        >>> cm = confusion_matrix(y_true, y_pred)
        >>> print(cm)
        [[2 0]
         [1 1]]
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    return np.array([[tn, fp], [fn, tp]])


# ============ ACCURACY ============

def accuracy(y_true, y_pred):
    """
    Accuracy
    
    Formula: (TP + TN) / (TP + TN + FP + FN)
    
    What: Percentage of correct predictions
    
    When to use: When classes are balanced
    
    Good if: > 0.9 (90%)
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy (0-1)
    """
    return np.mean(y_pred == y_true)


# ============ PRECISION ============

def precision(y_true, y_pred):
    """
    Precision
    
    Formula: TP / (TP + FP)
    
    What: Of things we predicted positive, how many were correct?
    
    When to use: When false positives are costly
    Example: Spam detection (don't flag good emails)
    
    Good if: > 0.9
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
    
    Returns:
        Precision (0-1)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    
    if (tp + fp) == 0:
        return 0.0
    
    return tp / (tp + fp)


# ============ RECALL ============

def recall(y_true, y_pred):
    """
    Recall (Sensitivity, True Positive Rate)
    
    Formula: TP / (TP + FN)
    
    What: Of actual positives, how many did we find?
    
    When to use: When false negatives are costly
    Example: Disease detection (don't miss sick people)
    
    Good if: > 0.9
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
    
    Returns:
        Recall (0-1)
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.flatten()
    
    if (tp + fn) == 0:
        return 0.0
    
    return tp / (tp + fn)


# ============ F1 SCORE ============

def f1_score(y_true, y_pred):
    """
    F1 Score
    
    Formula: 2 * (Precision * Recall) / (Precision + Recall)
    
    What: Balanced metric between precision and recall
    
    When to use: Imbalanced datasets, want balance
    
    Good if: > 0.85
    
    Args:
        y_true: Actual labels
        y_pred: Predicted labels
    
    Returns:
        F1 score (0-1)
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    
    if (p + r) == 0:
        return 0.0
    
    return 2 * (p * r) / (p + r)


# ============ ROC AUC ============

def roc_auc_score(y_true, y_pred_proba):
    """
    ROC AUC Score
    
    Formula: Area under ROC curve
    
    What: Quality of probability predictions
    
    Range: 0-1
    - 0.5: Random
    - 0.9+: Excellent
    
    When to use: Imbalanced data, probability judgments
    
    Args:
        y_true: Actual labels (0 or 1)
        y_pred_proba: Predicted probabilities (0-1)
    
    Returns:
        AUC score (0-1)
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    # Sort by probability
    sorted_indices = np.argsort(y_pred_proba)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    # Count positives
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Calculate TPR and FPR at each threshold
    tpr = np.cumsum(y_true_sorted) / n_pos
    fpr = np.cumsum(1 - y_true_sorted) / n_neg
    
    # AUC is area under TPR vs FPR curve
    auc = np.trapz(tpr, fpr)
    
    return auc


# ============ REGRESSION METRICS ============

def r2_score(y_true, y_pred):
    """
    R² Score
    
    Formula: 1 - (SS_res / SS_tot)
    
    What: Proportion of variance explained
    
    Range: -∞ to 1
    - 1.0: Perfect prediction
    - 0.0: Predicts mean
    - negative: Worse than mean
    
    When to use: Regression evaluation
    
    Good if: > 0.8
    
    Args:
        y_true: Actual values
        y_pred: Predictions
    
    Returns:
        R² score
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    
    Formula: sqrt((1/m) * Σ(y_pred - y_true)²)
    
    What: Average prediction error (in original units)
    
    When to use: Regression with scale interpretation
    
    Good if: Low (relative to target range)
    """
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


# ============ CLASSIFICATION REPORT ============

def classification_report(y_true, y_pred):
    """
    Print complete classification report
    
    Shows: Accuracy, Precision, Recall, F1
    """
    acc = accuracy(y_true, y_pred)
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print("="*50)
    print("CLASSIFICATION REPORT")
    print("="*50)
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\nConfusion Matrix:")
    print(f"           Predicted_0  Predicted_1")
    print(f"Actual_0:  {cm[0,0]:^12}  {cm[0,1]:^12}")
    print(f"Actual_1:  {cm[1,0]:^12}  {cm[1,1]:^12}")
    print("="*50)


# ============ TESTING ============

if __name__ == "__main__":
    print("="*50)
    print("Testing Evaluation Metrics")
    print("="*50)
    
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.2, 0.85, 0.6, 0.15, 0.1, 0.95, 0.9])
    
    print("\nTest Data:")
    print(f"True:  {y_true}")
    print(f"Pred:  {y_pred}")
    
    print("\n" + "-"*50)
    classification_report(y_true, y_pred)
    
    print(f"\nROC AUC: {roc_auc_score(y_true, y_pred_proba):.4f}")
    
    # Regression test
    print("\n" + "="*50)
    print("Regression Metrics")
    print("="*50)
    
    y_true_reg = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred_reg = np.array([1.1, 2.2, 2.9, 4.1, 4.9])
    
    print(f"RMSE: {rmse(y_true_reg, y_pred_reg):.4f}")
    print(f"R²:   {r2_score(y_true_reg, y_pred_reg):.4f}")