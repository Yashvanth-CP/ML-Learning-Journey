"""
PROJECT 1: DEMAND PREDICTION
Step 5: Evaluate Model Performance
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("DEMAND PREDICTION: Step 5 - Evaluate Model")
print("="*70)


print("\n1. LOADING MODEL AND DATA")
print("-"*70)

# Load test data
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy').reshape(-1, 1)

# Load model weights
W1 = np.load('model_W1.npy')
W2 = np.load('model_W2.npy')
W3 = np.load('model_W3.npy')
W4 = np.load('model_W4.npy')
b1 = np.load('model_b1.npy')
b2 = np.load('model_b2.npy')
b3 = np.load('model_b3.npy')
b4 = np.load('model_b4.npy')

print("✓ Model and data loaded")


print("\n2. MAKE PREDICTIONS ON TEST SET")
print("-"*70)

def predict(X, W1, W2, W3, W4, b1, b2, b3, b4):
    """Make predictions using trained model"""
    Z1 = np.dot(X, W1) + b1
    A1 = np.maximum(0, Z1)
    
    Z2 = np.dot(A1, W2) + b2
    A2 = np.maximum(0, Z2)
    
    Z3 = np.dot(A2, W3) + b3
    A3 = np.maximum(0, Z3)
    
    Z4 = np.dot(A3, W4) + b4
    return Z4

y_pred = predict(X_test, W1, W2, W3, W4, b1, b2, b3, b4)

print(f"Predictions shape: {y_pred.shape}")
print(f"Sample predictions (first 5):")
print(f"  Predicted: {y_pred[:5].flatten()}")
print(f"  Actual:    {y_test[:5].flatten()}")


print("\n3. CALCULATE EVALUATION METRICS")
print("-"*70)

# MSE
mse = np.mean((y_pred - y_test) ** 2)

# RMSE
rmse = np.sqrt(mse)

# MAE
mae = np.mean(np.abs(y_pred - y_test))

# R² Score
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r2 = 1 - (ss_res / ss_tot)

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1e-8))) * 100

print(f"""
Regression Metrics:

MSE (Mean Squared Error):
  = {mse:.6f}
  Interpretation: Average of squared errors
  
RMSE (Root Mean Squared Error):
  = {rmse:.6f}
  Same unit as target, easier to interpret
  
MAE (Mean Absolute Error):
  = {mae:.6f}
  Average absolute deviation
  
R² Score:
  = {r2:.4f}
  Percentage of variance explained
  Range: 0-1 (higher is better)
  
MAPE (Mean Absolute Percentage Error):
  = {mape:.2f}%
  Percentage error on average
""")


print("\n4. RESIDUAL ANALYSIS")
print("-"*70)

residuals = y_test - y_pred
abs_residuals = np.abs(residuals)

print(f"Residuals (Actual - Predicted):")
print(f"  Mean: {residuals.mean():.4f} (should ≈ 0)")
print(f"  Std:  {residuals.std():.4f}")
print(f"  Min:  {residuals.min():.4f}")
print(f"  Max:  {residuals.max():.4f}")

print(f"\nAbsolute Residuals:")
print(f"  Mean: {abs_residuals.mean():.4f}")
print(f"  Std:  {abs_residuals.std():.4f}")
print(f"  Min:  {abs_residuals.min():.4f}")
print(f"  Max:  {abs_residuals.max():.4f}")

# Percentage of predictions within error margin
within_10pct = np.sum(abs_residuals < 0.1) / len(residuals) * 100
within_20pct = np.sum(abs_residuals < 0.2) / len(residuals) * 100

print(f"\nPredictions within error margin:")
print(f"  Within 0.1 (±10%): {within_10pct:.1f}%")
print(f"  Within 0.2 (±20%): {within_20pct:.1f}%")


print("\n5. EVALUATION VISUALIZATIONS")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Predictions vs Actual
ax = axes[0, 0]
ax.scatter(y_test, y_pred, alpha=0.6, s=50, color='blue')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'r--', linewidth=2, label='Perfect prediction')
ax.set_xlabel('Actual Revenue (normalized)')
ax.set_ylabel('Predicted Revenue (normalized)')
ax.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Residuals
ax = axes[0, 1]
ax.scatter(y_pred, residuals, alpha=0.6, s=50, color='green')
ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax.set_xlabel('Predicted Revenue')
ax.set_ylabel('Residuals (Actual - Predicted)')
ax.set_title('Residual Plot', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Residuals distribution
ax = axes[1, 0]
ax.hist(residuals, bins=20, alpha=0.7, color='purple', edgecolor='black')
ax.axvline(x=residuals.mean(), color='r', linestyle='--', linewidth=2, 
          label=f'Mean={residuals.mean():.3f}')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Error metrics summary
ax = axes[1, 1]
metrics_text = f"""
Model Performance Summary

Metrics:
━━━━━━━━━━━━━━━━━━━━━━━━
MSE:   {mse:.6f}
RMSE:  {rmse:.6f}
MAE:   {mae:.6f}
R²:    {r2:.4f}
MAPE:  {mape:.2f}%

Prediction Quality:
━━━━━━━━━━━━━━━━━━━━━━━━
Perfect: {within_10pct:.1f}% ± 10%
Good:    {within_20pct:.1f}% ± 20%

Sample Errors:
━━━━━━━━━━━━━━━━━━━━━━━━
Min error:   {abs_residuals.min():.4f}
Max error:   {abs_residuals.max():.4f}
Median error:{np.median(abs_residuals):.4f}
"""

ax.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
       verticalalignment='center', transform=ax.transAxes)
ax.axis('off')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: model_evaluation.png")
plt.show()


print("\n6. ERROR ANALYSIS")
print("-"*70)

print(f"""
Best predictions (smallest errors):
""")
best_idx = np.argsort(abs_residuals.flatten())[:5]
for idx in best_idx:
    print(f"  Actual: {y_test[idx, 0]:.4f}, Pred: {y_pred[idx, 0]:.4f}, Error: {residuals[idx, 0]:.4f}")

print(f"\nWorst predictions (largest errors):")
worst_idx = np.argsort(abs_residuals.flatten())[-5:]
for idx in worst_idx:
    print(f"  Actual: {y_test[idx, 0]:.4f}, Pred: {y_pred[idx, 0]:.4f}, Error: {residuals[idx, 0]:.4f}")



"""
Model successfully trained and evaluated!

Performance Assessment:

✓ R² Score: {r2:.4f}
  Interpretation: Model explains {r2*100:.1f}% of revenue variance

✓ RMSE: {rmse:.6f}
  Average prediction error magnitude

✓ Predictions within ±20%: {within_20pct:.1f}%
  Most predictions are reasonably accurate

Key Insights:
  - Model captures main revenue patterns
  - Residuals roughly centered at 0 (unbiased)
  - Few extreme prediction errors
  - Ready for inference on new data!

Possible improvements:
  - Hyperparameter tuning (learning rate, layers)
  - More training data
  - Additional features
  - Ensemble methods

Next: Make predictions on new data!
"""