"""
OVERFITTING & REGULARIZATION:  
Learn how to detect and prevent overfitting using regularization.
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

# ✅ Add this at the very beginning!
os.makedirs('overfitting_graphs', exist_ok=True)
print("Created folder: overfitting_graphs\n")

print("=" * 50)
print("Part 1 : understadding OVERFITTING ")
print("=" * 50)

print("Example 1: OVERFITTING WITH POLYNOMIAL DEGREES\n")


# Generate data with noise 
np.random.seed(42)
X_train_raw = np.linspace(0, 10, 20)
Y_train_raw = 2 + 0.5*X_train_raw

# Generate test data (cleaner, represents true pattern)
X_test_raw = np.linspace(0, 10, 100)
Y_test_raw = 2 + 0.5*X_test_raw

#Reshape for sklearn

X_train = X_train_raw.reshape(-1,1)
Y_train = Y_train_raw
X_test = X_test_raw.reshape(-1, 1)
Y_test = Y_test_raw

print(f"Training data: {len(X_train)} samples (with noise)")
print(f"Test data : {len(X_test)} samples (clean, represent true pattern)")
print(f"True relationship : y = 2 + 0.5*x \n")

#  Train model with different polynomial  degrees 

degrees = [1, 3, 10, 15]
models = {}
train_errors = {}
test_errors = {}

print(f"{'Degree':<10} {'Train Error':<20} {'Test Error':<20} {'Overfitting?':<15}")
print("-"*65)

for degree in degrees:
    # creste polynomial feature 
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    #Train model 
    model = Ridge(alpha=0.0)
    model.fit(X_train_poly, Y_train)

    # Calculate error 
    Y_train_pred = model.predict(X_train_poly)
    Y_test_pred = model.predict(X_test_poly)

    train_error = np.mean((Y_train_pred - Y_train) ** 2)
    test_error = np.mean((Y_test_pred - Y_test) ** 2)

    models[degree] = (model, poly)
    train_errors[degree] = train_error
    test_errors[degree] = test_error
    
    # Detect overfitting 
    overfitting_ratio = test_error / train_error
    is_overfitting = "yes " if overfitting_ratio > 2 else " NO"

    print(f"{degree:<10} {train_error:<20.4f} {test_error:<20.4f} {is_overfitting:<15}")
 

# ============ VISUALIZE OVERFITTING ============
print("\nGenerating visualizations...\n")

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, degree in enumerate(degrees):
    ax = axes[idx]

    model, poly = models[degree]
    #create smooth curve 

    X_smooth = np.linspace(-1, 11, 200).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    Y_smooth = model.predict(X_smooth_poly)

    # plot training data 

    ax.scatter(X_train, Y_train, color='red', s=100, label='Training Data', zorder=3)

     # Plot test data
    ax.plot(X_test, Y_test, 'g--', linewidth=2, label='True Pattern (Test Data)', alpha=0.7)
    
    # Plot fitted curve
    ax.plot(X_smooth, Y_smooth, 'b-', linewidth=2.5, label='Fitted Model')

    # Styling 
    overfitting = test_errors[degree] / train_errors[degree] > 2
    title_color = 'red' if overfitting else 'green'
    title_suffix ="Overfitting" if overfitting else "Good fit"

    ax.set_xlim([-1, 11])
    ax.set_ylim([1, 15])
    ax.set_xlabel('x', fontsize= 11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f"Degree {degree} - {title_suffix}", fontsize= 11, fontweight= 'bold', color= title_color)
    ax.legend(fontsize= 9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('overfitting_graphs/01_polynomial_degrees.png', dpi=300, bbox_inches='tight')
plt.show()
plt.show()

 
# ============ TRAIN vs TEST ERROR PLOT ============
fig, ax = plt.subplots(figsize=(10, 6))
 
degrees_list = sorted(train_errors.keys())
train_vals = [train_errors[d] for d in degrees_list]
test_vals = [test_errors[d] for d in degrees_list]
 
ax.plot(degrees_list, train_vals, 'bo-', linewidth=2.5, markersize=10, label='Training Error')
ax.plot(degrees_list, test_vals, 'rs-', linewidth=2.5, markersize=10, label='Test Error')
 
# Mark the overfitting point
ax.axvline(x=3, color='green', linestyle='--', alpha=0.5, linewidth=2)
ax.text(3.2, max(test_vals)*0.9, 'Start of\nOverfitting', fontsize=10, color='green', fontweight='bold')
 
ax.set_xlabel('Polynomial Degree', fontsize=12)
ax.set_ylabel('Mean Squared Error', fontsize=12)
ax.set_title('Training Error vs Test Error\n(Signs of Overfitting)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(degrees_list)
 
plt.tight_layout()
plt.savefig('overfitting_graphs/02_train_vs_test_error.png', dpi=300, bbox_inches='tight')

plt.show()



"""REGULARIZATION adds a PENALTY to the cost function
 
New Cost = Original Cost + λ * Penalty
 
Common Regularization:
1. L2 (Ridge):  Penalty = λ *  Σ(w^2)
2. L1 (Lasso):  Penalty = λ * Σ|w|
"""
 
print("\n" + "=" * 80)
print("PART 2: REGULARIZATION - THE SOLUTION")
print("=" * 80)

print("📍 EXAMPLE 2: REGULARIZATION IN ACTION")

# Use high degree polynomial (prone to overfitting)
degree = 10
 
# Create features
poly = PolynomialFeatures(degree=degree)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)
 
# Try different lambda values
lambdas = [0, 0.01, 0.1, 1.0, 10.0]
results = {}
 
print(f"Testing different λ values:\n")
print(f"{'Lambda':<12} {'Train Error':<20} {'Test Error':<20} {'Improvement':<15}")
print("-" * 67)


for lam in lambdas:
    # Ridge regression (L2 regularization)
    # Note: sklearn's Ridge uses alpha=λ
    model = Ridge(alpha=lam)
    model.fit(X_train_poly, Y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    train_error = np.mean((y_train_pred - Y_train) ** 2)
    test_error = np.mean((y_test_pred - Y_test) ** 2)
    
    results[lam] = (train_error, test_error)

    if lam == 0:
        base_test_error = test_error
    
    improvement = ((base_test_error - test_error) / base_test_error * 100)
    
    print(f"{lam:<12.2f} {train_error:<20.4f} {test_error:<20.4f} {improvement:>6.1f}% {'↓' if improvement > 0 else '↑':<8}")
 
print("\n✓ Regularization reduces test error (better generalization)!")


# ============ VISUALIZE REGULARIZATION EFFECT ============
print("\nGenerating visualizations...\n")
 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes = axes.flatten()
 
for idx, lam in enumerate(lambdas):
    ax = axes[idx]
    
    # Train model
    model = Ridge(alpha=lam)
    model.fit(X_train_poly, Y_train)
    
    # Predictions
    X_smooth = np.linspace(-1, 11, 200).reshape(-1, 1)
    X_smooth_poly = poly.transform(X_smooth)
    y_smooth = model.predict(X_smooth_poly)
    
    # Plot
    ax.scatter(X_train, Y_train, color='red', s=100, label='Training Data', zorder=3)
    ax.plot(X_test, Y_test, 'g--', linewidth=2, label='True Pattern', alpha=0.7)
    ax.plot(X_smooth, y_smooth, 'b-', linewidth=2.5, label='Model Fit')
    
    train_err, test_err = results[lam]
    
    ax.set_xlim([-1, 11])
    ax.set_ylim([0, 15])
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(f'λ = {lam} (Train: {train_err:.2f}, Test: {test_err:.2f})', 
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
 
# Remove extra subplot
fig.delaxes(axes[-1])
 
plt.tight_layout()
plt.savefig('overfitting_graphs/03_regularization_effect.png', dpi=300, bbox_inches='tight')

plt.show()
 
# ============ LAMBDA EFFECT PLOT ============
fig, ax = plt.subplots(figsize=(10, 6))
 
lambdas_list = sorted(results.keys())
train_vals = [results[l][0] for l in lambdas_list]
test_vals = [results[l][1] for l in lambdas_list]
 
ax.plot(lambdas_list, train_vals, 'bo-', linewidth=2.5, markersize=10, label='Training Error')
ax.plot(lambdas_list, test_vals, 'rs-', linewidth=2.5, markersize=10, label='Test Error')
 
# Mark optimal lambda
optimal_idx = np.argmin(test_vals)
optimal_lambda = lambdas_list[optimal_idx]
ax.scatter([optimal_lambda], [test_vals[optimal_idx]], s=300, color='green', 
          marker='*', zorder=5, label='Optimal λ')
 
ax.set_xlabel('Lambda (λ) - Regularization Strength', fontsize=12)
ax.set_ylabel('Error', fontsize=12)
ax.set_title('Effect of Regularization Strength\n(Finding Optimal λ)', fontsize=13, fontweight='bold')
ax.set_xscale('log')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('overfitting_graphs/04_l1_vs_l2.png', dpi=300, bbox_inches='tight')
plt.show()
 
print(f"✓ Optimal λ = {optimal_lambda}")
print(f"  - Too small λ: Overfitting")
print(f"  - Too large λ: Underfitting")
 
 
# ============ PART 3: L1 vs L2 REGULARIZATION ============
print("\n" + "=" * 80)
print("PART 3: L1 vs L2 REGULARIZATION")
print("=" * 80)
 
# Train both models
lambda_val = 0.1
 
# L2 (Ridge)
ridge_model = Ridge(alpha=lambda_val)
ridge_model.fit(X_train_poly, Y_train)
 
# L1 (Lasso)
lasso_model = Lasso(alpha=lambda_val, max_iter=5000)
lasso_model.fit(X_train_poly, Y_train)
 
# Compare coefficients
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
# Plot 1: Coefficients comparison
ax1 = axes[0]
coef_indices = np.arange(len(ridge_model.coef_))
width = 0.35
 
ax1.bar(coef_indices - width/2, ridge_model.coef_, width, label='L2 (Ridge)', alpha=0.7)
ax1.bar(coef_indices + width/2, lasso_model.coef_, width, label='L1 (Lasso)', alpha=0.7)
ax1.set_xlabel('Coefficient Index', fontsize=11)
ax1.set_ylabel('Coefficient Value', fontsize=11)
ax1.set_title('Coefficients: L1 vs L2 Regularization', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3, axis='y')
 
# Plot 2: Model predictions
ax2 = axes[1]
 
X_smooth = np.linspace(-1, 11, 200).reshape(-1, 1)
X_smooth_poly = poly.transform(X_smooth)
 
y_ridge = ridge_model.predict(X_smooth_poly)
y_lasso = lasso_model.predict(X_smooth_poly)
 
ax2.scatter(X_train, Y_train, color='red', s=100, label='Training Data', zorder=3)
ax2.plot(X_test, Y_test, 'gray', linestyle='--', linewidth=2, label='True Pattern', alpha=0.7)
ax2.plot(X_smooth, y_ridge, 'b-', linewidth=2.5, label='L2 (Ridge)')
ax2.plot(X_smooth, y_lasso, 'g-', linewidth=2.5, label='L1 (Lasso)')
 
ax2.set_xlim([-1, 11])
ax2.set_ylim([0, 15])
ax2.set_xlabel('x', fontsize=11)
ax2.set_ylabel('y', fontsize=11)
ax2.set_title('Predictions: L1 vs L2', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('overfitting_graphs/lambda.png', dpi=300, bbox_inches='tight')
plt.show()
 



print("""
1. DETECT OVERFITTING:
   - Monitor training vs validation error
   - If validation error increases while training decreases → Overfitting!
 
2. SOLUTIONS:
   a) Use regularization (λ penalty)
   b) Get more training data
   c) Use simpler model (lower degree)
   d) Feature selection
   e) Early stopping
 
3. REGULARIZATION TYPES:
   - L2 (Ridge): General purpose ✓
   - L1 (Lasso): Feature selection
   
4. CHOOSING λ:
   - Use cross-validation
   - Find λ that minimizes validation error
   - Not too small (overfitting), not too large (underfitting)
""")





