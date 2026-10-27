"""
COMPLETE ML CONCEPTS - COST FUNCTION + GRADIENT DESCENT + POLYNOMIAL REGRESSION
================================================================================

This file shows how all three concepts work together in a complete
machine learning workflow.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

print("=" * 80)
print("COMPLETE WORKFLOW: COST FUNCTION → GRADIENT DESCENT → POLYNOMIAL REGRESSION")
print("=" * 80)

# ============ PART 1: GENERATE REALISTIC DATA ============
print("\n📍 PART 1: GENERATING TRAINING DATA")
print("-" * 80)

# Generate data: House prices vs square footage
# Real relationship: price = 150*sqft + noise (with some curvature)
np.random.seed(42)
sqft = np.array([800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600])
noise = np.random.normal(0, 20000, len(sqft))
price = 150 * sqft + 50000 + noise

print(f"Generated {len(sqft)} house examples:")
print(f"\nSquare Footage: {sqft}")
print(f"Prices ($):     {price.astype(int)}\n")

# Reshape for sklearn
x_train = sqft.reshape(-1, 1)
y_train = price

# ============ PART 2: LINEAR REGRESSION WITH GRADIENT DESCENT ============
print("\n" + "=" * 80)
print("📍 PART 2: IMPLEMENT GRADIENT DESCENT FROM SCRATCH")
print("-" * 80)

print("\nCustom Gradient Descent Implementation:")
print("-" * 80)

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.00001
iterations = 5000
m = len(x_train)

# Tracking
w_history = [w]
b_history = [b]
cost_history = []

print(f"Initial: w={w}, b={b}")
print(f"Learning rate: {learning_rate}")
print(f"Iterations: {iterations}\n")

# -------- GRADIENT DESCENT LOOP --------
for i in range(iterations):
    # ✅ STEP 1: MAKE PREDICTIONS
    y_pred = w * x_train + b
    
    # ✅ STEP 2: CALCULATE COST FUNCTION
    cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
    cost_history.append(cost)
    
    # ✅ STEP 3: CALCULATE GRADIENTS (derivatives)
    dw = (1 / m) * np.sum((y_pred - y_train) * x_train)
    db = (1 / m) * np.sum(y_pred - y_train)
    
    # ✅ STEP 4: UPDATE PARAMETERS
    w_new = w - learning_rate * dw
    b_new = b - learning_rate * db
    
    w_history.append(w_new)
    b_history.append(b_new)
    
    w = w_new
    b = b_new
    
    # Print progress
    if i % 1000 == 0:
        print(f"Iteration {i:>4d}: w={w:>8.2f}, b={b:>10.0f}, cost={cost:>15.0f}")

print(f"Iteration {iterations}: w={w:.2f}, b={b:.0f}, cost={cost:.0f}")

print(f"\n✅ CONVERGED!")
print(f"Final model: Price = {w:.2f} × SquareFeet + {b:.0f}")
print(f"Final cost: ${cost:,.0f}\n")

# Predictions
y_pred_gd = w * x_train + b
mae_gd = np.mean(np.abs(y_pred_gd - y_train))
print(f"Mean Absolute Error: ${mae_gd:,.0f}")


# ============ PART 3: TRY POLYNOMIAL REGRESSION ============
print("\n" + "=" * 80)
print("📍 PART 3: POLYNOMIAL REGRESSION (Degree 2)")
print("-" * 80)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)

print(f"Original features: [sqft]")
print(f"After poly transform: [1, sqft, sqft²]\n")

# Train polynomial model
poly_model = LinearRegression()
poly_model.fit(x_train_poly, y_train)
y_pred_poly = poly_model.predict(x_train_poly)

# Calculate cost
cost_poly = np.mean((y_pred_poly - y_train) ** 2)
mae_poly = np.mean(np.abs(y_pred_poly - y_train))

print(f"Polynomial coefficients:")
print(f"  w₀ (constant): {poly_model.intercept_:,.0f}")
print(f"  w₁ (linear):   {poly_model.coef_[1]:,.2f}")
print(f"  w₂ (quadratic): {poly_model.coef_[2]:,.6f}\n")

print(f"Polynomial cost: ${cost_poly:,.0f}")
print(f"Polynomial MAE: ${mae_poly:,.0f}\n")


# ============ PART 4: COMPARISON ============
print("=" * 80)
print("📊 COMPARISON: LINEAR vs POLYNOMIAL")
print("-" * 80)

print(f"{'Model':<20} {'Cost':<20} {'MAE':<20}")
print("-" * 60)
print(f"{'Linear (GD)':<20} ${cost:>15,.0f}   ${mae_gd:>15,.0f}")
print(f"{'Polynomial':<20} ${cost_poly:>15,.0f}   ${mae_poly:>15,.0f}")

improvement = ((cost - cost_poly) / cost * 100)
print(f"\nPolynomial is {improvement:.1f}% better!")


# ============ PART 5: VISUALIZATIONS ============
print("\n" + "=" * 80)
print("📊 GENERATING VISUALIZATIONS...")
print("-" * 80)

fig = plt.figure(figsize=(16, 10))

# -------- Plot 1: Cost Function Over Iterations --------
ax1 = plt.subplot(2, 3, 1)
ax1.plot(cost_history, 'b-', linewidth=2)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Cost', fontsize=11)
ax1.set_title('Cost Function Decreasing\n(Gradient Descent Working)', 
              fontsize=11, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')

# -------- Plot 2: Parameters Converging --------
ax2 = plt.subplot(2, 3, 2)
ax2_right = ax2.twinx()
ax2.plot(w_history, 'g-', linewidth=2, label='Weight (w)')
ax2_right.plot(b_history, 'orange', linewidth=2, label='Bias (b)')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Weight (w)', fontsize=11, color='g')
ax2_right.set_ylabel('Bias (b)', fontsize=11, color='orange')
ax2.set_title('Parameters Converging\n(Finding Best w and b)', 
              fontsize=11, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='g')
ax2_right.tick_params(axis='y', labelcolor='orange')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_right.legend(loc='upper right')

# -------- Plot 3: Linear Regression Results --------
ax3 = plt.subplot(2, 3, 3)
ax3.scatter(x_train, y_train, color='red', s=100, label='Actual Data', zorder=3)
ax3.plot(x_train, y_pred_gd, 'g-', linewidth=2.5, label='Linear (GD)')
ax3.set_xlabel('Square Feet', fontsize=11)
ax3.set_ylabel('Price ($)', fontsize=11)
ax3.set_title(f'Linear Regression\nCost: ${cost:,.0f}', fontsize=11, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# -------- Plot 4: Linear vs Polynomial --------
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(x_train, y_train, color='red', s=100, label='Data', zorder=3)
ax4.plot(x_train, y_pred_gd, 'g--', linewidth=2, label='Linear (GD)', alpha=0.7)
ax4.plot(x_train, y_pred_poly, 'b-', linewidth=2.5, label='Polynomial')
ax4.set_xlabel('Square Feet', fontsize=11)
ax4.set_ylabel('Price ($)', fontsize=11)
ax4.set_title('Linear vs Polynomial Comparison', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# -------- Plot 5: Residuals (Errors) --------
ax5 = plt.subplot(2, 3, 5)
residuals_linear = y_train - y_pred_gd
residuals_poly = y_train - y_pred_poly
ax5.scatter(x_train, residuals_linear, color='g', s=100, label='Linear', alpha=0.7)
ax5.scatter(x_train, residuals_poly, color='b', s=100, label='Polynomial', alpha=0.7)
ax5.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax5.set_xlabel('Square Feet', fontsize=11)
ax5.set_ylabel('Residuals ($)', fontsize=11)
ax5.set_title('Prediction Errors\n(Lower is Better)', fontsize=11, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)

# -------- Plot 6: 3D Cost Function Visualization --------
from mpl_toolkits.mplot3d import Axes3D
ax6 = plt.subplot(2, 3, 6, projection='3d')

# Create a grid of w and b values
w_range = np.linspace(0, 200, 30)
b_range = np.linspace(-100000, 200000, 30)
W, B = np.meshgrid(w_range, b_range)

# Calculate cost for each combination
Cost = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        y_test = W[j, i] * x_train + B[j, i]
        Cost[j, i] = (1 / (2 * m)) * np.sum((y_test - y_train) ** 2)

surf = ax6.plot_surface(W, B, Cost, cmap='viridis', alpha=0.7)
ax6.scatter([w], [b], [cost], color='red', s=200, marker='*', 
            label='Final Position', zorder=5)
ax6.set_xlabel('w', fontsize=10)
ax6.set_ylabel('b', fontsize=10)
ax6.set_zlabel('Cost', fontsize=10)
ax6.set_title('3D Cost Surface\n(Where Gradient Descent Rolled)', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

print("✅ Visualizations displayed!")


# ============ PART 6: MAKE PREDICTIONS ============
print("\n" + "=" * 80)
print("📍 PART 6: MAKING NEW PREDICTIONS")
print("-" * 80)

test_sqft = np.array([1500, 2500, 3500]).reshape(-1, 1)

# Linear predictions
test_pred_linear = w * test_sqft + b

# Polynomial predictions
test_poly = poly.transform(test_sqft)
test_pred_poly = poly_model.predict(test_poly)

print(f"\nNew houses to predict:\n")
print(f"{'Square Feet':<15} {'Linear Pred':<20} {'Polynomial Pred':<20}")
print("-" * 55)

for i, sqft_val in enumerate(test_sqft.flatten()):
    print(f"{sqft_val:<15.0f} ${test_pred_linear[i, 0]:>15,.0f}   ${test_pred_poly[i]:>15,.0f}")


# ============ PART 7: KEY LEARNINGS ============
print("\n" + "=" * 80)
print("🎯 KEY LEARNINGS")
print("=" * 80)

print("""
1️⃣ COST FUNCTION:
   - Measures how bad predictions are
   - Lower cost = Better model
   - Formula: J(w,b) = (1/2m) × Σ(ŷ - y)²

2️⃣ GRADIENT DESCENT:
   - Automatically finds best w and b
   - Takes steps opposite to gradient (downhill)
   - Update rule: w = w - α × (∂J/∂w)
   - Learning rate (α) controls step size

3️⃣ POLYNOMIAL REGRESSION:
   - Use when data is curved
   - Create higher powers of x as features
   - Then use linear regression on new features
   - Degree 2 or 3 usually works best

4️⃣ WORKFLOW:
   Data → Cost Function → Gradient Descent → Learn w,b
   → Make Predictions

5️⃣ WHEN TO USE POLYNOMIAL:
   - Data shows curved pattern
   - Linear regression gives high cost
   - Degree 2-3 for most cases
   - Watch out for overfitting with high degrees!
""")

print("=" * 80)
print("✅ COMPLETE WORKFLOW FINISHED!")
print("=" * 80)