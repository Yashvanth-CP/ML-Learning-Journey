"""
COST FUNCTION - How to Measure Prediction Error
================================================

Cost Function tells you how bad your model's predictions are.
Lower cost = Better model
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("PART 1: UNDERSTANDING COST FUNCTION")
print("=" * 60)

# ============ SIMPLE EXAMPLE ============
print("\n📍 SIMPLE EXAMPLE:")
print("-" * 60)

# Training data (house prices)
y_actual = np.array([300000, 400000, 500000])
y_predicted = np.array([280000, 410000, 490000])

print(f"Actual prices:    {y_actual}")
print(f"Predicted prices: {y_predicted}")

# Calculate errors
errors = y_predicted - y_actual
print(f"\nErrors: {errors}")

# Square errors (so negative doesn't cancel positive)
squared_errors = errors ** 2
print(f"Squared Errors: {squared_errors}")

# Calculate cost
m = len(y_actual)
cost = (1 / (2 * m)) * np.sum(squared_errors)

print(f"\nCost Function = (1/2m) × Σ(ŷ - y)²")
print(f"Cost = (1/{2*m}) × {np.sum(squared_errors)}")
print(f"Cost = {cost:.2f}")
print(f"\n✓ Lower cost = Better predictions")


# ============ REALISTIC EXAMPLE WITH HOUSE DATA ============
print("\n" + "=" * 60)
print("📍 REALISTIC EXAMPLE: PREDICTING HOUSE PRICES")
print("=" * 60)

# More house examples
x_train = np.array([1000, 1500, 2000, 2500, 3000])  # Square feet
y_actual = np.array([200000, 300000, 350000, 450000, 500000])

# Let's pretend our model predicts using: y = 150*x + 10000
# (We'll make some wrong predictions)
w = 150
b = 10000

y_predicted = w * x_train + b

print(f"\nOur simple model: y = {w}×x + {b}")
print(f"\nTraining data:")
print(f"{'Size(sqft)':<12} {'Actual($)':<15} {'Predicted($)':<15} {'Error($)':<12}")
print("-" * 60)

errors_list = []
for i in range(len(x_train)):
    error = y_predicted[i] - y_actual[i]
    errors_list.append(error)
    print(f"{x_train[i]:<12} {y_actual[i]:<15} {y_predicted[i]:<15} {error:<12.0f}")

# Calculate cost
m = len(y_actual)
cost = (1 / (2 * m)) * np.sum((y_predicted - y_actual) ** 2)

print("-" * 60)
print(f"Mean Squared Error (MSE) = {cost:,.0f}")


# ============ VISUALIZE COST FUNCTION WITH DIFFERENT w VALUES ============
print("\n" + "=" * 60)
print("📍 HOW COST CHANGES WITH DIFFERENT w VALUES")
print("=" * 60)

# Keep b fixed, change w
w_values = np.linspace(100, 200, 50)
cost_values = []

for w_test in w_values:
    y_pred_test = w_test * x_train + b
    cost_test = (1 / (2 * m)) * np.sum((y_pred_test - y_actual) ** 2)
    cost_values.append(cost_test)

# Find best w (minimum cost)
best_w_idx = np.argmin(cost_values)
best_w = w_values[best_w_idx]
min_cost = cost_values[best_w_idx]

print(f"\nBest w (lowest cost): {best_w:.2f}")
print(f"Minimum cost: {min_cost:,.0f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(w_values, cost_values, 'b-', linewidth=2, label='Cost vs w')
plt.scatter([best_w], [min_cost], color='red', s=200, marker='*', 
            label=f'Minimum (w={best_w:.2f})', zorder=5)
plt.xlabel('Weight (w)', fontsize=12)
plt.ylabel('Cost', fontsize=12)
plt.title('Cost Function: How Cost Changes with w', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()


# ============ 2D COST FUNCTION (w and b together) ============
print("\n" + "=" * 60)
print("📍 2D COST FUNCTION (Changes in both w and b)")
print("=" * 60)

# Create a grid of w and b values
w_range = np.linspace(100, 200, 30)
b_range = np.linspace(-50000, 50000, 30)
W, B = np.meshgrid(w_range, b_range)

# Calculate cost for each combination
Cost = np.zeros_like(W)
for i in range(len(w_range)):
    for j in range(len(b_range)):
        y_pred_test = W[j, i] * x_train + B[j, i]
        Cost[j, i] = (1 / (2 * m)) * np.sum((y_pred_test - y_actual) ** 2)

# Create 3D plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 5))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
surf = ax1.plot_surface(W, B, Cost, cmap='viridis', alpha=0.8)
ax1.set_xlabel('w (weight)', fontsize=10)
ax1.set_ylabel('b (bias)', fontsize=10)
ax1.set_zlabel('Cost', fontsize=10)
ax1.set_title('3D Cost Function Surface', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax1, shrink=0.5)

# Contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(W, B, Cost, levels=20, cmap='viridis')
ax2.clabel(contour, inline=True, fontsize=8)
ax2.set_xlabel('w (weight)', fontsize=10)
ax2.set_ylabel('b (bias)', fontsize=10)
ax2.set_title('Contour Plot of Cost Function', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

print("✓ The valley/crater shape shows where the minimum cost is!")
print("✓ Gradient descent will roll down this surface to find the minimum")
plt.tight_layout()
plt.savefig('C:\ML-Learning-Journey\Cost_Functions.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'Cost_Functions.png', dpi=150, bbox_inches='tight'")
print("\n✓ Visualization saved as 'Cost_Functions.png'")
plt.show()
