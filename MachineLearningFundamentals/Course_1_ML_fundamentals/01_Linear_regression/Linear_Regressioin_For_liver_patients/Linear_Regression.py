import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

"""
LINEAR REGRESSION :
1. We have data (x, y) - house size and price
2. We want to find a line y = wx + b that fits the data
3. We measure "badness of fit" using Cost Function, then minimize it
 
COST FUNCTION :
1. It measures how far our predictions are from actual values
2. Formula: J(w,b) = (1/2m) * Σ(ŷ - y)² where ŷ = wx + b
3. Lower cost = better line fit. Our job: find w,b that minimize cost.
"""

print("="*70)
print("PART 1: Simple Linear Regression Example")
print("="*70)


# Load CSV
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'indian_liver_patient.csv')
 
print(f"Looking for CSV at: {csv_path}")
print(f"File exists: {os.path.exists(csv_path)}")
 
df = pd.read_csv(csv_path)
print(df.head())
print(df.info())
print(df.describe())
 
# Predict Albumin from Age 
x_train = df['Age'].values.astype(np.float64)
y_train = df['Albumin'].values.astype(np.float64)
 
print(f"\nMissing values:\n{df.isnull().sum()}")
 
# Remove rows with NaN
mask = ~(np.isnan(x_train) | np.isnan(y_train))
x_train = x_train[mask]
y_train = y_train[mask]
 
m = len(x_train)
print(f"\nDataset: {m} training examples")
print(f"x_train (Age) range: {x_train.min():.1f} to {x_train.max():.1f}")
print(f"y_train (Albumin) range: {y_train.min():.2f} to {y_train.max():.2f}")
 
# Normalize data
x_mean, x_std = np.mean(x_train), np.std(x_train)
y_mean, y_std = np.mean(y_train), np.std(y_train)
 
x_norm = (x_train - x_mean) / x_std
y_norm = (y_train - y_mean) / y_std
 
print(f"\nNormalized x range: {x_norm.min():.2f} to {x_norm.max():.2f}")
print(f"Normalized y range: {y_norm.min():.2f} to {y_norm.max():.2f}")



def compute_cost(x, y, w, b):
    m = len(x)
    """
    Formula: J(w,b) = (1/2m) * Σ(ŷ - y)²
    where ŷ = w*x + b

    """
    y_pred = w * x + b
    errors = y_pred - y
    cost = np.sum(errors **2 ) / (2 * m) # vectorized computations 
    return cost

 
# Test 3 different lines
lines = [
    {"w": 0.3, "b": 0, "name": "y = 0.3x"},
    {"w": 0.5, "b": 0.1, "name": "y = 0.5x + 0.1"},
    {"w": 0.7, "b": 0, "name": "y = 0.7x"}
]
 
print("\n" + "-"*70)
print("Testing 3 different lines:")
print("-"*70)
 
for line in lines:
    w, b = line["w"], line["b"]
    cost = compute_cost(x_norm, y_norm, w, b)
    line["cost"] = cost
    print(f"{line['name']:25} -> cost = {cost:.6f}")
 
best_line = min(lines, key=lambda x: x["cost"])
print(f"\n✓ Best line so far: {best_line['name']} with cost {best_line['cost']:.6f}")

# Visualization of cost function

print("\n" + "="*70)
print("PART 2: Visualizing Cost Function")
print("="*70)


fig, axes = plt.subplots(1, 2, figsize = (14,5))


# Visualixation 2

print("\n" + "="*70)
print("PART 2: Creating 4 Visualizations (2x2 Grid)")
print("="*70)
 
# CREATE 2x2 GRID FIRST (IMPORTANT!)
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
 
# ===== VISUALIZATION 1: Data + 3 Lines (Top Left) =====
ax = axes[0, 0]
ax.scatter(x_norm, y_norm, alpha=0.5, s=50, label='Actual data', color='blue')
 
x_range = np.linspace(x_norm.min()-0.5, x_norm.max()+0.5, 100)
for line in lines:
    y_range = line["w"] * x_range + line["b"]
    ax.plot(x_range, y_range, label=f"{line['name']} (cost={line['cost']:.4f})", linewidth=2)
 
ax.set_xlabel('Age (normalized)', fontsize=11)
ax.set_ylabel('Albumin (normalized)', fontsize=11)
ax.set_title('Linear Regression: Age vs Albumin', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

## ===== VISUALIZATION 2: Cost Function vs w (Top Right) =====
ax = axes[0, 1]
w_values = np.linspace(-1, 2, 100)
costs_for_w = [compute_cost(x_norm, y_norm, w, 0) for w in w_values]
 
ax.plot(w_values, costs_for_w, linewidth=2, color='blue')
for line in lines:
    ax.scatter([line["w"]], [line["cost"]], s=150, zorder=5)
 
ax.set_xlabel('w (slope)', fontsize=11)
ax.set_ylabel('Cost J(w, b=0)', fontsize=11)
ax.set_title('Cost Function vs w (when b=0)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)


# ===== VISUALIZATION 3: 3D Surface (Bottom Left) =====
ax = fig.add_subplot(2, 2, 3, projection='3d')
w_range = np.linspace(-0.5, 2, 30)
b_range = np.linspace(-1, 2, 30)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W)
 
# Fill Z array correctly
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Z[i, j] = compute_cost(x_norm, y_norm, W[i, j], B[i, j])
 
surf = ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('w (slope)', fontsize=10)
ax.set_ylabel('b (intercept)', fontsize=10)
ax.set_zlabel('Cost J(w,b)', fontsize=10)
ax.set_title('3D Cost Function Surface', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax, shrink=0.5)

 
# ===== VISUALIZATION 4: Contour Plot (Bottom Right) =====
ax = axes[1, 1]
contour = ax.contour(W, B, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)
 
for idx, line in enumerate(lines):
    ax.scatter([line["w"]], [line["b"]], s=150, c='red', zorder=5)
    # Offset each annotation differently so they don't overlap
    offsets = [(10, 10), (10, -20), (-40, 10)]
    ax.annotate(f"cost={line['cost']:.3f}", (line["w"], line["b"]), 
                xytext=offsets[idx], textcoords='offset points', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
 
ax.set_xlabel('w (slope)', fontsize=11)
ax.set_ylabel('b (intercept)', fontsize=11)
ax.set_title('Contour Plot: Cost Function (Top View)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:\ML-Learning-Journey\LINEAR REGRESSION\Visualizationslinear_regression_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'linear_regression_visualization.png'")
plt.show()