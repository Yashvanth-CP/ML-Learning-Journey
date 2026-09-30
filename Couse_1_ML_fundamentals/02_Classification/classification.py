"""
CLASSIFICATION & DECISION BOUNDARY
===================================

Learn how classification works and what decision boundaries are.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.linear_model import LogisticRegression
import os

# ✅ Add this at the very beginning!
os.makedirs('classification_graphs', exist_ok=True)
print("Created folder: classification_graphs\n")

print("=" * 80)
print("PART 1: CLASSIFICATION BASICS")
print("=" * 80)

# ============ REGRESSION vs CLASSIFICATION ============
print("\n📍 REGRESSION vs CLASSIFICATION:")
print("-" * 80)

print("""
REGRESSION:
  Task: Predict a NUMBER
  Example: House price, temperature, stock price
  Output: Any continuous value
  
CLASSIFICATION:
  Task: Predict a CATEGORY/CLASS
  Example: Spam/Not Spam, Disease/Healthy, Cat/Dog
  Output: A class label (0 or 1, or multiple classes)
""")

# ============ EXAMPLE 1: BINARY CLASSIFICATION ============
print("\n" + "=" * 80)
print("📍 EXAMPLE 1: BINARY CLASSIFICATION (2 Classes)")
print("-" * 80)

# Generate simple binary classification data
np.random.seed(42)
X_binary = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                n_informative=2, random_state=42)
X = X_binary[0]
y = X_binary[1]

print(f"\nGenerated {len(X)} data points with 2 features each")
print(f"Class 0 (Not Spam): {np.sum(y == 0)} samples")
print(f"Class 1 (Spam): {np.sum(y == 1)} samples")

# Train logistic regression
model = LogisticRegression()
model.fit(X, y)

print(f"\nModel trained!")
print(f"Coefficients (weights): {model.coef_[0]}")
print(f"Intercept (bias): {model.intercept_[0]}")

# Make predictions on new data
test_point = np.array([[0, 0]])
pred_class = model.predict(test_point)[0]
pred_prob = model.predict_proba(test_point)[0]

print(f"\nTest point: {test_point[0]}")
print(f"Predicted class: {pred_class}")
print(f"Predicted probability: Class 0={pred_prob[0]:.2%}, Class 1={pred_prob[1]:.2%}")

# ========== VISUALIZE DECISION BOUNDARY ============
print("\nGenerating visualization...\n")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Data and decision boundary
ax1 = axes[0]

# Create mesh for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Plot decision regions
ax1.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)

# Plot decision boundary
ax1.contour(xx, yy, model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape),
           levels=[0], linewidths=2, colors='black')

# Plot data points
scatter1 = ax1.scatter(X[y == 0, 0], X[y == 0, 1], c='red', s=100, 
                       label='Class 0 (Not Spam)', marker='o', edgecolors='black')
scatter2 = ax1.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', s=100, 
                       label='Class 1 (Spam)', marker='s', edgecolors='black')

ax1.set_xlabel('Feature 1', fontsize=11)
ax1.set_ylabel('Feature 2', fontsize=11)
ax1.set_title('Linear Decision Boundary', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Probability distribution
ax2 = axes[1]

# Create a line and show probabilities
x_line = np.linspace(x_min, x_max, 100)
y_line = np.linspace(y_min, y_max, 100)
X_line = np.c_[x_line, np.zeros_like(x_line)]
probs = model.predict_proba(X_line)[:, 1]

ax2.plot(x_line, probs, 'g-', linewidth=2.5, label='P(Class 1)')
ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax2.fill_between(x_line, 0, probs, alpha=0.3, color='blue')
ax2.set_xlabel('Feature 1 (at Feature 2 = 0)', fontsize=11)
ax2.set_ylabel('Probability of Class 1', fontsize=11)
ax2.set_title('Predicted Probabilities Along a Line', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 1])
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Probability_distribution.png', dpi=300,  bbox_inches='tight' )
plt.show()


# ============ EXAMPLE 2: NON-LINEAR DECISION BOUNDARY ============
print("\n" + "=" * 80)
print("📍 EXAMPLE 2: WHEN LINEAR ISN'T ENOUGH - CIRCLES")
print("-" * 80)

# Generate circle dataset (can't separate with straight line)
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

print(f"\nGenerated circular dataset with {len(X_circles)} samples")
print(f"Class 0: {np.sum(y_circles == 0)}, Class 1: {np.sum(y_circles == 1)}")
print(f"\nThis dataset is NOT linearly separable!")
print(f"We need polynomial features or non-linear models!\n")

# Train linear model (will fail)
model_linear = LogisticRegression()
model_linear.fit(X_circles, y_circles)

# Train non-linear model using polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_circles_poly = poly.fit_transform(X_circles)
model_poly = LogisticRegression()
model_poly.fit(X_circles_poly, y_circles)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for idx, (model, X_data, poly_features, title) in enumerate([
    (model_linear, X_circles, None, 'Linear Model (Fails)'),
    (model_poly, X_circles, poly, 'Polynomial Model (Works)')
]):
    ax = axes[idx]
    
    # Create mesh
    x_min, x_max = X_circles[:, 0].min() - 0.5, X_circles[:, 0].max() + 0.5
    y_min, y_max = X_circles[:, 1].min() - 0.5, X_circles[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z_points = np.c_[xx.ravel(), yy.ravel()]
    
    if poly_features:
        Z_points = poly_features.transform(Z_points)
    
    Z = model.predict(Z_points).reshape(xx.shape)
    
    # Plot
    ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    
    ax.scatter(X_circles[y_circles == 0, 0], X_circles[y_circles == 0, 1], 
              c='red', s=100, label='Class 0', marker='o', edgecolors='black')
    ax.scatter(X_circles[y_circles == 1, 0], X_circles[y_circles == 1, 1], 
              c='blue', s=100, label='Class 1', marker='s', edgecolors='black')
    
    ax.set_xlabel('Feature 1', fontsize=11)
    ax.set_ylabel('Feature 2', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('NonLinearDicision.png', dpi=300,  bbox_inches='tight' )
plt.show()

# Scores
score_linear = model_linear.score(X_circles, y_circles)
score_poly = model_poly.score(X_circles_poly, y_circles)

print(f"📊 ACCURACY COMPARISON:")
print(f"  Linear Model:     {score_linear:.1%}")
print(f"  Polynomial Model: {score_poly:.1%}")
print(f"\n✓ Polynomial features allow non-linear decision boundaries!")


# ============ EXAMPLE 3: UNDERSTANDING DECISION BOUNDARIES ============
print("\n" + "=" * 80)
print("📍 EXAMPLE 3: HOW DECISION BOUNDARIES WORK")
print("-" * 80)

# Simple 2D classification
X_simple = np.array([[1, 1], [2, 2], [3, 3], [4, 4],
                     [1, 4], [2, 3], [3, 2], [4, 1]])
y_simple = np.array([0, 0, 0, 0, 1, 1, 1, 1])

model_simple = LogisticRegression()
model_simple.fit(X_simple, y_simple)

print(f"\nSimple dataset:")
print(f"{'x1':<5} {'x2':<5} {'Class':<6}")
print("-" * 16)
for i in range(len(X_simple)):
    print(f"{X_simple[i, 0]:<5.0f} {X_simple[i, 1]:<5.0f} {y_simple[i]:<6.0f}")

print(f"\nDecision boundary equation:")
print(f"w₁·x₁ + w₂·x₂ + b = 0")
print(f"{model_simple.coef_[0, 0]:.4f}·x₁ + {model_simple.coef_[0, 1]:.4f}·x₂ + {model_simple.intercept_[0]:.4f} = 0")

# Visualize
fig, ax = plt.subplots(figsize=(8, 8))

x_min, x_max = 0, 5
y_min, y_max = 0, 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = model_simple.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.3)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=3, label='Decision Boundary')

ax.scatter(X_simple[y_simple == 0, 0], X_simple[y_simple == 0, 1], 
          c='red', s=200, label='Class 0', marker='o', edgecolors='black', linewidth=2)
ax.scatter(X_simple[y_simple == 1, 0], X_simple[y_simple == 1, 1], 
          c='blue', s=200, label='Class 1', marker='s', edgecolors='black', linewidth=2)

ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
ax.set_xlabel('x₁', fontsize=12)
ax.set_ylabel('x₂', fontsize=12)
ax.set_title('Decision Boundary Separates Classes', fontsize=13, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

# Add annotations
ax.text(1.5, 1.5, 'Class 0\nRegion', fontsize=11, ha='center', fontweight='bold', color='red')
ax.text(3.5, 3.5, 'Class 1\nRegion', fontsize=11, ha='center', fontweight='bold', color='blue')

plt.tight_layout()
plt.savefig('classification.png', dpi=300,  bbox_inches='tight' )
plt.show()
