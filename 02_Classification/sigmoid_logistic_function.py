"""
SIGMOID, LOGISTIC LOSS & LOGISTIC REGRESSION
=============================================

Learn the sigmoid function and logistic loss for classification.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 80)
print("PART 1: THE SIGMOID FUNCTION")
print("=" * 80)

# ============ WHAT IS SIGMOID? ============
print("\n📍 SIGMOID FUNCTION:")
print("-" * 80)

def sigmoid(z):
    """Convert any score to probability (0 to 1)"""
    return 1 / (1 + np.exp(-z))

print("""
SIGMOID FUNCTION: σ(z) = 1 / (1 + e^(-z))

Purpose:
  - Takes any score (−∞ to +∞)
  - Outputs a probability (0 to 1)
  - Used in logistic regression for classification

Properties:
  - σ(0) = 0.5
  - σ(+∞) = 1.0
  - σ(-∞) = 0.0
  - Always between 0 and 1 ✓
""")

# ============ VISUALIZE SIGMOID ============
print("\n📊 Visualizing sigmoid...\n")

z = np.linspace(-10, 10, 1000)
sigma_z = sigmoid(z)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Sigmoid function
ax1 = axes[0]
ax1.plot(z, sigma_z, 'b-', linewidth=3, label='σ(z)')
ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision threshold = 0.5')
ax1.axvline(x=0, color='green', linestyle='--', linewidth=1, alpha=0.5)
ax1.scatter([0], [0.5], color='red', s=200, marker='*', zorder=3)

# Mark important points
important_points = [(-5, sigmoid(-5)), (-1, sigmoid(-1)), (0, 0.5), 
                    (1, sigmoid(1)), (5, sigmoid(5))]
for z_val, sig_val in important_points:
    ax1.scatter([z_val], [sig_val], color='purple', s=100, zorder=3)
    ax1.annotate(f'({z_val}, {sig_val:.2f})', xy=(z_val, sig_val), 
                xytext=(z_val+0.5, sig_val-0.1), fontsize=9)

ax1.set_xlabel('z (Score)', fontsize=12)
ax1.set_ylabel('σ(z) (Probability)', fontsize=12)
ax1.set_title('Sigmoid Function: Score → Probability', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_ylim([-0.1, 1.1])

# Plot 2: Classification interpretation
ax2 = axes[1]
z_range = np.linspace(-10, 10, 100)
prob = sigmoid(z_range)

ax2.fill_between(z_range[prob < 0.5], 0, 1, alpha=0.3, color='red', label='Class 0')
ax2.fill_between(z_range[prob >= 0.5], 0, 1, alpha=0.3, color='blue', label='Class 1')
ax2.plot(z_range, prob, 'g-', linewidth=2.5, label='P(Class 1)')
ax2.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision point')
ax2.set_xlabel('z = w·x + b', fontsize=12)
ax2.set_ylabel('Probability', fontsize=12)
ax2.set_title('How Sigmoid Makes Predictions', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============ SIGMOID NUMERICAL EXAMPLES ============
print("📊 SIGMOID NUMERICAL EXAMPLES:")
print("-" * 80)
print(f"{'Score (z)':<15} {'σ(z)':<15} {'Interpretation':<40}")
print("-" * 80)

test_scores = [-5, -2, -1, 0, 1, 2, 5]
for score in test_scores:
    prob = sigmoid(score)
    if prob < 0.5:
        interp = f"→ Predict Class 0 ({prob*100:.1f}% sure)"
    elif prob > 0.5:
        interp = f"→ Predict Class 1 ({prob*100:.1f}% sure)"
    else:
        interp = "→ Can't decide"
    
    print(f"{score:<15.0f} {prob:<15.4f} {interp:<40}")


# ============ PART 2: LOGISTIC LOSS ============
print("\n" + "=" * 80)
print("PART 2: LOGISTIC LOSS (Binary Cross-Entropy)")
print("=" * 80)

print("""
LOGISTIC LOSS = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

When y=1 (true label is 1):
  Loss = -log(ŷ)
  - Want ŷ close to 1 → Loss close to 0 ✓
  - If ŷ = 0.99 → Loss ≈ 0.01 (good!)
  - If ŷ = 0.01 → Loss ≈ 4.60 (very bad!)

When y=0 (true label is 0):
  Loss = -log(1-ŷ)
  - Want ŷ close to 0 → Loss close to 0 ✓
  - If ŷ = 0.01 → Loss ≈ 0.01 (good!)
  - If ŷ = 0.99 → Loss ≈ 4.60 (very bad!)
""")

# ============ COMPARE LOSS FUNCTIONS ============
print("\n📍 COMPARING LOSS FUNCTIONS:")
print("-" * 80)

predictions = np.linspace(0.01, 0.99, 100)

# Squared error
y_true = 1
squared_loss = (predictions - y_true) ** 2

# Logistic loss
logistic_loss_y1 = -np.log(predictions)
logistic_loss_y0 = -np.log(1 - predictions)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: For y=1
ax1 = axes[0]
ax1.plot(predictions, squared_loss, 'orange', linewidth=2.5, label='Squared Error')
ax1.plot(predictions, logistic_loss_y1, 'b-', linewidth=2.5, label='Logistic Loss')
ax1.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
ax1.set_xlabel('Predicted Probability (ŷ)', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Loss Functions When True Label y=1', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: For y=0
ax2 = axes[1]
ax2.plot(predictions, 1 - squared_loss, 'orange', linewidth=2.5, label='Squared Error')
ax2.plot(predictions, logistic_loss_y0, 'b-', linewidth=2.5, label='Logistic Loss')
ax2.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Threshold (0.5)')
ax2.set_xlabel('Predicted Probability (ŷ)', fontsize=11)
ax2.set_ylabel('Loss', fontsize=11)
ax2.set_title('Loss Functions When True Label y=0', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("""
KEY INSIGHT:
  ✓ Logistic loss PUNISHES wrong predictions EXPONENTIALLY!
  ✓ Squared error is linear (not as harsh)
  ✓ For classification, logistic loss is BETTER ✓
""")


# ============ PART 3: LOGISTIC REGRESSION FROM SCRATCH ============
print("\n" + "=" * 80)
print("PART 3: LOGISTIC REGRESSION IMPLEMENTATION")
print("=" * 80)

print("\nBuilding logistic regression step-by-step...\n")

# Training data
x_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_train = np.array([0, 0, 0, 1, 1, 1])

m = len(x_train)

print(f"Training data:")
print(f"  x: {x_train.flatten()}")
print(f"  y: {y_train}")
print(f"  Samples: {m}\n")

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.1
iterations = 100

print(f"Hyperparameters:")
print(f"  Initial w: {w}")
print(f"  Initial b: {b}")
print(f"  Learning rate: {learning_rate}")
print(f"  Iterations: {iterations}\n")

# Storage for visualization
w_history = [w]
b_history = [b]
loss_history = []
predictions_history = []

print(f"{'Iter':<6} {'w':<10} {'b':<10} {'Loss':<12} {'Accuracy':<10}")
print("-" * 50)

# Training loop
for i in range(iterations):
    # 1. Make predictions
    z = w * x_train + b
    y_pred = sigmoid(z)
    predictions_history.append(y_pred.copy())
    
    # 2. Calculate loss
    # Add small epsilon to avoid log(0)
    loss = -np.mean(y_train * np.log(y_pred + 1e-15) + 
                    (1 - y_train) * np.log(1 - y_pred + 1e-15))
    loss_history.append(loss)
    
    # 3. Calculate gradients
    dw = (1/m) * np.sum((y_pred - y_train) * x_train)
    db = (1/m) * np.sum(y_pred - y_train)
    
    # 4. Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    w_history.append(w)
    b_history.append(b)
    
    # Calculate accuracy
    y_pred_class = (y_pred > 0.5).astype(int).flatten()
    accuracy = np.mean(y_pred_class == y_train)
    
    if i % 20 == 0:
        print(f"{i:<6} {w:<10.4f} {b:<10.4f} {loss:<12.4f} {accuracy:<10.1%}")

print(f"{iterations:<6} {w:<10.4f} {b:<10.4f} {loss:<12.4f} {accuracy:<10.1%}")

print(f"\n✓ CONVERGED!")
print(f"Final parameters: w = {w:.4f}, b = {b:.4f}")

# ============ VISUALIZATIONS ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Loss over iterations
ax1 = axes[0, 0]
ax1.plot(loss_history, 'b-', linewidth=2)
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Loss', fontsize=11)
ax1.set_title('Loss Decreasing (Gradient Descent Working)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Parameters converging
ax2 = axes[0, 1]
ax2_twin = ax2.twinx()
ax2.plot(w_history, 'g-', linewidth=2, label='Weight (w)')
ax2_twin.plot(b_history, 'orange', linewidth=2, label='Bias (b)')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Weight (w)', fontsize=11, color='g')
ax2_twin.set_ylabel('Bias (b)', fontsize=11, color='orange')
ax2.set_title('Parameters Converging', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelcolor='g')
ax2_twin.tick_params(axis='y', labelcolor='orange')
ax2.grid(True, alpha=0.3)
ax2.legend(loc='upper left')
ax2_twin.legend(loc='upper right')

# Plot 3: Predictions over iterations
ax3 = axes[1, 0]
for idx in [0, 25, 50, 99]:
    ax3.plot(x_train, predictions_history[idx], 'o-', 
            label=f'Iteration {idx}', alpha=0.5)
ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax3.scatter(x_train, y_train, color='black', s=100, marker='X', zorder=5, label='True labels')
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('Predicted Probability', fontsize=11)
ax3.set_title('Model Predictions Evolving', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)
ax3.set_ylim([-0.1, 1.1])

# Plot 4: Final decision boundary
ax4 = axes[1, 1]
x_plot = np.linspace(0, 7, 200).reshape(-1, 1)
z_plot = w * x_plot + b
y_plot = sigmoid(z_plot)

ax4.plot(x_plot, y_plot, 'g-', linewidth=3, label='Logistic Regression')
ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision boundary')
ax4.scatter(x_train[y_train==0], y_train[y_train==0], color='red', s=200, 
           marker='o', label='Class 0', edgecolors='black', linewidth=2, zorder=3)
ax4.scatter(x_train[y_train==1], y_train[y_train==1], color='blue', s=200, 
           marker='s', label='Class 1', edgecolors='black', linewidth=2, zorder=3)
ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('Probability of Class 1', fontsize=11)
ax4.set_title('Final Logistic Regression Model', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_ylim([-0.1, 1.1])

plt.tight_layout()
plt.show()

# ============ MAKE PREDICTIONS ON NEW DATA ============
print("\n" + "=" * 80)
print("MAKING PREDICTIONS ON NEW DATA")
print("=" * 80)

new_x = np.array([1.5, 3.5, 5.5])
print(f"\nNew data points: {new_x}")

print(f"\n{'x':<8} {'Score (z)':<15} {'Probability':<15} {'Prediction':<12}")
print("-" * 50)

for x_val in new_x:
    z_val = w * x_val + b
    prob = sigmoid(z_val)
    pred = "Class 1" if prob > 0.5 else "Class 0"
    print(f"{x_val:<8.1f} {z_val:<15.4f} {prob:<15.4f} {pred:<12}")

print("\n✓ Model successfully trained and making predictions!")