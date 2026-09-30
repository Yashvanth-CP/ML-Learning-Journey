"""
GRADIENT DESCENT - How to Find the Best w and b
================================================

Gradient Descent automatically finds the best weights by going downhill
on the cost function surface.
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 70)
print("GRADIENT DESCENT - AUTOMATIC PARAMETER TUNING")
print("=" * 70)

# ============ SIMPLE TRAINING DATA ============
print("\n📍 EXAMPLE 1: SIMPLE LINEAR DATA")
print("-" * 70)

# Simple data: y = 2x (perfect linear relationship)
x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])

print(f"Training data: x = {x_train}")
print(f"               y = {y_train}")
print(f"(Notice: y = 2x perfectly)\n")

# Initialize parameters (start with wrong values)
w = 0.1  # Start with wrong slope
b = 0.0
learning_rate = 0.01
iterations = 1000
m = len(x_train)

print(f"Initial parameters:")
print(f"  w = {w} (should be ≈ 2.0)")
print(f"  b = {b} (should be ≈ 0.0)")
print(f"  Learning rate (α) = {learning_rate}")
print(f"  Iterations = {iterations}\n")

# Store history for visualization
w_history = [w]
b_history = [b]
cost_history = []

# ============ GRADIENT DESCENT LOOP ============
print("Running Gradient Descent...\n")

for i in range(iterations):
    # Step 1: Make predictions
    y_pred = w * x_train + b
    
    # Step 2: Calculate cost
    cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
    cost_history.append(cost)
    
    # Step 3: Calculate gradients (derivatives)
    # ∂J/∂w = (1/m) × Σ(ŷ - y) × x
    dw = (1 / m) * np.sum((y_pred - y_train) * x_train)
    
    # ∂J/∂b = (1/m) × Σ(ŷ - y)
    db = (1 / m) * np.sum(y_pred - y_train)
    
    # Step 4: Update parameters
    # w = w - α × (∂J/∂w)
    # b = b - α × (∂J/∂b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    # Store history
    w_history.append(w)
    b_history.append(b)
    
    # Print progress
    if i % 200 == 0:
        print(f"Iteration {i:>4d}: w={w:.6f}, b={b:.6f}, cost={cost:.6f}")

print(f"Iteration {iterations}: w={w:.6f}, b={b:.6f}, cost={cost:.6f}")

print(f"\n✓ Final parameters:")
print(f"  w = {w:.4f} (Target: 2.0)")
print(f"  b = {b:.4f} (Target: 0.0)")
print(f"  Final Cost = {cost:.6f}")

# ============ VISUALIZATION 1: COST OVER ITERATIONS ============
plt.figure(figsize=(15, 5))

# Cost function over time
plt.subplot(1, 3, 1)
plt.plot(cost_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Cost', fontsize=11)
plt.title('Cost Function Over Iterations', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # Log scale to see changes

# w parameter over time
plt.subplot(1, 3, 2)
plt.plot(w_history, 'g-', linewidth=2)
plt.axhline(y=2.0, color='r', linestyle='--', label='Target w=2.0')
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Weight (w)', fontsize=11)
plt.title('Weight (w) Converging to 2.0', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# Final prediction vs actual
plt.subplot(1, 3, 3)
y_final = w * x_train + b
plt.scatter(x_train, y_train, color='red', s=100, label='Actual Data', zorder=3)
plt.plot(x_train, y_final, 'g-', linewidth=2, label=f'Final Model: y={w:.2f}x+{b:.2f}')
plt.xlabel('x', fontsize=11)
plt.ylabel('y', fontsize=11)
plt.title('Final Learned Model', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('CostOverIteration.png', dpi=300,  bbox_inches='tight' )
plt.show()


# ============ EXAMPLE 2: REAL HOUSE PRICE DATA ============
print("\n" + "=" * 70)
print("📍 EXAMPLE 2: HOUSE PRICE PREDICTION")
print("-" * 70)

# House data
x_train = np.array([1000, 1500, 2000, 2500, 3000])  # Square feet
y_train = np.array([200000, 300000, 350000, 450000, 500000])  # Price

print(f"House data:")
print(f"  Square feet: {x_train}")
print(f"  Prices:      {y_train}\n")

# Initialize parameters
w = 0.0
b = 0.0
learning_rate = 0.00005  # Smaller learning rate for larger numbers
iterations = 1000
m = len(x_train)

print(f"Starting gradient descent...\n")

cost_history = []

for i in range(iterations):
    # Predictions
    y_pred = w * x_train + b
    
    # Cost
    cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
    cost_history.append(cost)
    
    # Gradients
    dw = (1 / m) * np.sum((y_pred - y_train) * x_train)
    db = (1 / m) * np.sum(y_pred - y_train)
    
    # Update
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if i % 200 == 0:
        print(f"Iteration {i:>4d}: w={w:.2f}, b={b:.0f}, cost={cost:,.0f}")

print(f"Iteration {iterations}: w={w:.2f}, b={b:.0f}, cost={cost:,.0f}")

print(f"\n✓ Learned Model: Price = {w:.2f} × Square_Feet + {b:.0f}")

# Visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(cost_history, 'b-', linewidth=2)
plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Cost', fontsize=11)
plt.title('Cost Decreasing Over Time', fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(1, 2, 2)
y_final = w * x_train + b
plt.scatter(x_train, y_train, color='red', s=100, label='Actual Prices', zorder=3)
plt.plot(x_train, y_final, 'g-', linewidth=2, 
         label=f'Learned Model')
plt.xlabel('Square Feet', fontsize=11)
plt.ylabel('Price ($)', fontsize=11)
plt.title('House Price Prediction', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('HousingData.png', dpi=300,  bbox_inches='tight' )
plt.show()

# ============ EXAMPLE 3: EFFECT OF LEARNING RATE ============
print("\n" + "=" * 70)
print("📍 EXAMPLE 3: HOW LEARNING RATE AFFECTS CONVERGENCE")
print("-" * 70)

x_train = np.array([1, 2, 3, 4, 5])
y_train = np.array([2, 4, 6, 8, 10])
m = len(x_train)

learning_rates = [0.001, 0.01, 0.05, 0.1]
colors = ['red', 'green', 'blue', 'purple']

plt.figure(figsize=(12, 5))

for lr, color in zip(learning_rates, colors):
    w = 0.1
    b = 0.0
    cost_history = []
    
    for i in range(1000):
        y_pred = w * x_train + b
        cost = (1 / (2 * m)) * np.sum((y_pred - y_train) ** 2)
        cost_history.append(cost)
        
        dw = (1 / m) * np.sum((y_pred - y_train) * x_train)
        db = (1 / m) * np.sum(y_pred - y_train)
        
        w = w - lr * dw
        b = b - lr * db
    
    plt.plot(cost_history, color=color, linewidth=2, label=f'α = {lr}')
    print(f"Learning rate {lr}: Final cost = {cost:.6f}")

plt.xlabel('Iteration', fontsize=11)
plt.ylabel('Cost', fontsize=11)
plt.title('Effect of Learning Rate on Convergence', fontsize=12, fontweight='bold')
plt.yscale('log')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('effectOfLearningRate.png', dpi=300,  bbox_inches='tight' )
plt.show()

print(f"\n✓ Smaller α = Slower convergence (more iterations)")
print(f"✓ Larger α = Faster convergence (fewer iterations)")
print(f"✓ Too large α = Divergence (doesn't work)")