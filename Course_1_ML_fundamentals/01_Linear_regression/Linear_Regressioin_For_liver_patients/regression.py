"""
LINEAR REGRESSION & COST FUNCTION
Concept + Visualization + Coding Tricks & Tips
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# PART 1: CONCEPT EXPLANATION (with code)
# ============================================================================

"""
LINEAR REGRESSION in 3 sentences:
1. We have data (x, y) - house size and price
2. We want to find a line y = wx + b that fits the data
3. We measure "badness of fit" using Cost Function, then minimize it

COST FUNCTION in 3 sentences:
1. It measures how far our predictions are from actual values
2. Formula: J(w,b) = (1/2m) * Σ(ŷ - y)² where ŷ = wx + b
3. Lower cost = better line fit. Our job: find w,b that minimize cost.
"""

# ============================================================================
# PART 2: SIMPLE EXAMPLE WITH VISUALIZATION
# ============================================================================

print("="*70)
print("PART 1: Simple Linear Regression Example")
print("="*70)

# Simple dataset: house size vs price
x_train = np.array([1, 2, 3, 4, 5])        # House size (thousands sq ft)
y_train = np.array([1, 2, 2.5, 4, 5])      # Price (hundreds of thousands)
m = len(x_train)  # number of examples

print(f"\nDataset: {m} training examples")
print(f"x_train (sizes): {x_train}")
print(f"y_train (prices): {y_train}")

# Let's try 3 different lines and see their costs
def compute_cost(x, y, w, b):
    """
    CODING TIP #1: Vectorized computation (FAST)
    Don't use loops! Use NumPy arrays.
    
    Formula: J(w,b) = (1/2m) * Σ(ŷ - y)²
    where ŷ = w*x + b
    """
    m = len(x)
    y_pred = w * x + b                    # All predictions at once
    errors = y_pred - y                   # All errors at once
    cost = np.sum(errors**2) / (2 * m)    # Sum and average
    return cost

# CODING TIP #2: Test with 3 different lines
lines = [
    {"w": 1, "b": 0, "name": "y = x"},
    {"w": 0.5, "b": 0.5, "name": "y = 0.5x + 0.5"},
    {"w": 1, "b": 1, "name": "y = x + 1"}
]

print("\n" + "-"*70)
print("Testing 3 different lines:")
print("-"*70)

for line in lines:
    w, b = line["w"], line["b"]
    cost = compute_cost(x_train, y_train, w, b)
    line["cost"] = cost
    print(f"{line['name']:20} → Cost = {cost:.4f}")

best_line = min(lines, key=lambda x: x["cost"])
print(f"\n✓ Best line so far: {best_line['name']} with cost {best_line['cost']:.4f}")

# ============================================================================
# PART 3: VISUALIZATION - See the Cost Function
# ============================================================================

print("\n" + "="*70)
print("PART 2: Visualizing Cost Function")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# VISUALIZATION 1: The data and the 3 lines
ax = axes[0, 0]
ax.scatter(x_train, y_train, s=100, c='red', label='Actual data', zorder=5)

x_range = np.linspace(0, 6, 100)
for line in lines:
    y_range = line["w"] * x_range + line["b"]
    ax.plot(x_range, y_range, label=f"{line['name']} (cost={line['cost']:.3f})", linewidth=2)

ax.set_xlabel('House Size (1000 sq ft)', fontsize=11)
ax.set_ylabel('Price (100k)', fontsize=11)
ax.set_title('3 Different Lines Fit to Data', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6)
ax.set_ylim(0, 6)

# VISUALIZATION 2: Cost function varies with w (b fixed at 0)
ax = axes[0, 1]
w_values = np.linspace(-0.5, 2, 100)
costs_for_w = [compute_cost(x_train, y_train, w, 0) for w in w_values]

ax.plot(w_values, costs_for_w, linewidth=2, color='blue')
for line in lines:
    ax.scatter([line["w"]], [line["cost"]], s=150, zorder=5, label=f"w={line['w']}")

ax.set_xlabel('w (slope)', fontsize=11)
ax.set_ylabel('Cost J(w, b=0)', fontsize=11)
ax.set_title('Cost Function vs w (when b=0)', fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# VISUALIZATION 3: 3D Surface of Cost Function
ax = fig.add_subplot(2, 2, 3, projection='3d')
w_range = np.linspace(-0.5, 2, 30)
b_range = np.linspace(-1, 2, 30)
W, B = np.meshgrid(w_range, b_range)
Z = np.zeros_like(W)

for i in range(len(w_range)):
    for j in range(len(b_range)):
        Z[j, i] = compute_cost(x_train, y_train, W[j, i], B[j, i])

surf = ax.plot_surface(W, B, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('w (slope)', fontsize=10)
ax.set_ylabel('b (intercept)', fontsize=10)
ax.set_zlabel('Cost J(w,b)', fontsize=10)
ax.set_title('3D Cost Function Surface', fontsize=12, fontweight='bold')
fig.colorbar(surf, ax=ax, shrink=0.5)

# VISUALIZATION 4: Contour plot (top-down view of 3D)
ax = axes[1, 1]
contour = ax.contour(W, B, Z, levels=20, cmap='viridis')
ax.clabel(contour, inline=True, fontsize=8)

for line in lines:
    ax.scatter([line["w"]], [line["b"]], s=150, c='red', zorder=5)
    ax.annotate(f"Cost={line['cost']:.3f}", (line["w"], line["b"]), 
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax.set_xlabel('w (slope)', fontsize=11)
ax.set_ylabel('b (intercept)', fontsize=11)
ax.set_title('Contour Plot: Cost Function (Top View)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/claude/linear_regression_visualization.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'linear_regression_visualization.png'")
plt.show()

# ============================================================================
# PART 4: CODING TRICKS & TIPS
# ============================================================================

print("\n" + "="*70)
print("PART 3: CODING TRICKS & TIPS")
print("="*70)

print("""
TRICK #1: Vectorization (Speed)
─────────────────────────────────
❌ SLOW (using loops):
    cost = 0
    for i in range(m):
        prediction = w * x[i] + b
        error = prediction - y[i]
        cost += error**2
    cost = cost / (2*m)

✓ FAST (using NumPy):
    y_pred = w * x + b
    cost = np.sum((y_pred - y)**2) / (2*m)

Why? NumPy is implemented in C, 100x faster!


TRICK #2: Check Dimensions (Debugging)
──────────────────────────────────────
Before computing cost, always check shapes:
    print(f"x shape: {x.shape}")      # Should be (m,)
    print(f"y shape: {y.shape}")      # Should be (m,)
    print(f"w: {w}, b: {b}")          # Scalars

This catches 90% of bugs immediately!


TRICK #3: Cost Should Decrease Over Time
─────────────────────────────────────────
When training (updating w,b), costs should form a decreasing curve:
    costs = [...]
    if costs[-1] > costs[-2]:
        print("⚠️ Cost increased! Learning rate too high? w,b updates wrong?")


TRICK #4: Normalize Features for Faster Learning
──────────────────────────────────────────────────
If features have very different scales (e.g., x ∈ [1000, 5000]):
    x_normalized = (x - np.mean(x)) / np.std(x)
    
This makes gradient descent converge faster!
Cost function updates are more stable.


TRICK #5: Sanity Check on Simple Data
──────────────────────────────────────
Test your compute_cost() function with data you can verify manually:
    x = np.array([1, 2])
    y = np.array([2, 3])
    w, b = 1, 0  # y = x (perfect fit!)
    cost = compute_cost(x, y, w, b)  # Should be ≈ 0
    
If it's not 0, you have a bug!


TRICK #6: Handle Edge Cases
────────────────────────────
    # What if m=0? You'll get division by zero!
    if m == 0:
        return 0  # or raise an error
    
    # What if all y values are the same?
    # Cost will be high (can't fit a line well)


TRICK #7: Use Float64 for Precision
────────────────────────────────────
    x = np.array([1, 2, 3], dtype=np.float64)  # Not int!
    
Integer arithmetic can cause rounding errors.
""")

# ============================================================================
# PART 5: HANDS-ON PRACTICE PROBLEMS
# ============================================================================

print("\n" + "="*70)
print("PART 4: HANDS-ON PRACTICE")
print("="*70)

print("""
EXERCISE 1 (BASIC): Calculate Cost by Hand
─────────────────────────────────────────────
Data: x = [1, 2], y = [2, 3], w = 0.5, b = 0.5

Step 1: Make predictions
    ŷ₁ = 0.5*1 + 0.5 = 1
    ŷ₂ = 0.5*2 + 0.5 = 1.5

Step 2: Calculate errors
    error₁ = 1 - 2 = -1
    error₂ = 1.5 - 3 = -1.5

Step 3: Calculate cost
    J = (1/2*2) * ((-1)² + (-1.5)²)
    J = (1/4) * (1 + 2.25) = 0.8125

YOUR TURN: Try w=1, b=0 with the same data.
What is the cost?
""")

# Let's verify Exercise 1
print("\n" + "-"*70)
print("EXERCISE 1 VERIFICATION:")
print("-"*70)
x_ex1 = np.array([1, 2])
y_ex1 = np.array([2, 3])

w, b = 0.5, 0.5
cost1 = compute_cost(x_ex1, y_ex1, w, b)
print(f"Cost for w={w}, b={b}: {cost1:.4f} ✓")

print("\nNow try w=1, b=0:")
w, b = 1, 0
cost2 = compute_cost(x_ex1, y_ex1, w, b)
print(f"Cost for w={w}, b={b}: {cost2:.4f}")

print("""
EXERCISE 2 (MEDIUM): Find Best w (brute force)
────────────────────────────────────────────────
Given: x = [1, 2, 3], y = [1, 2, 3], b = 0
Find the w that minimizes cost.

Hint: Try w from 0.5 to 1.5 in steps of 0.1
Which w gives lowest cost?
""")

print("\n" + "-"*70)
print("EXERCISE 2 SOLUTION:")
print("-"*70)
x_ex2 = np.array([1, 2, 3])
y_ex2 = np.array([1, 2, 3])
b = 0

w_values_ex2 = np.arange(0.5, 1.6, 0.1)
costs_ex2 = []

for w in w_values_ex2:
    cost = compute_cost(x_ex2, y_ex2, w, b)
    costs_ex2.append(cost)
    print(f"w = {w:.1f} → Cost = {cost:.6f}")

best_w_idx = np.argmin(costs_ex2)
print(f"\n✓ Best w = {w_values_ex2[best_w_idx]:.1f} with cost {costs_ex2[best_w_idx]:.6f}")

print("""
EXERCISE 3 (HARD): Implement Cost Function from Scratch
──────────────────────────────────────────────────────────
Write a function `my_cost_function(x, y, w, b)` without using NumPy operations.
Use loops instead. Then compare speed with vectorized version.

Hint:
def my_cost_function(x, y, w, b):
    cost = 0
    for i in range(len(x)):
        # Your code here
    return cost / (2 * len(x))
""")

print("\n" + "-"*70)
print("EXERCISE 3 SOLUTION (Loop Version):")
print("-"*70)

def my_cost_function_loop(x, y, w, b):
    """Non-vectorized version (for comparison)"""
    m = len(x)
    cost = 0
    for i in range(m):
        y_pred = w * x[i] + b
        error = y_pred - y[i]
        cost += error**2
    return cost / (2 * m)

# Compare speeds
import time

x_large = np.random.randn(1000)
y_large = 2 * x_large + 1 + np.random.randn(1000) * 0.1

# Vectorized version
start = time.time()
for _ in range(1000):
    cost_vec = compute_cost(x_large, y_large, 2, 1)
time_vec = time.time() - start

# Loop version
start = time.time()
for _ in range(1000):
    cost_loop = my_cost_function_loop(x_large, y_large, 2, 1)
time_loop = time.time() - start

print(f"Vectorized (1000 iterations): {time_vec:.4f}s")
print(f"Loop version (1000 iterations): {time_loop:.4f}s")
print(f"Speedup: {time_loop/time_vec:.1f}x faster with vectorization!")

# ============================================================================
# PART 6: COMMON MISTAKES
# ============================================================================

print("\n" + "="*70)
print("PART 5: COMMON MISTAKES & HOW TO AVOID THEM")
print("="*70)

print("""
MISTAKE #1: Forgetting to divide by 2m
────────────────────────────────────────
❌ Wrong: J = np.sum((y_pred - y)**2)
✓ Right: J = np.sum((y_pred - y)**2) / (2*m)

Why? The /2 makes math simpler in calculus. The /m makes cost scale-independent.


MISTAKE #2: Using integers instead of floats
──────────────────────────────────────────────
❌ Wrong: x = np.array([1, 2, 3])  # int64
         cost = np.sum(...) / 2 * m  # Integer division!

✓ Right: x = np.array([1, 2, 3], dtype=np.float64)


MISTAKE #3: Not checking if cost is NaN or Inf
───────────────────────────────────────────────
❌ If cost becomes NaN, your algorithm is broken:
    if np.isnan(cost):
        print("❌ Cost is NaN! Check your data and learning rate")

✓ Add this check in your training loop.


MISTAKE #4: Confusing prediction vs cost
──────────────────────────────────────────
PREDICTION (ŷ):
    For a SINGLE example: ŷ = w*x + b (gives one value)
    
COST (J):
    Averaged over ALL examples (measures total error)
    
They're different things!


MISTAKE #5: Not visualizing your data
──────────────────────────────────────
Always plot your data + predictions!
    plt.scatter(x, y, label='Actual')
    plt.plot(x, w*x + b, label='Prediction')
    plt.legend()
    plt.show()
    
If the line doesn't look right, there's a bug!
""")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("""
✓ Cost Function measures how bad our line fit is
✓ Lower cost = better fit
✓ Formula: J(w,b) = (1/2m) * Σ(ŷ - y)²
✓ Use vectorization (NumPy) for speed
✓ Always visualize data + cost function
✓ Test on simple data first to verify your code
✓ Check dimensions and data types
✓ Add debugging checks (NaN, Inf, cost decreasing)
""")