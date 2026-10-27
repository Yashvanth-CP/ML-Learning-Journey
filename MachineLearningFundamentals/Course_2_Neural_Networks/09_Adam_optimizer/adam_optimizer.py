"""
COURSE 2 TOPIC 4: ADAM OPTIMIZER


SGD (Stochastic Gradient Descent):
  w = w - lr * dw
  Problem: Slow, gets stuck in plateaus
  
Momentum:
  v = β*v + dw
  w = w - lr * v
  Better: Accelerates in consistent direction
  
RMSprop:
  s = β*s + dw²
  w = w - lr * dw / √s
  Better: Adapts learning rate per parameter
  
ADAM (Adaptive Moment Estimation):
  m = β1*m + (1-β1)*dw          ← Momentum
  v = β2*v + (1-β2)*dw²         ← RMSprop
  m_hat = m / (1 - β1^t)        ← Bias correction
  v_hat = v / (1 - β2^t)        ← Bias correction
  w = w - lr * m_hat / (√v_hat + ε)
  
  Best of both worlds!
  β1 = 0.9 (momentum coefficient)
  β2 = 0.999 (RMSprop coefficient)
  ε = 1e-8 (numerical stability)
"""


import numpy as np
import matplotlib.pyplot as plt


# Simulate loss landscape
np.random.seed(42)
 
# Function with valley
def loss_landscape(x, y):
    return x**2 + 50*y**2
 
x_vals = np.linspace(-5, 5, 100)
y_vals = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_landscape(X, Y)
 
# Simulate optimization paths (loss = x^2 + 50*y^2: a long, narrow valley)
def run_sgd(lr, steps=50):
    x, y = 4.0, 1.5
    path = [(x, y)]
    for _ in range(steps):
        x -= lr * 2 * x
        y -= lr * 100 * y
        path.append((x, y))
    return np.array(path)
 
def run_adam(lr, steps=50, b1=0.9, b2=0.999, eps=1e-8):
    x, y = 4.0, 1.5
    m, v = np.zeros(2), np.zeros(2)
    path = [(x, y)]
    for t in range(1, steps + 1):
        g = np.array([2 * x, 100 * y])
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        step = lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        x -= step[0]; y -= step[1]
        path.append((x, y))
    return np.array(path)
 
def end_loss(path):
    return path[-1, 0] ** 2 + 50 * path[-1, 1] ** 2
 
print(f"\nFINAL LOSS after 50 steps (start = {16 + 50 * 2.25:.1f}). Every optimizer needs its OWN learning rate:")
sgd_runs = {lr: run_sgd(lr) for lr in [0.005, 0.015, 0.05]}
adam_runs = {lr: run_adam(lr) for lr in [0.05, 0.1, 0.3, 1.0]}
for lr, pth in sgd_runs.items():
    L_ = end_loss(pth)
    print(f"  SGD  lr={lr:<6} -> final loss {L_:.3g}{'   (DIVERGED!)' if L_ > 1e3 else ''}")
for lr, pth in adam_runs.items():
    print(f"  ADAM lr={lr:<6} -> final loss {end_loss(pth):.3g}")
best_sgd = min(sgd_runs, key=lambda k: end_loss(sgd_runs[k]))
best_adam = min(adam_runs, key=lambda k: end_loss(adam_runs[k]))
print(f"Plotted below: the best of each grid -> SGD lr={best_sgd}, ADAM lr={best_adam}")
 
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, pth, name in [(axes[0], sgd_runs[best_sgd], f"SGD (lr={best_sgd})"),
                      (axes[1], adam_runs[best_adam], f"ADAM (lr={best_adam})")]:
    ax.contour(X, Y, Z, levels=20, alpha=0.6)
    ax.plot(pth[:, 0], pth[:, 1], 'o-', markersize=3, linewidth=1.5)
    ax.plot(pth[0, 0], pth[0, 1], 'go', markersize=10, label='Start')
    ax.plot(0, 0, 'r*', markersize=15, label='Minimum')
    ax.set_title(f"{name}: final loss {end_loss(pth):.3g}", fontweight='bold')
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(); ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('adam_vs_sgd.png', dpi=300, bbox_inches='tight')
print("✓ Saved: adam_vs_sgd.png")
plt.show()

"""
learning_rate (lr):
  Typical: 0.001, 0.0001
  Higher → Faster but may diverge
  Lower → Slower but more stable
 
beta1 (momentum coefficient):
  Default: 0.9
  How much to remember past gradients
  0.9 = Remember 90%, current = 10%
 
beta2 (RMSprop coefficient):
  Default: 0.999
  How much to smooth variance
  0.999 = Remember 99.9%, current = 0.1%
 
epsilon:
  Default: 1e-8
  Prevents division by zero
 
Typical defaults work well!
lr = 0.001 is usually safe to start
"""