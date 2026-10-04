"""
COURSE 2 TOPIC 5: BIAS-VARIANCE TRADEOFF
========================================
 
The MOST IMPORTANT concept in machine learning!
Understand overfitting/underfitting
"""

"""
Bias: Error from wrong assumptions
  - Oversimplified model
  - Can't fit training data
  - Underfitting
  - High bias → Model too simple
 
Variance: Sensitivity to training data fluctuations
  - Model too complex
  - Memorizes training data
  - Overfitting
  - High variance → Model too complex
 
Total Error = Bias² + Variance + Irreducible Error
"""

import numpy as np
import matplotlib.pyplot as plt
 


def true_func(x):
    return x**2 - 2*x + 3
 
rng = np.random.default_rng(42)
sigma = 2.0
n_train, n_sets = 20, 200
x_train_shared = np.linspace(-2, 4, n_train)
x_test = np.linspace(-2, 4, 100)
degrees = list(range(1, 13))
 
def fit_predict(x_tr, y_tr, x_te, deg):
    c = np.polynomial.polynomial.polyfit((x_tr - 1) / 3, y_tr, deg)      # x scaled to about [-1, 1]: numerically stable
    return np.polynomial.polynomial.polyval((x_te - 1) / 3, c)
 
preds = {d: np.zeros((n_sets, len(x_test))) for d in degrees}
train_mse = {d: [] for d in degrees}
first_y = None
for s_ in range(n_sets):
    y_tr = true_func(x_train_shared) + rng.normal(0, sigma, n_train)
    if first_y is None:
        first_y = y_tr
    for d in degrees:
        preds[d][s_] = fit_predict(x_train_shared, y_tr, x_test, d)
        train_mse[d].append(np.mean((y_tr - fit_predict(x_train_shared, y_tr, x_train_shared, d)) ** 2))
 
truth = true_func(x_test)
bias2 = np.array([np.mean((preds[d].mean(axis=0) - truth) ** 2) for d in degrees])
variance = np.array([np.mean(preds[d].var(axis=0)) for d in degrees])
noise_var = sigma ** 2
predicted_test = bias2 + variance + noise_var
train_err = np.array([np.mean(train_mse[d]) for d in degrees])
test_err = np.array([np.mean((preds[d] - (truth + rng.normal(0, sigma, (n_sets, len(x_test))))) ** 2) for d in degrees])
best_d = degrees[int(np.argmin(test_err))]
 
print(f"{'degree':>6} | {'bias^2':>8} | {'variance':>9} | {'bias^2+var+noise':>16} | {'measured test':>13} | {'train':>7}")
for i, d in enumerate(degrees):
    print(f"{d:>6} | {bias2[i]:8.3f} | {variance[i]:9.3f} | {predicted_test[i]:16.3f} | {test_err[i]:13.3f} | {train_err[i]:7.3f}")
print(f"\nNoise floor (irreducible) = sigma^2 = {noise_var:.1f}. Lowest test error at degree {best_d} (the true function is a degree-2 polynomial).")
print("Low degree: bias^2 dominates (underfit). High degree: variance dominates (overfit). Train error keeps falling.")




fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
ax = axes[0, 0]
ax.plot(degrees, train_err, 'o-', label='Train error', linewidth=2)
ax.plot(degrees, test_err, 's--', label='Test error', linewidth=2)
ax.axhline(noise_var, color='gray', linestyle=':', label='Noise floor (irreducible)')
ax.axvline(best_d, color='green', alpha=0.5, label=f'Lowest test error: degree {best_d}')
ax.set_yscale('log'); ax.set_xlabel('Polynomial degree (model complexity)'); ax.set_ylabel('Mean squared error (log)')
ax.set_title('Train error keeps falling, test error turns up', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(degrees, bias2, 'o-', label='Bias^2', linewidth=2)
ax.plot(degrees, variance, 's-', label='Variance', linewidth=2)
ax.plot(degrees, predicted_test, 'k--', label='Bias^2 + Variance + Noise', linewidth=2)
ax.set_yscale('log'); ax.set_xlabel('Polynomial degree'); ax.set_ylabel('Error (log)')
ax.set_title('Measured bias and variance', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
 
ax = axes[1, 0]
ax.scatter(x_train_shared, first_y, c='k', s=25, label='One training set')
ax.plot(x_test, truth, 'g-', linewidth=3, label='True function')
for d, col in [(1, 'tab:red'), (2, 'tab:blue'), (10, 'tab:orange')]:
    ax.plot(x_test, preds[d][0], color=col, linewidth=2, label=f'Degree {d}')
ax.set_ylim(-5, 25); ax.set_title('Same data, three models', fontweight='bold'); ax.legend(); ax.grid(True, alpha=0.3)
 
ax = axes[1, 1]
for s_ in range(30):
    ax.plot(x_test, preds[10][s_], color='tab:orange', alpha=0.25)
ax.plot(x_test, preds[10].mean(axis=0), 'b-', linewidth=2, label='Average of all fits')
ax.plot(x_test, truth, 'g-', linewidth=3, label='True function')
ax.set_ylim(-5, 25); ax.set_title('Variance: 30 degree-10 fits on different training sets', fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('bias_variance_tradeoff.png', dpi=300, bbox_inches='tight')
print("✓ Saved: bias_variance_tradeoff.png")
plt.show()