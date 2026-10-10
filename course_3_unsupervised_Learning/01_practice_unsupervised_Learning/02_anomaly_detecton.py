


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, multivariate_normal
from sklearn.covariance import EllipticEnvelope
 
np.random.seed(42)

mu    = 60.0   # mean server temperature
sigma = 5.0    # standard deviation
 
x_range = np.linspace(30, 100, 500)
p_x     = norm.pdf(x_range, loc=mu, scale=sigma)
 
plt.figure(figsize=(8, 4))
plt.plot(x_range, p_x, 'b-', linewidth=2)
plt.axvline(mu, color='green', linestyle='--', label=f'μ = {mu}')
plt.axvline(mu + 2*sigma, color='orange', linestyle='--', label=f'μ + 2σ')
plt.axvline(95, color='red', linestyle='--', label='New reading = 95°C (anomaly)')
plt.fill_between(x_range, p_x, where=(x_range > 90), alpha=0.3, color='red')
plt.xlabel("Temperature (°C)")
plt.ylabel("p(x)")
plt.title("1D Gaussian — anomaly region shaded red")
plt.legend()
plt.tight_layout()
plt.savefig("learn_c3_02a_gaussian_1d.png")
plt.close()
 
# Manual calculation for x = 95
x_new = 95.0
exponent  = -((x_new - mu)**2) / (2 * sigma**2)
p_manual  = (1 / np.sqrt(2 * np.pi * sigma**2)) * np.exp(exponent)
print(f"x = {x_new}°C,  μ={mu},  σ={sigma}")
print(f"Exponent term  : {exponent:.2f}")
print(f"p(x) manual    : {p_manual:.2e}")
print(f"p(x) scipy     : {norm.pdf(x_new, mu, sigma):.2e}")
print(f"That is {(x_new-mu)/sigma:.1f} standard deviations from the mean")
print("→ ANOMALY (very low probability)\n")
print("Saved: learn_c3_02a_gaussian_1d.png\n")

m_train = 200
cpu_normal  = np.random.normal(loc=60, scale=5,  size=m_train)
mem_normal  = np.random.normal(loc=40, scale=8,  size=m_train)
X_train     = np.column_stack([cpu_normal, mem_normal])  # (200, 2)
 
# Estimate parameters — one μ and σ² per feature
mu_vec    = X_train.mean(axis=0)       # (n,)
sigma2_vec = X_train.var(axis=0)       # (n,)  uses 1/m
 
print(f"Feature 0 (CPU%)  : μ = {mu_vec[0]:.2f},  σ² = {sigma2_vec[0]:.2f}")
print(f"Feature 1 (Mem%)  : μ = {mu_vec[1]:.2f},  σ² = {sigma2_vec[1]:.2f}\n")


def log_prob(x, mu, sigma2):
    """
    Compute log p(x) assuming feature independence.
 
    x      : (n,) one example
    mu     : (n,) mean per feature
    sigma2 : (n,) variance per feature
 
    Returns: scalar log-probability
    """
    n = len(x)
    log_p = 0.0
    for j in range(n):
        log_p += -0.5 * np.log(2 * np.pi * sigma2[j])
        log_p += -((x[j] - mu[j])**2) / (2 * sigma2[j])
    return log_p
 
# Equivalently (vectorised, same result):
def log_prob_vec(x, mu, sigma2):
    return (-0.5 * np.log(2 * np.pi * sigma2)
            - (x - mu)**2 / (2 * sigma2)).sum()
 
# Test on a normal example
x_normal  = np.array([62.0, 38.0])   # close to mean → should be high-ish
x_anomaly = np.array([95.0, 90.0])   # far from mean → should be very low
 
lp_normal  = log_prob(x_normal,  mu_vec, sigma2_vec)
lp_anomaly = log_prob(x_anomaly, mu_vec, sigma2_vec)
 
print(f"Normal  example log p(x) : {lp_normal:.3f}")
print(f"Anomaly example log p(x) : {lp_anomaly:.3f}")
print("(anomaly is MUCH lower — farther from what's normal)\n")
 
# Verify against scipy
lp_scipy = sum(norm.logpdf(x_normal[j], mu_vec[j], np.sqrt(sigma2_vec[j]))
               for j in range(2))
print(f"scipy verification: {lp_scipy:.6f}  vs manual: {log_prob_vec(x_normal, mu_vec, sigma2_vec):.6f}")
print(f"Match: {abs(lp_scipy - log_prob_vec(x_normal, mu_vec, sigma2_vec)) < 1e-9}\n")
 
 

 
# Build a validation set: 100 normal + 20 anomalies
X_val_normal  = np.column_stack([
    np.random.normal(60, 5, 100),
    np.random.normal(40, 8, 100)
])
X_val_anomaly = np.column_stack([
    np.random.uniform(80, 100, 20),
    np.random.uniform(75, 100, 20)
])
X_val   = np.vstack([X_val_normal, X_val_anomaly])
y_val   = np.array([0]*100 + [1]*20)   # 0 = normal, 1 = anomaly
 
# Compute log-probabilities for every validation example
log_probs_val = np.array([log_prob_vec(x, mu_vec, sigma2_vec) for x in X_val])
 
def f1_for_threshold(log_p, y_true, threshold):
    pred = (log_p < threshold).astype(int)   # flag as anomaly if below threshold
    TP = ((pred == 1) & (y_true == 1)).sum()
    FP = ((pred == 1) & (y_true == 0)).sum()
    FN = ((pred == 0) & (y_true == 1)).sum()
    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, precision, recall
 
# Try a range of thresholds
thresholds = np.linspace(log_probs_val.min(), log_probs_val.max(), 200)
f1_scores  = [f1_for_threshold(log_probs_val, y_val, t)[0] for t in thresholds]
 
best_idx = np.argmax(f1_scores)
best_eps = thresholds[best_idx]
best_f1, best_p, best_r = f1_for_threshold(log_probs_val, y_val, best_eps)
 
print(f"Best threshold (log ε) : {best_eps:.3f}")
print(f"Best F1                : {best_f1:.3f}")
print(f"Precision              : {best_p:.3f}")
print(f"Recall                 : {best_r:.3f}\n")
 
plt.figure(figsize=(6, 3))
plt.plot(thresholds, f1_scores, color='darkblue')
plt.axvline(best_eps, color='red', linestyle='--', label=f'Best ε: {best_eps:.1f}')
plt.xlabel("Log-probability threshold")
plt.ylabel("F1 score")
plt.title("Choosing ε — maximize F1 on validation set")
plt.legend()
plt.tight_layout()
plt.savefig("learn_c3_02b_epsilon_choice.png")
plt.close()
print("Saved: learn_c3_02b_epsilon_choice.png\n")
 
cpu_mv  = np.random.normal(60, 5, 300)
mem_mv  = 0.7 * cpu_mv + np.random.normal(0, 4, 300)   # correlated with CPU
X_mv    = np.column_stack([cpu_mv, mem_mv])
 
# Estimate full covariance
mu_mv    = X_mv.mean(axis=0)
Sigma_mv = np.cov(X_mv, rowvar=False)   # (2, 2) covariance matrix
 
print(f"μ  = {mu_mv}")
print(f"Σ  = \n{Sigma_mv}\n")
print("Off-diagonal values = covariance (non-zero → features correlated)")
 
# Score a new point using multivariate Gaussian
x_test = np.array([90.0, 70.0])
lp_mv  = multivariate_normal.logpdf(x_test, mean=mu_mv, cov=Sigma_mv)
print(f"\nAnomaly score for [CPU=90, Mem=70]: log p = {lp_mv:.3f}")
print("(More negative → more anomalous)\n")
 
# Compare independence model vs multivariate
mu_indep     = X_mv.mean(axis=0)
sigma2_indep = X_mv.var(axis=0)
lp_indep     = log_prob_vec(x_test, mu_indep, sigma2_indep)
 
print(f"Independence model : log p = {lp_indep:.3f}")
print(f"Multivariate model : log p = {lp_mv:.3f}")
print("Multivariate is more sensitive to correlation-breaking anomalies.\n")
 
# sklearn sanity check
env = EllipticEnvelope(random_state=0).fit(X_mv)
print(f"sklearn EllipticEnvelope (equivalent to multivariate Gaussian):")
print(f"  Anomaly flag for [90, 70]: {env.predict(x_test.reshape(1,-1))[0] == -1}")