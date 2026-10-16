"""
ANOMALY DETECTION
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 74)
print("ANOMALY DETECTION")
print("=" * 74)

"""
THE GAUSSIAN (bell curve) for ONE feature
  p(x; mu, var) = 1 / sqrt(2*pi*var) * exp( -(x - mu)^2 / (2*var) )
  mu  = average of the feature in normal data          var = its variance (spread)
  Close to mu -> high p.  Far from mu (many standard deviations) -> tiny p.
 
MANY FEATURES (independent model): multiply the one-feature probabilities
  p(x) = p(x1; mu1, var1) * p(x2; mu2, var2) * ... * p(xn; mun, varn)
 
"""

# Numeric examples 

def gauss_pdf(x, mu, var):
    return np.exp(-(x - mu)** 2/ (2 * var)) / np.sqrt(2*np.pi * var)

print("Numeric Example : one feature, mu = 10, std = 2(var = 4)")

for x in [10, 12, 14, 16] : 
    print(f"x = {x:>2} ({abs(x-10)/2:.0f} std from the mean) : p(x) = {gauss_pdf(x,10,4):.5f}")


print(" Every extra standard deviation away makes p(x) drop fast.")
print("NUMERIC EXAMPLE 2: choosing epsilon on a tiny labeled cross-validation set")
print("=" * 74)
p_cv = np.array([0.20, 0.15, 0.10, 0.03, 0.01, 0.002])
y_cv = np.array([0, 0, 0, 1, 0, 1])
print(f"  p(x) values: {p_cv}")
print(f"  true labels: {y_cv}   (1 = real anomaly)")
for eps in [0.005, 0.02, 0.05]:
    flag = p_cv < eps
    tp = int(np.sum(flag & (y_cv == 1))); fp = int(np.sum(flag & (y_cv == 0))); fn = int(np.sum(~flag & (y_cv == 1)))
    P = tp / (tp + fp) if tp + fp else 0; R = tp / (tp + fn) if tp + fn else 0
    F = 2 * P * R / (P + R) if P + R else 0
    print(f"  epsilon = {eps:<6}: flagged {flag.sum()}  TP={tp} FP={fp} FN={fn}  precision={P:.2f} recall={R:.2f} F1={F:.2f}")
print("  -> epsilon = 0.05 has the best F1, so we pick it.")


# example 

def fit_independent(X):
    return X.mean(axis=0), X.var(axis=0)                      # variance with divide-by-m (maximum likelihood)
 
def logp_independent(X, mu, var):
    """log p(x) = sum of the per-feature log densities. We use LOGS so tiny numbers don't become 0."""
    return (-0.5 * np.log(2 * np.pi * var) - (X - mu) ** 2 / (2 * var)).sum(axis=1)
 
def fit_multivariate(X):
    return X.mean(axis=0), np.cov(X.T, bias=True)             # full covariance matrix (n x n)
 
def logp_multivariate(X, mu, cov):
    n = len(mu)
    diff = X - mu
    maha = np.einsum("ij,jk,ik->i", diff, np.linalg.inv(cov), diff)   # (x-mu)^T Sigma^-1 (x-mu)
    return -0.5 * (n * np.log(2 * np.pi) + np.linalg.slogdet(cov)[1] + maha)
 
def prf1(pred, y):
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f, tp, fp, fn
 
def select_threshold(logp_cv, y_cv):
    """Try every possible cut on the CV set; return the cut with the best F1 (placed between two scores)."""
    cand = np.unique(logp_cv)
    f1s = np.array([prf1(logp_cv <= s, y_cv)[2] for s in cand])
    i = int(np.argmax(f1s))
    thr = (cand[i] + cand[i + 1]) / 2 if i + 1 < len(cand) else cand[i]
    return thr, cand, f1s
 
# ---------------------------------------------------------------------------
# CHECK THE MATH AGAINST SCIPY / SKLEARN
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("CHECK: are my formulas right? (compared with scipy / sklearn)")
print("=" * 74)
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import f1_score
rc = np.random.default_rng(0)
Xc = rc.standard_normal((200, 3)) @ np.array([[1, 0.5, 0], [0, 1, 0.3], [0, 0, 1]]) + [1, 2, 3]
mu_c, cov_c = fit_multivariate(Xc)
print(f"1-D Gaussian pdf equals scipy.stats.norm.pdf:        {np.allclose(gauss_pdf(Xc[:, 0], mu_c[0], cov_c[0, 0]), norm.pdf(Xc[:, 0], mu_c[0], np.sqrt(cov_c[0, 0])))}")
print(f"multivariate log-pdf equals scipy multivariate_normal: {np.allclose(logp_multivariate(Xc, mu_c, cov_c), multivariate_normal(mu_c, cov_c).logpdf(Xc))}")
diag = np.diag(np.diag(cov_c))
print(f"multivariate with a DIAGONAL covariance == independent model: {np.allclose(logp_multivariate(Xc, mu_c, diag), logp_independent(Xc, mu_c, np.diag(cov_c)))}")
yy = (rc.random(200) < 0.1).astype(int); pp = rc.random(200) < 0.2
print(f"my F1 equals sklearn f1_score:                         {np.isclose(prf1(pp, yy)[2], f1_score(yy, pp))}")


# Synthetic dataset 

MU_A, SD_A = np.array([14.0, 15.0]), np.array([1.5, 1.6])      # throughput (mb/s), latency (ms)
def normal_A(n, r):
    return MU_A + SD_A * r.standard_normal((n, 2))
def anom_A(n, r):
    ang, rad = r.uniform(0, 2 * np.pi, n), r.uniform(3.0, 5.5, n)          # 3 to 5.5 std away, any direction
    return MU_A + SD_A * np.c_[rad * np.cos(ang), rad * np.sin(ang)]
 
MU_B, SD_B, RHO = np.array([50.0, 60.0]), np.array([10.0, 8.0]), 0.9       # CPU %, memory % move TOGETHER
LB = np.linalg.cholesky(np.array([[1, RHO], [RHO, 1]]))
def normal_B(n, r):
    return MU_B + SD_B * (r.standard_normal((n, 2)) @ LB.T)
def anom_B(n, r):
    a, sg = r.uniform(1.8, 3.0, n), r.choice([-1, 1], n)                    # CPU high but memory LOW (or reverse)
    return MU_B + SD_B * np.c_[sg * a, -sg * a * r.uniform(0.8, 1.0, n)]
 
def normal_C(n, r):      # feature 1 is SKEWED (like file sizes), feature 2 is bell-shaped
    return np.c_[np.exp(2.0 + 0.6 * r.standard_normal(n)), 10 + 2 * r.standard_normal(n)]
def anom_C(n, r):        # anomalies are 3.5-5.5 std away in LOG space
    ang, rad = r.uniform(0, 2 * np.pi, n), r.uniform(3.5, 5.5, n)
    return np.c_[np.exp(2.0 + 0.6 * rad * np.cos(ang)), 10 + 2 * rad * np.sin(ang)]
 
def make_sets(normal_fn, anom_fn, seed):
    r = np.random.default_rng(seed)
    Xtr = normal_fn(1000, r)                                                    # training: normal only
    Xcv = np.vstack([normal_fn(500, r), anom_fn(10, r)]); ycv = np.r_[np.zeros(500), np.ones(10)].astype(int)
    Xte = np.vstack([normal_fn(500, r), anom_fn(10, r)]); yte = np.r_[np.zeros(500), np.ones(10)].astype(int)
    return Xtr, Xcv, ycv, Xte, yte
 
def run(normal_fn, anom_fn, seed, model="indep", transform=None):
    Xtr, Xcv, ycv, Xte, yte = make_sets(normal_fn, anom_fn, seed)
    if transform is not None:
        Xtr, Xcv, Xte = transform(Xtr), transform(Xcv), transform(Xte)
    if model == "indep":
        mu, var = fit_independent(Xtr)
        score = lambda X: logp_independent(X, mu, var)
    else:
        mu, cov = fit_multivariate(Xtr)
        score = lambda X: logp_multivariate(X, mu, cov)
    thr, cand, f1s = select_threshold(score(Xcv), ycv)
    pred = score(Xte) <= thr
    return prf1(pred, yte), dict(score=score, thr=thr, cand=cand, f1s=f1s, Xte=Xte, yte=yte, pred=pred, Xtr=Xtr)
 
# ---------------------------------------------------------------------------
# PART 1: DATASET A (independent features)
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 1: server data (throughput, latency). Train 1000 normal | CV 500 normal + 10 anomalies | Test same")
print("=" * 74)
(pA, rA, fA, tpA, fpA, fnA), infoA = run(normal_A, anom_A, 0)
print(f"Chosen epsilon (as log p): {infoA['thr']:.3f}   (p = {np.exp(infoA['thr']):.2e})")
print(f"TEST: TP={tpA} FP={fpA} FN={fnA} TN={510 - tpA - fpA - fnA}   precision={pA:.2f} recall={rA:.2f} F1={fA:.2f}")
print(f"A model that says 'everything is normal' would score {500/510*100:.1f}% ACCURACY and find 0 anomalies.")
print("Not perfect: some real anomalies sit inside the normal range and some normal points are rare ones.")
 
# ---------------------------------------------------------------------------
# PART 2: log p and underflow
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 2: why we use LOG probabilities")
print("=" * 74)
r1 = np.random.default_rng(1)
X1k = r1.standard_normal((5, 1000))
prod = np.prod(gauss_pdf(X1k, 0.0, 1.0), axis=1)
print("Typical normal examples with 1000 features:")
print(f"  p(x) as a product of 1000 numbers : {prod}   <- all underflow to 0.0")
print(f"  log p(x) as a SUM of 1000 numbers  : {np.round(logp_independent(X1k, np.zeros(1000), np.ones(1000)), 1)}")
print("  Compare log p to a log threshold - never multiply many small numbers.")
 
# ---------------------------------------------------------------------------
# PART 3: CORRELATED FEATURES
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 3: correlated features - anomalies that hide from the independent model")
print("=" * 74)
print("CPU% and memory% normally rise TOGETHER. An anomaly: CPU high but memory LOW.")
print("Each feature alone looks fine (within ~3 std), only the COMBINATION is strange.\n")
N_SEEDS = 20
def summarize(name, fn_n, fn_a, model, transform=None):
    res = np.array([run(fn_n, fn_a, s, model, transform)[0][:3] for s in range(N_SEEDS)])
    print(f"{name:<40} | precision {res[:,0].mean():.2f}+-{res[:,0].std():.2f} | recall {res[:,1].mean():.2f}+-{res[:,1].std():.2f} | F1 {res[:,2].mean():.2f}+-{res[:,2].std():.2f}")
    return res[:, 2]
print(f"(mean +- std over {N_SEEDS} different random datasets - with only 10 anomalies one run is noisy)\n")
f1_A = summarize("A: independent features, independent model", normal_A, anom_A, "indep")
f1_Bi = summarize("B: correlated, INDEPENDENT Gaussian model", normal_B, anom_B, "indep")
f1_Bm = summarize("B: correlated, MULTIVARIATE Gaussian model", normal_B, anom_B, "mvg")
mu_b, cov_b = fit_multivariate(run(normal_B, anom_B, 0, "mvg")[1]["Xtr"])
_Xte, _yte = run(normal_B, anom_B, 0, "mvg")[1]["Xte"], run(normal_B, anom_B, 0, "mvg")[1]["yte"]
_d = _Xte - mu_b
d2 = np.einsum("ij,jk,ik->i", _d, np.linalg.inv(cov_b), _d)
print(f"\nSeed 0 test set, squared Mahalanobis distance (how 'odd' a point is): "
      f"largest NORMAL = {d2[_yte == 0].max():.1f}, smallest ANOMALY = {d2[_yte == 1].min():.1f}")
print("The perfect score is because these SYNTHETIC anomalies are extreme in the combined sense. Real anomalies are rarely this clean.")
print("\nThe multivariate model learns the covariance matrix: p(x) = exp(-0.5 (x-mu)^T Sigma^-1 (x-mu)) / ...")
print("It draws a TILTED ellipse around normal data; the independent model can only draw axis-aligned ones.")
print("Trade-off: multivariate needs more data (n*n covariance numbers) and Sigma must be invertible.")
print("Alternative: keep the cheap independent model and ADD a feature like cpu/memory.")
 
# ---------------------------------------------------------------------------
# PART 4: SKEWED FEATURES -> TRANSFORM
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("PART 4: skewed features - make them look Gaussian first")
print("=" * 74)
def skewness(x):
    return float(np.mean((x - x.mean()) ** 3) / x.std() ** 3)
Xt = make_sets(normal_C, anom_C, 0)[0]
print(f"Skewness of feature 1 (0 = symmetric bell): raw = {skewness(Xt[:, 0]):.2f}   after log = {skewness(np.log(Xt[:, 0])):.2f}")
logt = lambda X: np.c_[np.log(X[:, 0]), X[:, 1]]
f1_Cr = summarize("C: skewed feature, raw values", normal_C, anom_C, "indep")
f1_Cl = summarize("C: skewed feature, after log transform", normal_C, anom_C, "indep", logt)
print("Other transforms: sqrt(x), log(x + c), x^(1/3). Plot a histogram of each feature - it should look like a bell.")
 
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def draw_boundary(ax, info, xlim, ylim, color, label, fill=False):
    gx, gy = np.meshgrid(np.linspace(*xlim, 250), np.linspace(*ylim, 250))
    z = info["score"](np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    if fill:
        ax.contourf(gx, gy, np.maximum(z, info["thr"] - 15), levels=20, cmap="Blues", alpha=0.5)
    cs = ax.contour(gx, gy, z, levels=[info["thr"]], colors=color, linewidths=2.5)
    ax.plot([], [], color=color, linewidth=2.5, label=label)
 
def scatter_test(ax, info):
    X, y, pred = info["Xte"], info["yte"], info["pred"]
    ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, alpha=0.5, color="tab:blue", label="normal")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], s=70, marker="x", color="red", linewidths=2, label="true anomaly")
    ax.scatter(X[pred, 0], X[pred, 1], s=150, facecolors="none", edgecolors="k", linewidths=1.3, label="flagged")
 
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
 
ax = axes[0, 0]
draw_boundary(ax, infoA, (4, 24), (5, 25), "black", "decision boundary", fill=True)
scatter_test(ax, infoA)
ax.set_title(f"A: independent model (test F1 = {fA:.2f})", fontweight="bold")
ax.set_xlabel("throughput (mb/s)"); ax.set_ylabel("latency (ms)"); ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.3)
 
ax = axes[0, 1]
ax.plot(infoA["cand"], infoA["f1s"], linewidth=1.5)
ax.axvline(infoA["thr"], color="red", linestyle=":", label="chosen epsilon")
ax.set_xlabel("threshold on log p(x)"); ax.set_ylabel("F1 on the cross-validation set")
ax.set_title("Choosing epsilon by F1", fontweight="bold"); ax.legend(); ax.grid(alpha=0.3)
 
(_, _, fBi0, *_), infoBi = run(normal_B, anom_B, 0, "indep")
(_, _, fBm0, *_), infoBm = run(normal_B, anom_B, 0, "mvg")
ax = axes[0, 2]
xl, yl = (10, 95), (25, 95)
gx, gy = np.meshgrid(np.linspace(*xl, 250), np.linspace(*yl, 250))
grid = np.c_[gx.ravel(), gy.ravel()]
ax.contour(gx, gy, infoBi["score"](grid).reshape(gx.shape), levels=[infoBi["thr"]], colors="tab:orange", linewidths=2.5)
ax.contour(gx, gy, infoBm["score"](grid).reshape(gx.shape), levels=[infoBm["thr"]], colors="tab:green", linewidths=2.5)
scatter_test(ax, infoBm)
ax.plot([], [], color="tab:orange", linewidth=2.5, label=f"independent model (F1 = {fBi0:.2f})")
ax.plot([], [], color="tab:green", linewidth=2.5, label=f"multivariate model (F1 = {fBm0:.2f})")
ax.set_title("B: correlated features (seed 0)", fontweight="bold")
ax.set_xlabel("CPU %"); ax.set_ylabel("memory %"); ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
 
Xr = make_sets(normal_C, anom_C, 0)[0]
ax = axes[1, 0]
ax.hist(Xr[:, 0], bins=40, color="tab:orange", alpha=0.8)
ax.set_title(f"Feature 1, raw (skewness {skewness(Xr[:,0]):.2f})", fontweight="bold"); ax.grid(alpha=0.3)
ax = axes[1, 1]
ax.hist(np.log(Xr[:, 0]), bins=40, color="tab:green", alpha=0.8)
ax.set_title(f"Feature 1 after log (skewness {skewness(np.log(Xr[:,0])):.2f})", fontweight="bold"); ax.grid(alpha=0.3)
 
ax = axes[1, 2]
labels = ["A\nindependent", "B\nindependent\nmodel", "B\nmultivariate\nmodel", "C\nraw", "C\nlog transform"]
allf = [f1_A, f1_Bi, f1_Bm, f1_Cr, f1_Cl]
ax.bar(labels, [f.mean() for f in allf], yerr=[f.std() for f in allf], capsize=4,
       color=["tab:blue", "tab:orange", "tab:green", "tab:orange", "tab:green"])
ax.set_ylim(0, 1.05); ax.set_ylabel(f"test F1 (mean +- std, {N_SEEDS} datasets)"); ax.grid(alpha=0.3, axis="y")
ax.set_title("Which modelling choice helps?", fontweight="bold")
 
plt.tight_layout()
plt.savefig("c3_anomaly_detection_results.png", dpi=160, bbox_inches="tight")
print("\nSaved: c3_anomaly_detection_results.png")
plt.show()

