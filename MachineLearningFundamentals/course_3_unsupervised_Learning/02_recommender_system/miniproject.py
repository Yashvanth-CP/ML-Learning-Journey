"""
COURSE 3 - MINI PROJECT: FARM INSIGHTS  (PCA + K-means + anomaly detection on soil/weather data)
================================================================================
Run:  python C3_MINI_PROJECT_farm_insights.py

GOAL (links to your TerraYield AI project)
  Given farm readings  [N, P, K, temperature, humidity, pH, rainfall]  we want to:
    1. GROUP farms into soil-climate ZONES                       (K-means, Topic 1)
    2. SEE the data in 2D and know which features matter        (PCA, Topic 7)
    3. CATCH wrong sensor readings automatically                 (anomaly detection, Topic 2)
    4. Give a short REPORT for any new farm                      (zone + warning + most unusual feature)

DATA
  - If a file  Crop_recommendation.csv  is in the SAME folder (columns N,P,K,temperature,humidity,ph,rainfall,label)
    the project uses it.  (I could not test with your real file, only with a file of the same format - see the notes.)
  - Otherwise it makes SYNTHETIC farms with 6 hidden zones, so the project runs anywhere. Synthetic data is NOT real
    agronomy - it only lets us check that the code works because we know the right answers.
"""
import csv
import os
from collections import Counter

import numpy as np
from sklearn.metrics import adjusted_rand_score, silhouette_score

CSV_PATH = "Crop_recommendation.csv"
MAKE_PLOT = True
WRITE_CSV = True                         # writes farm_zones_output.csv (zone + anomaly flag for every row)
SEED = 0
np.set_printoptions(precision=2, suppress=True, linewidth=130)
rng = np.random.default_rng(SEED)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
# a cheap first check every sensor system should have: physically possible values
VALID_LO = np.array([0, 0, 0, -10, 0, 0, 0], dtype=float)
VALID_HI = np.array([300, 300, 300, 60, 100, 14, 1000], dtype=float)


# ---------------------------------------------------------------------------
# STEP 0: data
# ---------------------------------------------------------------------------
def make_synthetic(n=2200):
    """6 hidden soil-climate zones. Columns follow FEATURES."""
    means = np.array([
        [90, 45, 45, 24, 82, 6.3, 230],     # zone 0
        [20, 60, 30, 28, 55, 6.8, 70],      # zone 1
        [110, 25, 190, 22, 82, 6.0, 110],   # zone 2
        [35, 130, 195, 18, 90, 5.9, 90],    # zone 3
        [60, 55, 20, 33, 60, 7.4, 45],      # zone 4
        [25, 25, 20, 27, 90, 5.4, 190],     # zone 5
    ], dtype=float)
    stds = np.array([15, 12, 12, 3, 8, 0.4, 30], dtype=float)
    z = rng.integers(0, len(means), n)
    X = means[z] + rng.standard_normal((n, 7)) * stds
    X = np.clip(X, [0, 5, 5, 8, 14, 3.5, 20], [140, 145, 205, 44, 100, 9.9, 300])
    return X, np.array([f"zone_{i}" for i in z])


def load_data(path):
    if os.path.exists(path):
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
        X = np.array([[float(r[c]) for c in FEATURES] for r in rows])
        labels = np.array([r.get("label", "?") for r in rows])
        return X, labels, True
    X, labels = make_synthetic()
    return X, labels, False


X_all, labels_all, REAL = load_data(CSV_PATH)
print("=" * 74)
print("STEP 0: DATA")
print("=" * 74)
if REAL:
    print(f"Loaded {CSV_PATH}: {len(X_all)} rows, {len(set(labels_all))} different labels (crops).")
else:
    print(f"{CSV_PATH} not found -> using SYNTHETIC data: {len(X_all)} farms, 6 hidden zones.")
    print("(put Crop_recommendation.csv in this folder to run on your real data)")
print("feature ranges in the data (min .. max):")
for j, name in enumerate(FEATURES):
    print(f"  {name:12s} {X_all[:, j].min():8.1f} .. {X_all[:, j].max():8.1f}   mean {X_all[:, j].mean():8.1f}")

# split: 60% train (fit everything), 20% validation (choose the alarm threshold), 20% test (final grade)
perm = rng.permutation(len(X_all))
n_tr, n_va = int(0.6 * len(X_all)), int(0.2 * len(X_all))
i_tr, i_va, i_te = perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]
Xtr, Xva, Xte = X_all[i_tr], X_all[i_va], X_all[i_te]
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)                      # scaling numbers come from TRAIN only
scale = lambda A: (A - mu) / sd
Ztr = scale(Xtr)
print(f"\nsplit: train {len(Xtr)} | validation {len(Xva)} | test {len(Xte)}.  Scaling uses train mean/std only.")
print("Why scale? Rainfall is in hundreds, pH in single digits. Without scaling rainfall would dominate\n"
      "both PCA and K-means (Topics 1 and 7).")


# ---------------------------------------------------------------------------
# STEP 1: PCA
# ---------------------------------------------------------------------------
def pca_fit(Z):
    C = np.cov(Z.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    flip = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])])
    return vals, vecs * np.where(flip == 0, 1, flip)


print("\n" + "=" * 74)
print("STEP 1: PCA - what are the main directions in the data?")
print("=" * 74)
pvals, pvecs = pca_fit(Ztr)
ratio = pvals / pvals.sum()
print("component | variance share | cumulative")
for i in range(len(FEATURES)):
    print(f"   PC{i + 1}    |     {ratio[i]:6.1%}     |  {ratio[:i + 1].sum():6.1%}")
for c in (0, 1):
    top = np.argsort(-np.abs(pvecs[:, c]))[:3]
    txt = ", ".join(f"{FEATURES[j]} ({pvecs[j, c]:+.2f})" for j in top)
    print(f"PC{c + 1} is mostly:  {txt}")
print("Reading it: a big weight (+ or -) means that feature drives that axis. Features with big weights in the same")
print("direction move together; opposite signs move in opposite directions.")
pca2 = lambda Z: Z @ pvecs[:, :2]


# ---------------------------------------------------------------------------
# STEP 2: K-means zones
# ---------------------------------------------------------------------------
def kmeans(Z, k, n_init=10, iters=100, seed=0):
    """K-means with k-means++ start and several restarts. Returns centers, labels, inertia."""
    r = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [Z[r.integers(len(Z))]]
        for _ in range(k - 1):
            d2 = ((Z[:, None, :] - np.array(centers)[None]) ** 2).sum(2).min(1)
            centers.append(Z[r.choice(len(Z), p=d2 / d2.sum())])
        C = np.array(centers)
        for _ in range(iters):
            lab = ((Z[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
            newC = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        lab = ((Z[:, None, :] - C[None]) ** 2).sum(2).argmin(1)
        inertia = ((Z - C[lab]) ** 2).sum()
        if best is None or inertia < best[2]:
            best = (C, lab, inertia)
    return best


print("\n" + "=" * 74)
print("STEP 2: K-MEANS - how many zones? (pick k with the silhouette score, higher = clearer groups)")
print("=" * 74)
print("   k |  inertia | silhouette")
sil = {}
for k in range(2, 11):
    C_, lab_, inert_ = kmeans(Ztr, k)
    sil[k] = silhouette_score(Ztr, lab_)
    print(f"  {k:2d} | {inert_:8.0f} |   {sil[k]:.3f}")
K = max(sil, key=sil.get)
centers, lab_tr, _ = kmeans(Ztr, K)
print(f"\nBest silhouette at k = {K}.  (Also look at the elbow in inertia - the two should roughly agree.)")
if REAL:
    ari = adjusted_rand_score(labels_all[i_tr], lab_tr)
    print(f"Do zones line up with the crop labels? adjusted Rand index = {ari:.2f}  (1 = same grouping, 0 = random). "
          "Several crops can share a zone, so 1.0 is not expected.")
else:
    ari = adjusted_rand_score(labels_all[i_tr], lab_tr)
    print(f"Check against the HIDDEN truth (only possible because the data is synthetic): adjusted Rand index = {ari:.2f} (1 = perfect)")

# K-means on only a few PCA components: does compressing hurt the zones?
print("\nK-means on PCA features instead of all 7 (same k):")
for kk in (2, 3, 4, 7):
    Zp = Ztr @ pvecs[:, :kk]
    _, lp, _ = kmeans(Zp, K)
    print(f"  {kk} components ({ratio[:kk].sum():5.1%} of variance): agreement with the full-feature zones (ARI) = "
          f"{adjusted_rand_score(lab_tr, lp):.2f}")

print("\nZONE PROFILES (average of each feature in ORIGINAL units):")
print("  zone | farms | " + " | ".join(f"{n[:8]:>8s}" for n in FEATURES) + (" | most common crops" if REAL else ""))
order = np.argsort(centers[:, 0])
zone_name = {j: chr(ord('A') + r_) for r_, j in enumerate(order)}      # A, B, C ... sorted by nitrogen
for j in order:
    rows = Xtr[lab_tr == j]
    extra = ""
    if REAL:
        extra = " | " + ", ".join(f"{c}" for c, _ in Counter(labels_all[i_tr][lab_tr == j]).most_common(3))
    print(f"   {zone_name[j]}   | {len(rows):5d} | " + " | ".join(f"{v:8.1f}" for v in rows.mean(0)) + extra)


# ---------------------------------------------------------------------------
# STEP 3: anomaly detection (bad sensor readings)
# ---------------------------------------------------------------------------
def fit_gauss(Z, lab, k, reg=1e-3):
    out = []
    for j in range(k):
        Zj = Z[lab == j]
        cov = np.cov(Zj.T) + reg * np.eye(Z.shape[1])
        out.append((Zj.mean(0), np.linalg.inv(cov), np.linalg.slogdet(cov)[1]))
    return out


def anomaly_score(Z, params):
    """log-density under the BEST-fitting zone. Low = does not fit any normal zone = suspicious."""
    d = Z.shape[1]
    logs = []
    for m_, inv, logdet in params:
        diff = Z - m_
        maha2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
        logs.append(-0.5 * (maha2 + logdet + d * np.log(2 * np.pi)))
    return np.max(logs, axis=0)


def inject_faults(X, frac=0.04, seed=0):
    """Simulated sensor faults: ONE feature of a reading is shifted by 2.5-5 (whole-dataset) standard deviations."""
    r = np.random.default_rng(seed)
    Xf, is_fault = X.copy(), np.zeros(len(X), dtype=bool)
    idx = r.choice(len(X), int(frac * len(X)), replace=False)
    feat = r.integers(0, X.shape[1], len(idx))
    shift = r.uniform(2.5, 5.0, len(idx)) * r.choice([-1, 1], len(idx)) * sd[feat]
    Xf[idx, feat] += shift
    is_fault[idx] = True
    return Xf, is_fault


def prf(flag, truth):
    tp = int(np.sum(flag & truth))
    p = tp / max(1, int(flag.sum()))
    rc = tp / max(1, int(truth.sum()))
    return p, rc, 2 * p * rc / max(1e-12, p + rc)


def best_threshold(score, truth):
    cands = np.unique(np.percentile(score, np.linspace(0.5, 30, 150)))
    f1s = [prf(score < c, truth)[2] for c in cands]
    return cands[int(np.argmax(f1s))]


print("\n" + "=" * 74)
print("STEP 3: CATCHING BAD SENSOR READINGS")
print("=" * 74)
Xva_f, fa_va = inject_faults(Xva, seed=1)
Xte_f, fa_te = inject_faults(Xte, seed=2)
inrange = np.all((Xte_f[fa_te] >= VALID_LO) & (Xte_f[fa_te] <= VALID_HI), axis=1).mean()
print(f"We copy the validation and test farms and corrupt 4% of the rows (one feature shifted by 2.5-5 std).")
print(f"Test set: {fa_te.sum()} faulty readings among {len(Xte_f)}.  {inrange:.0%} of the faulty readings are still inside the")
print("physically possible range, so a simple 'is pH between 0 and 14?' check alone would NOT catch them.")
print("Normal model = a Gaussian fitted on clean TRAIN data. Two versions:")
print("  GLOBAL     : one Gaussian for all farms")
print("  ZONE-AWARE : one Gaussian per K-means zone; a reading is normal if it fits ANY zone")
glob = fit_gauss(Ztr, np.zeros(len(Ztr), dtype=int), 1)
zone = fit_gauss(Ztr, lab_tr, K)
res = []
for name, params in (("global Gaussian", glob), ("zone-aware", zone)):
    s_va, s_te = anomaly_score(scale(Xva_f), params), anomaly_score(scale(Xte_f), params)
    eps = best_threshold(s_va, fa_va)                                  # threshold from validation faults
    res.append((name, "best F1 on validation", *prf(s_te < eps, fa_te)))
    eps_rule = np.percentile(anomaly_score(Ztr, params), 1.0)          # no fault labels in real life? flag the lowest 1%
    res.append((name, "lowest 1% of train (no labels)", *prf(s_te < eps_rule, fa_te)))
    if name == "zone-aware":
        zone_eps, zone_scores_te = eps, s_te
print(f"\n  model            | how the alarm threshold was chosen | precision | recall |   F1")
for name, how, p, rc, f in res:
    print(f"  {name:16s} | {how:34s} |   {p:5.2f}   |  {rc:5.2f} | {f:5.2f}")
print("precision = of the alarms, how many were real faults; recall = of the real faults, how many we caught.")
print("The numbers depend on how big the simulated faults are. Bigger faults are easier to catch, smaller ones are not.")

n_fp = int(np.sum((zone_scores_te < zone_eps) & ~fa_te))
print(f"\nZone-aware, test set: {int(np.sum((zone_scores_te < zone_eps) & fa_te))} of {fa_te.sum()} faults caught, "
      f"{n_fp} false alarms among {int((~fa_te).sum())} good readings.")


# ---------------------------------------------------------------------------
# STEP 4: farm report for new farms
# ---------------------------------------------------------------------------
def farm_report(values, title):
    x = np.array(values, dtype=float)
    z = scale(x[None, :])
    zone_id = int(((z - centers) ** 2).sum(1).argmin())
    score = float(anomaly_score(z, zone)[0])
    members = Ztr[lab_tr == zone_id]
    zscore = (z[0] - members.mean(0)) / members.std(0)             # how unusual is each feature FOR THIS ZONE
    worst = int(np.argmax(np.abs(zscore)))
    flagged = score < zone_eps
    print(f"\n  {title}")
    print("    readings :", ", ".join(f"{n}={v:g}" for n, v in zip(FEATURES, x)))
    print(f"    zone     : {zone_name[zone_id]}   (PCA 2D position: PC1 = {pca2(z)[0, 0]:+.2f}, PC2 = {pca2(z)[0, 1]:+.2f})")
    print(f"    anomaly score {score:8.1f}  (alarm if below {zone_eps:.1f})  -> {'WARNING: check the sensors' if flagged else 'looks normal'}")
    print(f"    most unusual feature for this zone: {FEATURES[worst]} ({zscore[worst]:+.1f} standard deviations from the zone average)")
    if REAL:
        print("    crops usually grown in this zone:", ", ".join(c for c, _ in Counter(labels_all[i_tr][lab_tr == zone_id]).most_common(3)))


print("\n" + "=" * 74)
print("STEP 4: FARM REPORTS FOR NEW FARMS  (edit the numbers and run again)")
print("=" * 74)
base_zone = int(np.argmax(np.bincount(lab_tr)))
typical = np.median(Xtr[lab_tr == base_zone], axis=0)
farm_report(typical, f"Farm 1: a typical farm of zone {zone_name[base_zone]}")
broken = typical.copy(); broken[FEATURES.index("ph")] = typical[FEATURES.index("ph")] + 3.0
farm_report(broken, "Farm 2: same farm, but the pH sensor reads 3 units too high")
odd = typical.copy(); odd[FEATURES.index("humidity")] = 20.0; odd[FEATURES.index("rainfall")] = 280.0
farm_report(odd, "Farm 3: very dry air (humidity 20) together with heavy rain (280)")

# ---------------------------------------------------------------------------
# OUTPUT FILE + FIGURE
# ---------------------------------------------------------------------------
Z_all = scale(X_all)
zone_idx_all = ((Z_all[:, None, :] - centers[None]) ** 2).sum(2).argmin(1)
zone_all = np.array([zone_name[j] for j in zone_idx_all])
score_all = anomaly_score(Z_all, zone)
if WRITE_CSV:
    with open("farm_zones_output.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(FEATURES + ["zone", "anomaly_score", "flagged"] + (["label"] if REAL else []))
        for i in range(len(X_all)):
            w.writerow([f"{v:g}" for v in X_all[i]] + [zone_all[i], f"{score_all[i]:.2f}", int(score_all[i] < zone_eps)]
                       + ([labels_all[i]] if REAL else []))
    print(f"\nSaved: farm_zones_output.csv  ({len(X_all)} rows; {int((score_all < zone_eps).sum())} flagged by the zone-aware alarm)")
    print("  note: this file includes the TRAIN rows, so a few real-but-unusual farms may be flagged - check them by hand.")

if MAKE_PLOT:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    P = pca2(Ztr)
    ax[0].scatter(P[:, 0], P[:, 1], c=lab_tr, cmap="tab10", s=8)
    ax[0].scatter(*pca2(centers).T, c="k", marker="X", s=140, label="zone centers")
    ax[0].set(title=f"Farms in PCA 2D, colored by K-means zone (k={K})", xlabel="PC1", ylabel="PC2")
    ax[0].legend()
    ax[1].plot(list(sil.keys()), list(sil.values()), "o-")
    ax[1].axvline(K, color="r", ls="--")
    ax[1].set(title="Silhouette score vs k", xlabel="k", ylabel="silhouette")
    ax[1].grid(alpha=0.3)
    ax[2].hist(zone_scores_te[~fa_te], bins=50, alpha=0.7, label="good readings")
    ax[2].hist(zone_scores_te[fa_te], bins=20, alpha=0.7, label="faulty readings")
    ax[2].axvline(zone_eps, color="k", ls="--", label="alarm threshold")
    ax[2].set(title="Anomaly score on the test set (low = suspicious)", xlabel="log density")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig("c3_mini_project.png", dpi=150, bbox_inches="tight")
    print("Saved: c3_mini_project.png")
    plt.show()

print("\n" + "=" * 74)
print("WHAT TO TRY NEXT (ideas for your portfolio)")
print("=" * 74)
print("""
  1. Run it on your real Crop_recommendation.csv and read the zone profiles - do the zones make agronomic sense?
  2. Replace the simulated faults with real bad readings if you ever collect sensor data.
  3. Add the zone as an input feature to your crop-recommendation model (TerraYield) and compare accuracy.
  4. Use the anomaly alarm as a data-cleaning step BEFORE training any model.
  5. Honest limit: K-means finds ROUND groups and Gaussians assume bell-shaped features. Rainfall is usually skewed,
     so try np.log1p(rainfall) (Topic 2 showed this fixes skewed features).
""")