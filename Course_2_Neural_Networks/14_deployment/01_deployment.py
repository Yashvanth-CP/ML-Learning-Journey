"""
 DEPLOYMENT
 """



import json
import os
import sys
import time
from datetime import datetime
 
import numpy as np
import matplotlib.pyplot as plt
 
HERE = os.path.dirname(os.path.abspath(__file__))          # always relative to THIS file, not the terminal's folder
MODEL_DIR = os.path.join(HERE, "deployed_model")
sys.path.insert(0, HERE)
 


FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
CLASSES = ["rice", "wheat", "maize"]
MEANS = np.array([[80, 48, 40, 24, 82, 6.4, 236],
                  [50, 55, 40, 17, 65, 6.8, 75],
                  [78, 48, 20, 22, 65, 6.2, 85]], dtype=float)
STDS = np.array([[12, 8, 8, 2.5, 6, 0.5, 40],
                 [12, 8, 8, 3.0, 8, 0.6, 25],
                 [12, 8, 6, 3.0, 8, 0.6, 25]], dtype=float)
 
rng = np.random.default_rng(0)
Xs, ys = [], []
for k in range(3):
    Xs.append(MEANS[k] + STDS[k] * rng.standard_normal((400, 7)))
    ys.append(np.full(400, k))
X_all, y_all = np.vstack(Xs), np.concatenate(ys)
X_all[:, 5] = np.clip(X_all[:, 5], 3.5, 9.5); X_all[:, 4] = np.clip(X_all[:, 4], 14, 100)
X_all = np.maximum(X_all, 0)
perm = rng.permutation(len(X_all))
X_all, y_all = X_all[perm], y_all[perm]
Xtr_raw, ytr, Xte_raw, yte = X_all[:840], y_all[:840], X_all[840:], y_all[840:]
print(f"Synthetic data: train {Xtr_raw.shape}, test {Xte_raw.shape}; raw feature ranges differ hugely:")
print(f"  ph spans ~{X_all[:,5].min():.1f}-{X_all[:,5].max():.1f}, rainfall spans ~{X_all[:,6].min():.0f}-{X_all[:,6].max():.0f}")
 
# ---------------------------------------------------------------------------
# TRAINING (same NumPy network as earlier topics)
# ---------------------------------------------------------------------------
def init_params(sizes, seed=0):
    r = np.random.default_rng(seed)
    p = {}
    for l in range(1, len(sizes)):
        p[f"W{l}"] = r.standard_normal((sizes[l - 1], sizes[l])) * np.sqrt(2.0 / sizes[l - 1])
        p[f"b{l}"] = np.zeros((1, sizes[l]))
    return p
 
def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)
 
def forward(X, p, L):
    A, As, Zs = X, [X], []
    for l in range(1, L + 1):
        Z = A @ p[f"W{l}"] + p[f"b{l}"]
        A = softmax(Z) if l == L else np.maximum(0, Z)
        Zs.append(Z); As.append(A)
    return A, As, Zs
 
def train(sizes, X, Y, lr=0.01, epochs=80, batch=32, seed=0):
    L = len(sizes) - 1
    p = init_params(sizes, seed)
    m_ = {k: np.zeros_like(v) for k, v in p.items()}; v_ = {k: np.zeros_like(v) for k, v in p.items()}
    r = np.random.default_rng(seed); t = 0
    for _ in range(epochs):
        order = r.permutation(len(X))
        for s in range(0, len(X), batch):
            idx = order[s:s + batch]
            _, As, Zs = forward(X[idx], p, L)
            m = len(idx); dZ = As[L] - Y[idx]; g = {}
            for l in range(L, 0, -1):
                g[f"W{l}"] = As[l - 1].T @ dZ / m
                g[f"b{l}"] = dZ.sum(axis=0, keepdims=True) / m
                if l > 1:
                    dZ = (dZ @ p[f"W{l}"].T) * (Zs[l - 2] > 0)
            t += 1
            for k in p:
                m_[k] = 0.9 * m_[k] + 0.1 * g[k]; v_[k] = 0.999 * v_[k] + 0.001 * g[k] ** 2
                p[k] -= lr * (m_[k] / (1 - 0.9 ** t)) / (np.sqrt(v_[k] / (1 - 0.999 ** t)) + 1e-8)
    return p
 
print("\n" + "=" * 74)
print("STEP 1: train (with standardization) and record what we must save")
print("=" * 74)
mean = Xtr_raw.mean(axis=0)
std = np.maximum(Xtr_raw.std(axis=0), 1e-8)                 # guard against std = 0
sizes = [7, 32, 16, 3]
L = len(sizes) - 1
params = train(sizes, (Xtr_raw - mean) / std, np.eye(3)[ytr])
train_acc = float(np.mean(forward((Xtr_raw - mean) / std, params, L)[0].argmax(1) == ytr))
test_acc = float(np.mean(forward((Xte_raw - mean) / std, params, L)[0].argmax(1) == yte))
print(f"Train accuracy {train_acc*100:.1f}%   Test accuracy {test_acc*100:.1f}%")
 
# ---------------------------------------------------------------------------
# STEP 2: SAVE
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 2: save the model package")
print("=" * 74)
os.makedirs(MODEL_DIR, exist_ok=True)
arrays = {f"W{l}": params[f"W{l}"] for l in range(1, L + 1)}
arrays.update({f"b{l}": params[f"b{l}"] for l in range(1, L + 1)})
np.savez(os.path.join(MODEL_DIR, "model_v1.npz"), mean=mean, std=std, **arrays)
meta = {
    "version": "v1",
    "created": datetime.now().isoformat(timespec="seconds"),
    "n_layers": L, "layer_sizes": sizes,
    "features": FEATURES, "classes": CLASSES,
    "feature_min": {f: float(Xtr_raw[:, i].min()) for i, f in enumerate(FEATURES)},
    "feature_max": {f: float(Xtr_raw[:, i].max()) for i, f in enumerate(FEATURES)},
    "n_train": int(len(Xtr_raw)), "test_accuracy": round(test_acc, 4),
    "note": "SYNTHETIC demo data",
}
with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2)
for fn in sorted(os.listdir(MODEL_DIR)):
    print(f"  {fn:<15} {os.path.getsize(os.path.join(MODEL_DIR, fn)):>6,} bytes")
n_params = sum(a.size for k, a in arrays.items())
print(f"  parameters: {n_params:,}")
 
# ---------------------------------------------------------------------------
# STEP 3: LOAD IN A *FRESH* OBJECT AND VERIFY
# ---------------------------------------------------------------------------
from deployment_serve_flask import Model, create_app
 
print("\n" + "=" * 74)
print("STEP 3: load the saved package and check it reproduces the training-time predictions")
print("=" * 74)
model = Model(MODEL_DIR)
ref = forward((Xte_raw - mean) / std, params, L)[0]
loaded = model.predict_proba(Xte_raw)
print(f"Loaded model gives the same probabilities as the in-memory model: {np.allclose(ref, loaded)}")
print(f"Max difference: {np.abs(ref - loaded).max():.2e}")
 
# ---------------------------------------------------------------------------
# STEP 4: THE CLASSIC DEPLOYMENT BUGS
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 4: classic deployment bugs (same model, same test data - only the pipeline differs)")
print("=" * 74)
def acc_of(m, X):
    return float(np.mean(m.predict_proba(X).argmax(1) == yte))
 
no_scaler = Model(MODEL_DIR); no_scaler.mean = np.zeros(7); no_scaler.std = np.ones(7)
swapped = Xte_raw[:, [1, 0, 2, 3, 4, 5, 6]]                         # N and P swapped
shift = np.zeros(7); shift[6] = 2 * std[6]; shift[3] = 1 * std[3]   # rainfall sensor +2 std, temperature +1 std
drifted = Xte_raw + shift
results = [("Correct pipeline", acc_of(model, Xte_raw)),
           ("Forgot the scaler", acc_of(no_scaler, Xte_raw)),
           ("Wrong feature order (N<->P)", acc_of(model, swapped)),
           ("Drifted inputs (2 sensors off)", acc_of(model, drifted))]
for name, a in results:
    print(f"  {name:<32} accuracy = {a*100:5.1f}%")
print("The model file was perfect in every row - the PIPELINE around it broke it.")
 
# ---------------------------------------------------------------------------
# STEP 5: SPEED
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 5: speed - one request at a time vs a batch (times depend on your computer)")
print("=" * 74)
records = [dict(zip(FEATURES, map(float, row))) for row in Xte_raw[:300]]
t0 = time.perf_counter()
for _ in range(5):
    for rec in records:
        model.predict_record(rec)
single_ms = (time.perf_counter() - t0) / (5 * len(records)) * 1000
Xbig = np.tile(Xte_raw, (20, 1))[:5000]
t0 = time.perf_counter()
for _ in range(5):
    model.predict_proba(Xbig)
batch_ms = (time.perf_counter() - t0) / (5 * len(Xbig)) * 1000
print(f"  one record at a time (incl. validation): {single_ms:.4f} ms per prediction")
print(f"  batch of {len(Xbig)} at once              : {batch_ms:.5f} ms per prediction  (~{single_ms/batch_ms:.0f}x faster per item)")
print("  Vectorization again (Topic 1): process many records together when you can.")
 
# ---------------------------------------------------------------------------
# STEP 6: MONITORING INPUT DRIFT (works WITHOUT labels)
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 6: monitoring - has the live data moved away from the training data?")
print("=" * 74)
ZLIM = 0.25
def drift_report(live, name):
    z = (live.mean(axis=0) - mean) / std
    print(f"\n  {name}  (n={len(live)}; natural wobble is only ~{1/np.sqrt(len(live)):.2f} std)")
    for f, zi in zip(FEATURES, z):
        flag = "  <-- ALERT" if abs(zi) > ZLIM else ""
        print(f"    {f:<12} shift = {zi:+.2f} std{flag}")
    return z
z_ok = drift_report(Xte_raw, "Live batch A (healthy)")
z_bad = drift_report(drifted, "Live batch B (two sensors off)")
print(f"\n  Rule used here: alert when a feature's average moves more than {ZLIM} training-std.")
print("  Real systems also track: prediction mix, confidence, error rate (when labels arrive), latency.")
 
# ---------------------------------------------------------------------------
# STEP 7: THE REST API (tested in-process with Flask's test client)
# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("STEP 7: REST API with Flask (deployment_serve_flask.py)")
print("=" * 74)
client = create_app(MODEL_DIR).test_client()
good = dict(zip(FEATURES, [85.0, 45.0, 40.0, 24.0, 82.0, 6.4, 230.0]))
 
r = client.get("/health")
print(f"GET  /health                    -> {r.status_code} {r.get_json()}")
r = client.post("/predict", json={"features": good})
print(f"POST /predict (valid record)    -> {r.status_code} {r.get_json()}")
bad_missing = {k: v for k, v in good.items() if k != "ph"}
r = client.post("/predict", json={"features": bad_missing})
print(f"POST /predict (missing 'ph')    -> {r.status_code} {r.get_json()}")
r = client.post("/predict", json={"features": {**good, "ph": "acidic"}})
print(f"POST /predict (ph = 'acidic')   -> {r.status_code} {r.get_json()}")
r = client.post("/predict", json={"features": {**good, "rainfall": 5000.0}})
print(f"POST /predict (rainfall 5000)   -> {r.status_code} {r.get_json()}")
r = client.post("/predict", data="not json", content_type="text/plain")
print(f"POST /predict (not JSON)        -> {r.status_code} {r.get_json()}")
 
print("""
To run the REAL server on your computer:
  1. Run THIS file once (it creates the 'deployed_model' folder next to it)
  2. python deployment_serve_flask.py          -> server on http://127.0.0.1:5000
  3. In another terminal (Python):
       import requests
       r = requests.post("http://127.0.0.1:5000/predict", json={"features": {
           "N": 85, "P": 45, "K": 40, "temperature": 24, "humidity": 82, "ph": 6.4, "rainfall": 230}})
       print(r.json())
""")
 
# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
names = ["Correct", "No scaler", "Wrong\nfeature order", "Drifted\ninputs"]
vals = [a for _, a in results]
ax.bar(names, np.array(vals) * 100, color=["tab:green", "tab:red", "tab:red", "tab:orange"])
for i, v in enumerate(vals):
    ax.text(i, v * 100 + 1, f"{v*100:.0f}%", ha="center", fontweight="bold")
ax.set_ylim(0, 110); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=0.3, axis="y")
ax.set_title("Same model file, broken pipeline", fontweight="bold")
 
ax = axes[1]
x_ = np.arange(7)
ax.bar(x_ - 0.2, z_ok, 0.4, label="batch A (healthy)", color="tab:green")
ax.bar(x_ + 0.2, z_bad, 0.4, label="batch B (sensors off)", color="tab:red")
ax.axhline(ZLIM, color="k", linestyle=":"); ax.axhline(-ZLIM, color="k", linestyle=":")
ax.set_xticks(x_); ax.set_xticklabels(FEATURES, rotation=30)
ax.set_ylabel("mean shift (training std units)"); ax.legend(); ax.grid(alpha=0.3, axis="y")
ax.set_title("Drift monitor (dotted = alert limits)", fontweight="bold")
 
ax = axes[2]
ax.bar(["one at a time", "in a batch"], [single_ms, batch_ms], color=["tab:blue", "tab:green"])
ax.set_yscale("log"); ax.set_ylabel("ms per prediction (log scale)"); ax.grid(alpha=0.3, axis="y")
ax.set_title("Inference speed (your machine will differ)", fontweight="bold")
 
plt.tight_layout()
plt.savefig("deployment_results.png", dpi=200, bbox_inches="tight")
print("Saved: deployment_results.png")
plt.show()