import copy
import time
import numpy as np
import matplotlib.pyplot as plt

v_u_ex, v_m_ex, mean_ex = np.array([0.8, 0.2]), np.array([0.5, 0.9]), 3.2
print(f"  v_u = {v_u_ex}   v_m = {v_m_ex}   average rating = {mean_ex}")
print(f"  v_u . v_m = 0.8*0.5 + 0.2*0.9 = {v_u_ex @ v_m_ex:.2f}")
print(f"  predicted rating = {mean_ex} + {v_u_ex @ v_m_ex:.2f} = {mean_ex + v_u_ex @ v_m_ex:.2f}")

# SYNTHETIC DATA

GENRES = ["action", "comedy", "drama", "scifi", "romance"]
G = len(GENRES)
rng = np.random.default_rng(0)
n_u, n_m = 300, 200
 
genres = np.zeros((n_m, G))                       # each movie has 1 or 2 genres
for i in range(n_m):
    genres[i, rng.choice(G, rng.choice([1, 2], p=[0.6, 0.4]), replace=False)] = 1
year_n = (rng.uniform(1980, 2020, n_m) - 2000) / 12
quality = rng.normal(0, 0.4, n_m)                 # hidden movie quality (NOT available as a feature)
taste = rng.normal(0, 1, (n_u, G))                # hidden genre taste of each user
newer = rng.normal(0, 0.5, n_u)                   # hidden 'likes newer films' trait
affinity = taste + rng.normal(0, 0.3, (n_u, G))   # what we OBSERVE: noisy genre history ...
age_n = -1.2 * newer + rng.normal(0, 0.3, n_u)    # ... and age (young users like newer films)
 
clean = (3.2 + (taste @ genres.T) / genres.sum(1)[None, :] + newer[:, None] * year_n[None, :] + quality[None, :])
Y = np.clip(np.round((clean + rng.normal(0, 0.3, clean.shape)) * 2) / 2, 0.5, 5.0)     # (n_u, n_m)
observed = rng.random((n_u, n_m)) < 0.25
Xu_raw = np.c_[affinity, age_n]                   # (n_u, 6)
Xi_raw = np.c_[genres, year_n]                    # (n_m, 6)
 
new_u = np.arange(n_u) >= 240                     # the last 60 users are NEW (no ratings used in training)
new_i = np.arange(n_m) >= 160                     # the last 40 movies are NEW
ee = observed & ~new_u[:, None] & ~new_i[None, :]
idx = np.argwhere(ee)
rng.shuffle(idx)
n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
 
 
def to_mask(rows):
    m = np.zeros((n_u, n_m), dtype=bool)
    m[rows[:, 0], rows[:, 1]] = True
    return m
 
 
M_tr, M_va, M_te = to_mask(idx[:n_tr]), to_mask(idx[n_tr:n_tr + n_va]), to_mask(idx[n_tr + n_va:])
M_nu = observed & new_u[:, None] & ~new_i[None, :]
M_ni = observed & ~new_u[:, None] & new_i[None, :]
M_nn = observed & new_u[:, None] & new_i[None, :]
 
# standardize features using EXISTING users/items only
mu_u, sd_u = Xu_raw[~new_u].mean(0), Xu_raw[~new_u].std(0)
mu_i, sd_i = Xi_raw[~new_i].mean(0), Xi_raw[~new_i].std(0)
Xu, Xi = (Xu_raw - mu_u) / sd_u, (Xi_raw - mu_i) / sd_i
 
print("\n" + "=" * 74)
print("DATA")
print("=" * 74)
print(f"{n_u} users x {n_m} movies; {observed.mean()*100:.0f}% of the cells are rated. 6 features for users, 6 for movies.")
print(f"Training ratings: {M_tr.sum():,} (existing users x existing movies only). Validation {M_va.sum():,}, test {M_te.sum():,}")
print("COLD-START test sets (NOTHING of these users/movies was seen in training):")
print(f"  new users x known movies: {M_nu.sum():,} ratings | known users x new movies: {M_ni.sum():,} | new users x new movies: {M_nn.sum():,}")


# NETWORK CODE
def init_mlp(sizes, r):
    return [[r.standard_normal((a, b)) * np.sqrt(2.0 / a), np.zeros(b)] for a, b in zip(sizes[:-1], sizes[1:])]
 
 
def mlp_fwd(X, layers):
    acts, pre, A = [X], [], X
    for l, (W, b) in enumerate(layers):
        Z = A @ W + b
        pre.append(Z)
        A = Z if l == len(layers) - 1 else np.maximum(0, Z)       # ReLU hidden, linear output
        acts.append(A)
    return A, (acts, pre)
 
 
def mlp_bwd(dOut, cache, layers):
    acts, pre = cache
    grads, d = [None] * len(layers), dOut
    for l in range(len(layers) - 1, -1, -1):
        if l < len(layers) - 1:
            d = d * (pre[l] > 0)
        grads[l] = [acts[l].T @ d, d.sum(axis=0)]
        d = d @ layers[l][0].T
    return grads
 
 
def batch_loss_and_grads(Lu, Li, ub, ib, yb, mean_r):
    vu, cu = mlp_fwd(Xu[ub], Lu)
    vm, cm = mlp_fwd(Xi[ib], Li)
    err = mean_r + (vu * vm).sum(1) - yb
    loss = 0.5 * np.mean(err ** 2)
    dp = err / len(yb)
    return loss, mlp_bwd(dp[:, None] * vm, cu, Lu), mlp_bwd(dp[:, None] * vu, cm, Li)
 
 
def init_adam(layers):
    return ([[np.zeros_like(W), np.zeros_like(b)] for W, b in layers],
            [[np.zeros_like(W), np.zeros_like(b)] for W, b in layers])
 
 
def adam_apply(layers, grads, ms, vs, t, lr):
    for l in range(len(layers)):
        for k in range(2):
            ms[l][k] = 0.9 * ms[l][k] + 0.1 * grads[l][k]
            vs[l][k] = 0.999 * vs[l][k] + 0.001 * grads[l][k] ** 2
            layers[l][k] -= lr * (ms[l][k] / (1 - 0.9 ** t)) / (np.sqrt(vs[l][k] / (1 - 0.999 ** t)) + 1e-8)
 
 
def embed_all(Lu, Li):
    return mlp_fwd(Xu, Lu)[0], mlp_fwd(Xi, Li)[0]
 
 
def predict_matrix(Lu, Li, mean_r, clip=True):
    Vu, Vm = embed_all(Lu, Li)
    P = mean_r + Vu @ Vm.T
    return np.clip(P, 0.5, 5.0) if clip else P
 
 
def rmse(P, mask):
    return float(np.sqrt(np.mean((P[mask] - Y[mask]) ** 2)))

# GRADIENT CHECK

print("CHECK: is the two-tower backprop correct? (numerical gradient vs backprop)")
print("=" * 74)
rc = np.random.default_rng(3)
Lu_c, Li_c = init_mlp([6, 8, 4, 3], rc), init_mlp([6, 8, 4, 3], rc)
ub_c, ib_c = rc.integers(0, n_u, 16), rc.integers(0, n_m, 16)
yb_c = Y[ub_c, ib_c]
_, gu, gi = batch_loss_and_grads(Lu_c, Li_c, ub_c, ib_c, yb_c, 3.2)
worst = 0.0
for layers, grads in [(Lu_c, gu), (Li_c, gi)]:
    for l in range(len(layers)):
        for k in range(2):
            arr = layers[l][k]
            for pos in [tuple(int(rc.integers(0, s)) for s in arr.shape) for _ in range(4)]:
                old = arr[pos]
                arr[pos] = old + 1e-6; lp = batch_loss_and_grads(Lu_c, Li_c, ub_c, ib_c, yb_c, 3.2)[0]
                arr[pos] = old - 1e-6; lm = batch_loss_and_grads(Lu_c, Li_c, ub_c, ib_c, yb_c, 3.2)[0]
                arr[pos] = old
                num = (lp - lm) / 2e-6
                worst = max(worst, abs(num - grads[l][k][pos]) / max(1e-8, abs(num) + abs(grads[l][k][pos])))
print(f"Worst relative error over sampled parameters of BOTH towers: {worst:.1e}  ->  {'PASS' if worst < 1e-5 else 'FAIL'}")

print("\n" + "=" * 74)
print("TRAINING: both towers 6 -> 32 -> 16 -> 8, Adam, batch 128, early stopping on validation")
print("=" * 74)
r = np.random.default_rng(1)
Lu, Li = init_mlp([6, 32, 16, 8], r), init_mlp([6, 32, 16, 8], r)
msu, vsu = init_adam(Lu)
msi, vsi = init_adam(Li)
mean_r = Y[M_tr].mean()
tri = np.argwhere(M_tr)
hist = {"train": [], "val": []}
best_val, best_state, wait, t = np.inf, None, 0, 0
MAX_EPOCHS = 300
for epoch in range(1, MAX_EPOCHS + 1):
    perm = r.permutation(len(tri))
    for s in range(0, len(tri), 128):
        sel = tri[perm[s:s + 128]]
        ub, ib = sel[:, 0], sel[:, 1]
        _, gu_, gi_ = batch_loss_and_grads(Lu, Li, ub, ib, Y[ub, ib], mean_r)
        t += 1
        adam_apply(Lu, gu_, msu, vsu, t, 0.002)
        adam_apply(Li, gi_, msi, vsi, t, 0.002)
    P = predict_matrix(Lu, Li, mean_r)
    hist["train"].append(rmse(P, M_tr))
    hist["val"].append(rmse(P, M_va))
    if hist["val"][-1] < best_val - 1e-4:
        best_val, best_state, wait = hist["val"][-1], (copy.deepcopy(Lu), copy.deepcopy(Li)), 0
    else:
        wait += 1
    if epoch % 20 == 0 or epoch == 1:
        print(f"  epoch {epoch:3d} | train RMSE {hist['train'][-1]:.3f} | validation RMSE {hist['val'][-1]:.3f}")
    if wait >= 12:
        print(f"  early stop at epoch {epoch} (validation has not improved for 12 epochs)")
        break
else:
    print(f"  stopped because the limit of {MAX_EPOCHS} epochs was reached (validation was still improving)")
Lu, Li = best_state
P = predict_matrix(Lu, Li, mean_r)
print(f"Best validation RMSE = {best_val:.3f}")
 
# ---------------------------------------------------------------------------
# RESULTS: normal + cold start
# ---------------------------------------------------------------------------
g_mean = Y[M_tr].mean()
item_mean = np.where(M_tr.sum(0) > 0, (Y * M_tr).sum(0) / np.maximum(M_tr.sum(0), 1), g_mean)   # new movies -> global mean
P_global, P_item = np.full(Y.shape, g_mean), np.tile(item_mean[None, :], (n_u, 1))
print("\n" + "=" * 74)
print("RESULTS (test RMSE - lower is better)")
print("=" * 74)
print(f"{'situation':<36} | {'ratings':>7} | {'global avg':>10} | {'movie avg':>9} | {'two-tower':>9}")
for name, m in [("known users, known movies (test)", M_te), ("NEW users, known movies", M_nu),
                ("known users, NEW movies", M_ni), ("NEW users, NEW movies", M_nn)]:
    print(f"{name:<36} | {m.sum():>7,} | {rmse(P_global, m):10.3f} | {rmse(P_item, m):9.3f} | {rmse(P, m):9.3f}")
print("""
Reading the table:
  - 'movie avg' knows each movie's average from other people (includes hidden QUALITY) but
    gives EVERYONE the same prediction - no personalization; for NEW movies it has nothing.
  - the two-tower model personalizes using features, so it works in ALL four situations.
  - It cannot see hidden movie quality (we gave it no ID or quality feature), which limits it.""")
 
# ---------------------------------------------------------------------------
# SIMILAR MOVIES + RETRIEVAL
# ---------------------------------------------------------------------------
Vu, Vm = embed_all(Lu, Li)
print("=" * 74)
print("SIMILAR MOVIES: smallest ||v_m(k) - v_m(i)||^2")
print("=" * 74)
D2 = ((Vm[:, None, :] - Vm[None, :, :]) ** 2).sum(-1)
np.fill_diagonal(D2, np.inf)
nn3 = np.argsort(D2, axis=1)[:, :3]
 
 
def share(a, b):
    return bool((genres[a] * genres[b]).sum() > 0)
 
 
nn_share = np.mean([share(i, k) for i in range(n_m) for k in nn3[i]])
rp = np.random.default_rng(5)
rand_share = np.mean([share(a, b) for a, b in rp.integers(0, n_m, (4000, 2)) if a != b])
print(f"  Share at least one genre with their 3 nearest neighbours: {nn_share*100:.0f}%  (random pairs: {rand_share*100:.0f}%)")
 
print("\n" + "=" * 74)
print("RETRIEVAL for the 60 NEW users: do we find their true top-10 movies?")
print("=" * 74)
 
 
def precision_at_10(scores):
    hits = []
    for u in np.where(new_u)[0]:
        hits.append(len(set(np.argsort(-clean[u])[:10]) & set(np.argsort(-scores[u])[:10])) / 10)
    return float(np.mean(hits))
 
 
raw_scores = predict_matrix(Lu, Li, mean_r, clip=False)
print(f"  random                  : {precision_at_10(np.random.default_rng(9).random(Y.shape))*100:5.1f}%")
print(f"  most popular (movie avg): {precision_at_10(P_item)*100:5.1f}%")
print(f"  two-tower (no ratings of these users used!): {precision_at_10(raw_scores)*100:5.1f}%")
 
N_BIG = 100_000
big = np.random.default_rng(2).standard_normal((N_BIG, 6))
t0 = time.perf_counter()
Vbig = mlp_fwd(big, Li)[0]
t_tower = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(20):
    sc = Vbig @ Vu[0]
t_dot = (time.perf_counter() - t0) / 20
print(f"\nSPEED with {N_BIG:,} movies (your computer will differ):")
print(f"  run the movie tower on all movies at request time: {t_tower*1000:7.2f} ms")
print(f"  precomputed movie vectors, just dot products     : {t_dot*1000:7.3f} ms   (~{t_tower/t_dot:.0f}x faster)")
 
# ---------------------------------------------------------------------------
# PLOTS (saved when YOU run the file)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
ax.plot(hist["train"], label="train RMSE", linewidth=2)
ax.plot(hist["val"], "--", label="validation RMSE", linewidth=2)
ax.set_xlabel("epoch")
ax.set_ylabel("RMSE")
ax.set_title("Training curves", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3)
 
ax = axes[1]
names = ["known/known", "NEW users", "NEW movies", "NEW/NEW"]
masks = [M_te, M_nu, M_ni, M_nn]
w = 0.27
ax.bar(np.arange(4) - w, [rmse(P_global, m) for m in masks], w, label="global average")
ax.bar(np.arange(4), [rmse(P_item, m) for m in masks], w, label="movie average")
ax.bar(np.arange(4) + w, [rmse(P, m) for m in masks], w, label="two-tower")
ax.set_xticks(range(4))
ax.set_xticklabels(names)
ax.set_ylabel("test RMSE")
ax.set_title("Cold start: only the two-tower personalizes", fontweight="bold")
ax.legend()
ax.grid(alpha=0.3, axis="y")
 
ax = axes[2]
ax.scatter(Y[M_nu], P[M_nu], s=8, alpha=0.4)
ax.plot([0.5, 5], [0.5, 5], "r--")
ax.set_xlabel("actual rating")
ax.set_ylabel("predicted rating")
ax.set_title(f"NEW users, known movies (RMSE {rmse(P, M_nu):.2f})", fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("c3_content_based.png", dpi=150, bbox_inches="tight")
print("\nSaved: c3_content_based.png")
plt.show()
