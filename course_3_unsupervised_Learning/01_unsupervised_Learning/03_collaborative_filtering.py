#  COLLABORATIVE FILTERING

import numpy as np
import matplotlib.pyplot as plt

print(" Numeric Example  : ")

X_ex = np.array([[1.0, 0.0],[0.0, 1.0]])
W_ex = np.array([[5.0, 0.0], [0.0, 5.0]])
b_ex = np.zeros(2)
Y_ex = np.array([[5.0, 0.0], [1.0, 5.0]])
R_ex = np.array([[1.0], [1.1]])
P_ex = X_ex @ W_ex + b_ex
print(" Prediction ( rows = movies, cols = users): \n", P_ex)
print("actual rating: \n", Y_ex, " \nwhich cells were rated ( 1 = yes):\n", R_ex)

terms = ((P_ex - Y_ex) * R_ex) ** 2

print("squared errors on RATED cells only:\n", terms)
print(f"cost (no regularization) = 1/2 * {terms.sum():.1f} = {0.5 * terms.sum():.2f}")
print("The empty cell (movie 1, user 2) is ignored in the cost - but the model still PREDICTS it:", P_ex[0, 1])

# CODE: COST, GRADIENTS, TRAINING

def cf_cost(X, W, b, Yn, R, lam):
    E = (X @ W.T + b[None, : ] - Yn) * R
    return 0.5 * np.sum(E ** 2) + 0.5 * lam * (np.sum(X ** 2) + np.sum(W ** 2))

def cf_grads(X, W, b, Yn, R, lam):
    E = ( X @ W.T + b[None, :] - Yn) * R
    return E @ W + lam * X, E.T @ X + lam * W, E.sum(axis = 0)

def rmse(P, Y, mask): 
    return float(np.sqrt(np.mean((P[mask] - Y[mask]) ** 2)))

def fit_cf(Y, R, K, lam, iters=600, lr=0.05, seed = 0, mean_norm=True, Rval = None, every = 25):
    r = np.random.default_rng(seed)
    n_m, n_u = Y.shape
    cnt = R.sum(axis = 1)
    gmean = Y[R].mean()
    mu = np.where(cnt > 0, (Y * R).sum(axis=1) / np.maximum(cnt, 1), gmean) if mean_norm else np.zeros(n_m)
    Yn = Y - mu[:, None]
    X, W, b = 0.1 * r.standard_normal((n_m, K)), 0.1 * r.standard_normal((n_u, K)), np.zeros(n_u)
    params = dict(X = X, W = W, b = b)
    state = { k : [np.zeros_like(v), np.zeros_like(v)] for k, v in  params.items()}
    hist = {"it": [], "cost": [], "val": []}

    for t in range(1, iters + 1) : 
        dX, dW, db = cf_grads(params["X"], params["W"], params["b"], Yn, R, lam)
        for k, g in zip(["X", "W", "b"], [dX, dW, db]):
            m_, v_ = state[k]
            m_[:] = 0.9 * m_ + 0.1 * g
            v_[:] = 0.999 * v_ + 0.001 * g ** 2
            params[k] -= lr * (m_ / (1 - 0.9 ** t)) / (np.sqrt(v_ / (1 - 0.999 ** t)) + 1e-8)

        if t % every == 0 or t == 1 : 
            hist["it"].append(t)
            hist["cost"].append(cf_cost(params["X"], params["W"], params["b"], Yn, R, lam))

            if Rval is not None : 
                p = np.clip(params["X"] @ params["W"].T + params["b"][None, : ] + mu[:, None], 0.5, 5.0)
                hist["val"].append(rmse(p, Y, Rval))

    return dict(X=params["X"], W=params["W"], b=params["b"], mu=mu, hist=hist)

def predict_all(model, clip=True):
    P = model["X"] @ model["W"].T + model["b"][None, :] + model["mu"][:, None]
    return np.clip(P, 0.5, 5.0) if clip else P


# GRADIENT CHECK

rg = np.random.default_rng(1)
Xg, Wg, bg = rg.standard_normal((6, 2)), rg.standard_normal((4, 2)), rg.standard_normal(4)
Yg, Rg = rg.uniform(1, 5, (6, 4)), rg.random((6, 4)) < 0.6
dXg, dWg, dbg = cf_grads(Xg, Wg, bg, Yg, Rg, 0.7)
worst = 0.0

for arr, g in [(Xg, dXg), (Wg, dWg), (bg, dbg)]:
    for idx in np.ndindex(arr.shape):
        old = arr[idx]
        arr[idx] = old + 1e-6; cp = cf_cost(Xg, Wg, bg, Yg, Rg, 0.7)
        arr[idx] = old - 1e-6; cm = cf_cost(Xg, Wg, bg, Yg, Rg, 0.7)
        arr[idx] = old
        num = ( cp - cm) / 2e-6
        worst = max(worst, abs(num - g[idx]) / max(1e-8, abs(num) + abs(g[idx])))

    print(f"Worst relative error over ALL parameters; {worst:.1e}  -> {'PASS' if worst < 1e-6 else 'Fail'}")




# SYNTHETIC RATINGS

n_m, n_u, K_TRUE = 200, 100, 3
r0 = np.random.default_rng(0)
X_true = r0.normal(0, 0.8, (n_m, K_TRUE))             # hidden movie features (3 'genres')
W_true = r0.normal(0, 0.8, (n_u, K_TRUE))             # hidden user tastes
item_bias = r0.normal(0, 0.5, n_m)                    # some movies are simply better
user_bias = r0.normal(0, 0.3, n_u)                    # some users rate generously
clean = 3.2 + item_bias[:, None] + user_bias[None, :] + X_true @ W_true.T
Y = np.clip(np.round((clean + r0.normal(0, 0.3, clean.shape)) * 2) / 2, 0.5, 5.0)   # 0.5-star steps like MovieLens
observed = r0.random((n_m, n_u)) < 0.20
idx = np.argwhere(observed)
r0.shuffle(idx)
n_tr, n_va = int(0.70 * len(idx)), int(0.15 * len(idx))
 
 
def to_mask(rows):
    m = np.zeros((n_m, n_u), dtype=bool)
    m[rows[:, 0], rows[:, 1]] = True
    return m
 
 
R_tr, R_va, R_te = to_mask(idx[:n_tr]), to_mask(idx[n_tr:n_tr + n_va]), to_mask(idx[n_tr + n_va:])
print("\n" + "=" * 74)
print("DATA")
print("=" * 74)
print(f"{n_m} movies x {n_u} users = {n_m*n_u:,} cells, only {observed.mean()*100:.0f}% have a rating ({observed.sum():,} ratings)")
print(f"Split of the known ratings -> train {R_tr.sum():,} | validation {R_va.sum():,} | test {R_te.sum():,}")
print("(The true data was made from 3 hidden genres + movie quality + user generosity + noise.)")
 
# BASELINES

g_mean = Y[R_tr].mean()
item_mean = np.where(R_tr.sum(1) > 0, (Y * R_tr).sum(1) / np.maximum(R_tr.sum(1), 1), g_mean)
user_mean = np.where(R_tr.sum(0) > 0, (Y * R_tr).sum(0) / np.maximum(R_tr.sum(0), 1), g_mean)
P_global = np.full(Y.shape, g_mean)
P_item = np.tile(item_mean[:, None], (1, n_u))
P_user = np.tile(user_mean[None, :], (n_m, 1))
print("\n" + "=" * 74)
print("BASELINES (RMSE on the validation ratings - lower is better)")
print("=" * 74)
print(f"  predict the global average rating : {rmse(P_global, Y, R_va):.3f}")
print(f"  predict each MOVIE's average      : {rmse(P_item, Y, R_va):.3f}")
print(f"  predict each USER's average       : {rmse(P_user, Y, R_va):.3f}")
print("  (any smart model must beat these)")
 

# CHOOSE K AND LAMBDA ON VALIDATION

print("\n" + "=" * 74)
print("TUNING: number of features K and regularization lambda (validation RMSE)")
print("=" * 74)
Ks, lams = [1, 2, 3, 5, 10], [0.1, 1.0, 3.0, 10.0]
grid = {}
print(f"{'K':>3} | " + " | ".join(f"lam={l:<5}" for l in lams))
for K in Ks:
    row = []
    for lam in lams:
        m = fit_cf(Y, R_tr, K, lam)
        grid[(K, lam)] = rmse(predict_all(m), Y, R_va)
        row.append(grid[(K, lam)])
    print(f"{K:>3} | " + " | ".join(f"{v:9.3f}" for v in row))
bestK, bestlam = min(grid, key=grid.get)
print(f"\nBest on validation: K = {bestK}, lambda = {bestlam}   (true hidden number of genres = {K_TRUE})")
print("Too little regularization with many features -> overfits the few known ratings. (Topic 9 again!)")
 

# FINAL MODEL ON TEST

final = fit_cf(Y, R_tr, bestK, bestlam, Rval=R_va)
P = predict_all(final)
nm_final = fit_cf(Y, R_tr, bestK, bestlam, mean_norm=False)
print("\n" + "=" * 74)
print("FINAL RESULT (TEST ratings, never used for any choice)")
print("=" * 74)
print(f"  global average baseline : {rmse(P_global, Y, R_te):.3f}")
print(f"  movie-average baseline  : {rmse(P_item, Y, R_te):.3f}")
print(f"  collaborative filtering : {rmse(P, Y, R_te):.3f}     (train {rmse(P, Y, R_tr):.3f})")
print(f"  same model WITHOUT mean normalization (validation): {rmse(predict_all(nm_final), Y, R_va):.3f} vs {rmse(P, Y, R_va):.3f} with it")
print("  Floor: ratings have noise 0.3 + rounding to 0.5 stars, so ~0.33 is the best anyone could do.")
 

# COLD START: a NEW USER WITH NO RATINGS

print("\n" + "=" * 74)
print("NEW USER with ZERO ratings: why mean normalization matters")
print("=" * 74)
Y2 = np.hstack([Y, np.zeros((n_m, 1))])
R2 = np.hstack([R_tr, np.zeros((n_m, 1), dtype=bool)])
with_mn = fit_cf(Y2, R2, bestK, bestlam, mean_norm=True)
without = fit_cf(Y2, R2, bestK, bestlam, mean_norm=False)
pw = predict_all(with_mn)[:5, -1]
po = predict_all(without)[:5, -1]
print("Predicted ratings for the new user, movies 0-4:")
print(f"  with mean normalization    : {np.round(pw, 2)}   (= the movies' average ratings: {np.round(item_mean[:5], 2)})")
print(f"  WITHOUT mean normalization : {np.round(po, 2)}   (clipped to the lowest rating - useless)")
print("Cold start is the weakness of pure collaborative filtering - Topic 4 (content-based) helps with it.")
 
 
# LEARNED FEATURES: do they mean anything?

def pair_dists(F):
    D = np.sqrt(np.maximum(((F[:, None, :] - F[None, :, :]) ** 2).sum(-1), 0))
    return D[np.triu_indices(len(F), 1)]
 
 
def ranks(a):
    return np.argsort(np.argsort(a)).astype(float)
 
 
d_learned, d_true = pair_dists(final["X"]), pair_dists(X_true)
spearman = float(np.corrcoef(ranks(d_learned), ranks(d_true))[0, 1])
print("\n" + "=" * 74)
print("DID IT LEARN MEANINGFUL FEATURES?")
print("=" * 74)
print("The learned features x(i) are numbers nobody labelled. We CAN check them because we know the hidden truth:")
print(f"  rank correlation between 'distance of two movies in LEARNED features' and in the TRUE hidden genres = {spearman:.2f}")
print("  (1 = same ordering of which movies are alike). Related movies: smallest ||x(k) - x(i)||^2.")
i0 = 0
dd = ((final["X"] - final["X"][i0]) ** 2).sum(1)
dd[i0] = np.inf
near = np.argsort(dd)[:3]
tdd = ((X_true - X_true[i0]) ** 2).sum(1)
tdd[i0] = np.inf
true_near = set(np.argsort(tdd)[:20])
print(f"  movie {i0}: 3 most similar by learned features = {near.tolist()}  -> {sum(int(n in true_near) for n in near)} of 3 are among its 20 truly most similar")
 

# TOP-10 RECOMMENDATIONS QUALITY

seen = R_tr | R_va
P_raw = predict_all(final, clip=False)    # RANK with unclipped scores (clipping at 5 would create ties)
 
 
def precision_at_10(scorer):
    hits = []
    for j in range(n_u):
        cand = np.where(~seen[:, j])[0]
        truth = set(cand[np.argsort(-clean[cand, j])[:10]])      # the 10 unseen movies this user would truly like most
        pick = cand[np.argsort(-scorer[cand, j])[:10]]
        hits.append(len(truth & set(pick)) / 10)
    return float(np.mean(hits))
 
 
rr = np.random.default_rng(7)
print("\n" + "=" * 74)
print("TOP-10 RECOMMENDATIONS: how many of the user's true favourite 10 unseen movies do we find?")
print("=" * 74)
print(f"  random movies           : {precision_at_10(rr.random(Y.shape))*100:5.1f}%")
print(f"  most popular (movie avg): {precision_at_10(P_item)*100:5.1f}%")
print(f"  collaborative filtering : {precision_at_10(P_raw)*100:5.1f}%")
u0 = 0
cand = np.where(~seen[:, u0])[0]
top5 = cand[np.argsort(-P_raw[cand, u0])[:5]]
print(f"\nTop-5 recommendations for user {u0}: movies {top5.tolist()} with scores {np.round(P_raw[top5, u0], 2).tolist()}")
print("(Scores above 5.0 are fine for RANKING; clip to 5 only when you SHOW a star rating.)")


fig, axes = plt.subplots(1, 3, figsize=(18, 5))
ax = axes[0]
ax.plot(final["hist"]["it"], final["hist"]["cost"], linewidth=2)
ax.set_xlabel("iteration")
ax.set_ylabel("cost J (training ratings)")
ax.set_yscale("log")
ax.set_title("Cost goes down as features are learned", fontweight="bold")
ax.grid(alpha=0.3)
 
ax = axes[1]
for lam in lams:
    ax.plot(Ks, [grid[(K, lam)] for K in Ks], "o-", label=f"lambda = {lam}")
ax.axhline(rmse(P_item, Y, R_va), color="gray", linestyle=":", label="movie-average baseline")
ax.set_xlabel("number of features K")
ax.set_ylabel("validation RMSE")
ax.set_title("Choosing K and lambda", fontweight="bold")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
 
ax = axes[2]
ax.scatter(Y[R_te], P[R_te], s=8, alpha=0.4)
ax.plot([0.5, 5], [0.5, 5], "r--")
ax.set_xlabel("actual rating (test)")
ax.set_ylabel("predicted rating")
ax.set_title(f"Test ratings (RMSE {rmse(P, Y, R_te):.2f})", fontweight="bold")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("c3_collaborative_filtering.png", dpi=150, bbox_inches="tight")
print("\nSaved: c3_collaborative_filtering.png")
plt.show()
 