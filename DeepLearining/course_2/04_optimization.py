import numpy as np

# X = np.array([[1000.,1.], [2000.,3.], [3000., 2.], [4000., 5.], [5000., 4.]])
# y = X @ np.array([0.002,1.0])

# mu = X.mean(axis=0)
# sigma = X.std(axis=0)
# X_norm = (X - mu) / sigma

# def train(X, y, alpha, steps):
#     m, n = X.shape
#     w, b = np.zeros(n), 0.0

#     for _ in range(steps):
#         error = X @ w + b - y # forward pass
#         w -= alpha * (X.T @ error) / m
#         b -= alpha * error.mean()

#     return np.mean((X @ w + b - y) ** 2) / 2 # final cost

# print(train(X, y, 0.1, 100))
# print(train(X, y, 1e-7, 100))
# print(train(X_norm, y, 0.1, 100))



# # 2. Vanishing and exploding gradients
# print("Vanishing and exploding gradients")
# def forward(scale):
#     rng = np.random.default_rng(1)
#     A = rng.standard_normal((500, 1000)) # 500 feature, 1000 examples

#     for layer in range(10):
#         W  = rng.standard_normal((500, 500)) * scale
#         A = np.maximum(0, W @ A)
#     return A.std()

# print(forward(0.01))               # about 1e-8   (vanished)
# print(forward(1.0))                # about 1e12   (exploded)
# print(forward(np.sqrt(2 / 500))) 

# print("+"*25 + "gradient checking")

# # difference = ||approx - grad|| / (||approx|| + ||grad||)

# rng = np.random.default_rng(0)
# Xs = rng.standard_normal((5,3))
# ys = rng.standard_normal(5)
# theta = rng.standard_normal(3)

# def cost(th):
#     return np.mean((Xs @ th - ys) ** 2) / 2
# def grad(th) : 
#     return Xs.T @ (Xs @ th - ys) / len(ys)

# def gradient_check(theta, analytic, eps = 1e-7):
#     approx = np.zeros_like(theta)
#     for i in range(len(theta)):
#         plus, minus = theta.copy(), theta.copy()
#         plus[i] += eps
#         minus[i] -= eps
#         approx[i] = (cost(plus) - cost(minus)) / (2 * eps)
#     top = np.linalg.norm(approx - analytic)
#     bottom = np.linalg.norm(approx) + np.linalg.norm(analytic)

#     return top/bottom

# print(gradient_check(theta, grad(theta)))       # about 5e-10: correct
# print(gradient_check(theta, 2 * grad(theta)))   # about 0.33: bug
        
a = np.random.default_rng(0)
b= np.random.default_rng(5)
print(a.standard_normal(10))
print(b.standard_normal(2))

import numpy as np

a = np.random.default_rng(5)
b = np.random.default_rng(5)

first = a.standard_normal(2)
x = a.standard_normal(2)
y = b.standard_normal(2)
z = b.standard_normal(2)
print("first:", first)
print("x    :", x)
print("y    :", y)
print(" z    :", z)