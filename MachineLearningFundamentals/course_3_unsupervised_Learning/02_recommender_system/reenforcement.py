import numpy as np

# States: 0=poor_soil, 1=normal_soil, 2=fertile_soil
# Rewards for being in each state
R = np.array([5, 20, 50])

# Next state after following optimal policy
# (we assume policy already determined)
next_states = np.array([1, 2, 2])

# Discount factor
gamma = 0.9

# Initialize V(s)
V = np.zeros(3)

# Bellman equation: V(s) = R(s) + γ · V(s')
# We need to solve this iteratively

print("Calculating State Values using Bellman Equation:\n")

# Iteration 1
V[2] = R[2] + gamma * V[2]  # Fertile stays fertile
print(f"Iteration 1: V(fertile) = {R[2]} + {gamma} × {V[2]} = {V[2]}")

V[1] = R[1] + gamma * V[2]  # Normal → Fertile
print(f"Iteration 1: V(normal) = {R[1]} + {gamma} × {V[2]} = {V[1]}")

V[0] = R[0] + gamma * V[1]  # Poor → Normal
print(f"Iteration 1: V(poor) = {R[0]} + {gamma} × {V[1]} = {V[0]}")

print("\n" + "="*50 + "\n")

# Iteration 2 (more accurate)
V[2] = R[2] + gamma * V[2]
V[1] = R[1] + gamma * V[2]
V[0] = R[0] + gamma * V[1]

print(f"Iteration 2:")
print(f"V(poor_soil) = {V[0]:.2f}")
print(f"V(normal_soil) = {V[1]:.2f}")
print(f"V(fertile_soil) = {V[2]:.2f}")

# Continue iterating until convergence
for iteration in range(3, 20):
    V_old = V.copy()
    V[2] = R[2] + gamma * V[2]
    V[1] = R[1] + gamma * V[2]
    V[0] = R[0] + gamma * V[1]
    
    if np.allclose(V, V_old):
        print(f"\nConverged at iteration {iteration}")
        break

print(f"\nFinal State Values:")
print(f"V(poor_soil) = {V[0]:.2f}")
print(f"V(normal_soil) = {V[1]:.2f}")
print(f"V(fertile_soil) = {V[2]:.2f}")

print(f"\nInterpretation:")
print(f"Fertile soil is worth {V[2] - V[1]:.1f} more than normal soil")
print(f"This is because it gives better immediate + future rewards")