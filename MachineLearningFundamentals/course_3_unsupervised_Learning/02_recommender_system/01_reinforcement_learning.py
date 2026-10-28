import numpy as np
 
np.random.seed(42)
 
rewards_sequence = [0, 0, 0, 0, 100]   # reward only at the last step
gamma = 0.9

G = 0.0
for t, r in enumerate(rewards_sequence):
    G += (gamma ** t) * r
print(f"Rewards : {rewards_sequence}")
print(f"γ = {gamma}")
print(f"Return G = {G:.2f}  (discounted)")
 
# Show how G changes with γ
print("\nHow γ affects the value of a future reward:")
for g in [0.0, 0.5, 0.9, 0.99, 1.0]:
    G_g = sum((g**t) * r for t, r in enumerate(rewards_sequence))
    print(f"  γ={g:.2f} → G = {G_g:.2f}")

print("LEVEL 2: Mars Rover — true Q values via Bellman")

n_states = 6
rewards  = np.array([100, 0, 0, 0, 0, 40])  # R(s), 0-indexed
gamma    = 0.5

Q = np.zeros((n_states, 2))    # Q[s, a]
 
def next_state(s, a):
    """Deterministic transition."""
    if s == 0: return 0     # left terminal stays
    if s == 5: return 5     # right terminal stays
    return s - 1 if a == 0 else s + 1
 
for _ in range(1000):          # iterate until convergence
    Q_new = np.zeros_like(Q)
    for s in range(n_states):
        for a in range(2):
            s_next = next_state(s, a)
            Q_new[s, a] = rewards[s] + gamma * Q[s_next].max()
    Q = Q_new

print("True Q* table (rows=states 1..6, cols=left/right):")
print("State  Q(s,left)  Q(s,right)  Best action")
for s in range(n_states):
    best = "left" if Q[s, 0] > Q[s, 1] else "right"
    print(f"  {s+1}    {Q[s,0]:8.2f}   {Q[s,1]:8.2f}     {best}")
 
true_Q_state_values = Q.max(axis=1)
print(f"\nV*(s) = max_a Q*(s,a): {true_Q_state_values.round(2)}")
print("Expected: [100. 50. 25. 12.5 20. 40]")
 
# Verify Bellman residual
bellman_residual = 0.0
for s in range(n_states):
    for a in range(2):
        s_next = next_state(s, a)
        target = rewards[s] + gamma * Q[s_next].max()
        bellman_residual += abs(Q[s, a] - target)
print(f"\nBellman residual: {bellman_residual:.6f}  (should be ~0)")
 
print("LEVEL 3: Q-learning algorithm from scratch")
print("=" * 60)
 
def q_learning(n_episodes=5000, alpha=0.05, gamma=0.5,
               epsilon=0.2, decay=0.999, p_miss=0.0):
    """
    Q-learning on the Mars Rover environment.
 
    p_miss : probability that the action has the opposite effect
             (stochastic environment when > 0)
    """
    Q_learn = np.zeros((n_states, 2))
    eps     = epsilon
    rewards_per_ep = []
 
    for ep in range(n_episodes):
        s           = np.random.randint(1, 5)   # start in middle states 2-5
        ep_reward   = 0.0
        done        = False
 
        while not done:
            # ε-greedy action selection
            if np.random.rand() < eps:
                a = np.random.randint(2)        # explore: random
            else:
                a = Q_learn[s].argmax()         # exploit: greedy
 
            # Stochastic step: sometimes action is reversed
            actual_a = 1 - a if np.random.rand() < p_miss else a
            s_next   = next_state(s, actual_a)
            r        = rewards[s]
 
            # Q-learning update
            target       = r + gamma * Q_learn[s_next].max()
            Q_learn[s, a] += alpha * (target - Q_learn[s, a])
 
            ep_reward += r
            s          = s_next
 
            # Terminal: left (0) or right (5)
            if s in [0, 5]:
                Q_learn[s, actual_a] = rewards[s]   # terminal update
                ep_reward += rewards[s]
                done = True
 
        rewards_per_ep.append(ep_reward)
        eps = max(0.01, eps * decay) 


 return Q_learn, rewards_per_ep
 
Q_learned, reward_history = q_learning(n_episodes=5000)
 
print("Learned Q table:")
print("State  Q(s,left)  Q(s,right)  Best action")
for s in range(n_states):
    best = "left" if Q_learned[s, 0] > Q_learned[s, 1] else "right"
    print(f"  {s+1}    {Q_learned[s,0]:8.2f}   {Q_learned[s,1]:8.2f}     {best}")
 
error = np.abs(Q_learned - Q).mean()
print(f"\nMean Q-table error vs true Q*: {error:.4f}  (close to 0 = well learned)")
 
# Policy comparison
learned_policy = Q_learned.argmax(axis=1)
true_policy    = Q.argmax(axis=1)
print(f"Same policy as optimal: {np.all(learned_policy == true_policy)}")
 
import matplotlib.pyplot as plt
# Smooth reward curve
window = 200
smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
plt.figure(figsize=(7, 3))
plt.plot(smoothed)
plt.xlabel("Episode")
plt.ylabel("Return (smoothed)")
plt.title("Q-learning: return over training episodes")
plt.tight_layout()
plt.savefig("learn_c3_04a_qlearning.png")
plt.close()
print("Saved: learn_c3_04a_qlearning.png\n")
 
 
# =============================================================================
# LEVEL 4 — EXPLORATION VS EXPLOITATION
# =============================================================================
# What happens when ε is too low or too high?
 
print("=" * 60)
print("LEVEL 4: Effect of ε (exploration vs exploitation)")
print("=" * 60)
 
def evaluate_policy(Q_table, p_miss=0.0, n_eval=500):
    """Average return of the greedy policy from Q_table."""
    total = 0.0
    for _ in range(n_eval):
        s = 2  # always start from state 3 (index 2)
        done = False
        while not done:
            a      = Q_table[s].argmax()
            actual = 1 - a if np.random.rand() < p_miss else a
            s_next = next_state(s, actual)
            if s_next in [0, 5]:
                total += rewards[s_next]
                done = True
            s = s_next
    return total / n_eval