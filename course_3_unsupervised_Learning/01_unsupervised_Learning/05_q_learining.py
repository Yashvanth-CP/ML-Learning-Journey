# REINFORCEMENT LEARNING + Q-LEARNING (tabular, from scratch)



import numpy as np
 
np.set_printoptions(precision=2, suppress=True, linewidth=120)
LEFT, RIGHT = 0, 1
 
 
class Corridor:
    """States 0..n-1 in a row. Action 0 = left, 1 = right. State 0 and n-1 are terminal.
    p_miss = chance that the agent moves in the OPPOSITE direction (a 'misstep')."""
 
    def __init__(self, rewards, p_miss=0.0):
        self.R = np.array(rewards, dtype=float)
        self.n = len(rewards)
        self.p_miss = p_miss
        self.terminal = np.zeros(self.n, dtype=bool)
        self.terminal[[0, self.n - 1]] = True
 
    def step(self, s, a, rng):
        d = -1 if a == LEFT else +1
        if self.p_miss > 0 and rng.random() < self.p_miss:
            d = -d
        return s + d
 
 
def discounted_return(rewards, gamma):
    """R1 + gamma*R2 + gamma^2*R3 + ...   (first reward is not discounted)"""
    return sum(r * gamma ** t for t, r in enumerate(rewards))
 
 
def optimal_Q(env, gamma, sweeps=500):
    """The TRUE best Q-values, by repeating the Bellman equation until it stops changing.
    (This needs to KNOW the world. Q-learning later does NOT - it only uses experience.)"""
    Q = np.zeros((env.n, 2))
    for _ in range(sweeps):
        V = Q.max(axis=1)
        Qn = np.zeros_like(Q)
        for s in range(env.n):
            if env.terminal[s]:
                Qn[s, :] = env.R[s]                      # terminal: Q = just the reward
                continue
            for a in (LEFT, RIGHT):
                d = -1 if a == LEFT else 1
                Qn[s, a] = env.R[s] + gamma * ((1 - env.p_miss) * V[s + d] + env.p_miss * V[s - d])
        Q = Qn
    return Q
 
 
def policy_string(env, Q):
    """'<' means go left, '>' means go right, '*' = terminal state (ties print '='). 1-based states."""
    out = []
    for s in range(env.n):
        if env.terminal[s]:
            out.append("*")
        elif Q[s, LEFT] > Q[s, RIGHT] + 1e-9:
            out.append("<")
        elif Q[s, RIGHT] > Q[s, LEFT] + 1e-9:
            out.append(">")
        else:
            out.append("=")
    return " ".join(out)


rint("=" * 74)
print("PART 1: THE MARS ROVER  (6 states in a row, rewards 100 ... 40)")
print("=" * 74)
rover = Corridor([100, 0, 0, 0, 0, 40])
print("""
   state :   1     2     3     4     5     6
   reward: 100     0     0     0     0    40
           GOAL                         small goal      (states 1 and 6 end the episode)
""")
print("Start at state 4. Two simple plans (gamma = 0.5):")
ret_left = discounted_return([0, 0, 0, 100], 0.5)     # rewards seen at states 4,3,2,1
ret_right = discounted_return([0, 0, 40], 0.5)        # rewards seen at states 4,5,6
print(f"  always LEFT : rewards 0,0,0,100 -> return = 0 + 0.5*0 + 0.25*0 + 0.125*100 = {ret_left}")
print(f"  always RIGHT: rewards 0,0,40    -> return = 0 + 0.5*0 + 0.25*40            = {ret_right}")
print("  The goal on the right is CLOSER (2 steps vs 3), yet LEFT wins: 100 is much bigger than 40.")
 
print("\nSame world, different gamma: which way does the rover go from each state? (<  left, > right)")
print("  gamma | policy for states 1..6 | state 4 return LEFT / RIGHT")
for g in (0.1, 0.3, 0.5, 0.9, 0.99):
    Qg = optimal_Q(rover, g)
    print(f"  {g:5.2f} |  {policy_string(rover, Qg)}  |  {Qg[3, LEFT]:7.2f} / {Qg[3, RIGHT]:7.2f}")
print("  Small gamma = impatient -> grabs the nearby 40. Big gamma = patient -> walks to the 100.")
print("  (From state 4: left = 100*g^3, right = 40*g^2, so they tie at gamma = 0.4.)")