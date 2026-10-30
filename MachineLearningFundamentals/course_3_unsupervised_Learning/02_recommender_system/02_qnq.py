# =============================================================================
# COURSE 3 — CONCEPT FILE 05: DEEP Q-NETWORK (DQN)
# Beginner → Intermediate
# =============================================================================
# WHAT YOU WILL LEARN:
#   Level 1 (Beginner)   : Why we need DQN (limits of tabular Q-learning)
#   Level 2 (Basic)      : Neural network Q-function, forward pass
#   Level 3 (Intermediate): Replay buffer, target network, soft update
#   Level 4 (Apply)      : Training loop, ablation — what happens if you
#                           remove each trick
# =============================================================================
 
import numpy as np
 
np.random.seed(42)
 
 
# =============================================================================
# LEVEL 1 — WHY DQN?
# =============================================================================
# Tabular Q-learning stores Q[s, a] in a table.
# That works when states are small (e.g. 6 states for Mars Rover).
#
# Real problems have HUGE or CONTINUOUS state spaces:
#   - CartPole: state = [position, velocity, angle, angular_velocity] ∈ ℝ⁴
#   - Atari games: state = pixel image (84×84×4 frames)
#   - Robot: state = joint angles + velocities
#
# You can't build a table for ℝ⁴ (infinitely many states).
# Solution: approximate Q(s, a) with a NEURAL NETWORK.
#   Input: state vector s
#   Output: Q value for EVERY action  [Q(s,a₀), Q(s,a₁), ...]
#
# The network replaces the table. Training it is harder — three tricks fix it.
 
print("=" * 60)
print("LEVEL 1: Why neural networks for Q-learning?")
print("=" * 60)
print("""
Tabular Q:    Q[s_discrete, a]   → can't handle continuous states
DQN:          Q_theta(s, a)      → neural net maps any state to Q values
 
Three problems that arise and their fixes:
  PROBLEM 1: Consecutive (s,a,r,s') samples are correlated
             → biases gradient updates (like training on sorted data)
  FIX:       Replay buffer — store transitions, sample random mini-batches
 
  PROBLEM 2: Target (r + γ*max Q(s')) changes every step → unstable
  FIX:       Target network — a FROZEN copy of Q, updated slowly
 
  PROBLEM 3: Soft vs hard update for target network
  FIX:       Soft update: θ_target ← τ*θ + (1-τ)*θ_target (τ small, e.g. 0.005)
""")
 
 
# =============================================================================
# LEVEL 2 — NEURAL NETWORK Q-FUNCTION
# =============================================================================
 
print("=" * 60)
print("LEVEL 2: Neural network for Q — forward pass")
print("=" * 60)
 
class QNetwork:
    """
    A simple 2-layer neural network for Q-value approximation.
 
    Architecture: input_dim → hidden_dim → n_actions
    Activations:  ReLU after hidden layer, LINEAR output (no activation)
 
    Why linear output? Q values can be any real number — we don't
    want to squash them with sigmoid or tanh.
    """
    def __init__(self, input_dim, hidden_dim, n_actions, seed=0):
        rng = np.random.default_rng(seed)
        # He initialisation: scale by sqrt(2/fan_in) for ReLU layers
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / hidden_dim)
        self.W1 = rng.standard_normal((input_dim,  hidden_dim)) * scale1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.standard_normal((hidden_dim, n_actions))  * scale2
        self.b2 = np.zeros(n_actions)
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_actions  = n_actions
 
    def forward(self, s):
        """
        s : (batch, input_dim) OR (input_dim,) for a single state
        Returns Q values: (batch, n_actions) OR (n_actions,)
        """
        single = s.ndim == 1
        if single:
            s = s.reshape(1, -1)
        self.h1_pre = s @ self.W1 + self.b1   # (batch, hidden)
        self.h1     = np.maximum(0, self.h1_pre)  # ReLU
        self.q_out  = self.h1 @ self.W2 + self.b2 # (batch, n_actions)
        self.s      = s
        return self.q_out[0] if single else self.q_out
 
    def backward(self, dQ):
        """
        Backprop through the network.
        dQ : gradient of loss w.r.t. Q outputs  (batch, n_actions)
        Returns gradients w.r.t. all parameters.
        """
        batch = dQ.shape[0]
        # Output layer
        dW2 = self.h1.T @ dQ / batch           # (hidden, n_actions)
        db2 = dQ.mean(axis=0)                  # (n_actions,)
        dh1 = dQ @ self.W2.T                   # (batch, hidden)
        # ReLU gradient: pass through where h1_pre > 0, zero otherwise
        dh1_pre = dh1 * (self.h1_pre > 0)
        # Hidden layer
        dW1 = self.s.T @ dh1_pre / batch       # (input, hidden)
        db1 = dh1_pre.mean(axis=0)             # (hidden,)
        return dW1, db1, dW2, db2
 
    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]
 
    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = [p.copy() for p in params]
 
    def copy_params_from(self, other):
        self.set_params(other.get_params())
 
 
# Gradient check
print("Gradient check:")
net  = QNetwork(input_dim=4, hidden_dim=8, n_actions=2, seed=7)
s_   = np.random.randn(3, 4)       # batch of 3 states
q_   = net.forward(s_)
# Suppose we have a simple MSE loss: L = mean((q - target)²)
target = np.zeros_like(q_)
loss   = 0.5 * ((q_ - target)**2).mean()
dQ_    = (q_ - target) / q_.shape[0]
dW1, db1, dW2, db2 = net.backward(dQ_)
 
eps = 1e-5
def loss_fn(net):
    q = net.forward(s_)
    return 0.5 * ((q - target)**2).mean()
 
# Numerical grad for W1[0,0]
net.W1[0,0] += eps; l_plus  = loss_fn(net)
net.W1[0,0] -= 2*eps; l_minus = loss_fn(net)
net.W1[0,0] += eps
num_grad = (l_plus - l_minus) / (2*eps)
print(f"  dW1[0,0] analytical: {dW1[0,0]:.6f}")
print(f"  dW1[0,0] numerical : {num_grad:.6f}")
print(f"  Difference         : {abs(dW1[0,0]-num_grad):.2e}  (should be < 1e-5)\n")
 
 
# =============================================================================
# LEVEL 3 — REPLAY BUFFER & TARGET NETWORK
# =============================================================================
 
print("=" * 60)
print("LEVEL 3a: Replay buffer")
print("=" * 60)
 
class ReplayBuffer:
    """
    Circular buffer storing (s, a, r, s', done) transitions.
 
    Why circular? Old memories are overwritten as new ones arrive.
    Random sampling breaks temporal correlation between training samples.
    """
    def __init__(self, capacity, state_dim):
        self.cap   = capacity
        self.pos   = 0          # next write position
        self.size  = 0          # current number of stored transitions
        self.S     = np.zeros((capacity, state_dim))
        self.A     = np.zeros(capacity, dtype=int)
        self.R     = np.zeros(capacity)
        self.S2    = np.zeros((capacity, state_dim))
        self.Done  = np.zeros(capacity, dtype=bool)