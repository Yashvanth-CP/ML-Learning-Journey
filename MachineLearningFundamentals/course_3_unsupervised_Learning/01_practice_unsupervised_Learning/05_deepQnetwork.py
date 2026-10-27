import numpy as np
 
np.random.seed(42)



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
 
    def store(self, s, a, r, s2, done):
        """Add one transition."""
        self.S[self.pos]  = s
        self.A[self.pos]  = a
        self.R[self.pos]  = r
        self.S2[self.pos] = s2
        self.Done[self.pos] = done
        self.pos  = (self.pos + 1) % self.cap   # wrap around
        self.size = min(self.size + 1, self.cap)
 
    def sample(self, batch_size):
        """Return a random mini-batch of transitions."""
        idx = np.random.choice(self.size, size=batch_size, replace=False)
        return (self.S[idx], self.A[idx], self.R[idx],
                self.S2[idx], self.Done[idx])
 
# Show how the buffer works
buf = ReplayBuffer(capacity=5, state_dim=2)
for i in range(7):               # add 7 transitions to a buffer of size 5
    s  = np.array([i, i])
    s2 = np.array([i+1, i+1])
    buf.store(s, 0, float(i), s2, False)


# Show soft update
main_net   = QNetwork(4, 8, 2, seed=1)
target_net = QNetwork(4, 8, 2, seed=2)
tau = 0.005
 
# Before update
diff_before = abs(main_net.W1 - target_net.W1).mean()
 
# Soft update
for (mp, tp) in zip(main_net.get_params(), target_net.get_params()):
    tp[:] = tau * mp + (1 - tau) * tp   # in-place update
 
diff_after = abs(main_net.W1 - target_net.W1).mean()
print(f"Mean |W1_main - W1_target| before soft update: {diff_before:.4f}")
print(f"Mean |W1_main - W1_target| after  soft update: {diff_after:.4f}")
print(f"Target moved slightly toward main (τ={tau})\n")


n_states  = 6
n_actions = 2
rewards   = np.array([100, 0, 0, 0, 0, 40], dtype=float)
gamma     = 0.5
tau_soft  = 0.005
lr        = 5e-4
batch_sz  = 64
 
def state_vec(s):
    """Convert state index to one-hot vector."""
    v = np.zeros(n_states)
    v[s] = 1.0
    return v
 
def next_state(s, a):
    if s == 0: return 0
    if s == 5: return 5
    return s - 1 if a == 0 else s + 1
 
# Adam optimiser (simple implementation)
class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0
 
    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            if i not in self.m:
                self.m[i] = np.zeros_like(p)
                self.v[i] = np.zeros_like(p)
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*g**2
            m_hat = self.m[i] / (1 - self.b1**self.t)
            v_hat = self.v[i] / (1 - self.b2**self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
 
main_q   = QNetwork(n_states, 32, n_actions, seed=0)
target_q = QNetwork(n_states, 32, n_actions, seed=0)
target_q.copy_params_from(main_q)
 
buffer   = ReplayBuffer(capacity=10000, state_dim=n_states)
optim    = Adam(lr=lr)
eps_val  = 1.0
 
# Collect initial experience
for _ in range(200):
    s = np.random.randint(1, 5)
    a = np.random.randint(2)
    sn = next_state(s, a)
    r  = rewards[s]
    done = sn in [0, 5]
    if done:
        r += rewards[sn]
    buffer.store(state_vec(s), a, r, state_vec(sn), done)
 
# Training
losses = []
for step in range(8000):
    # Sample mini-batch
    S, A, R, S2, Done = buffer.sample(min(batch_sz, buffer.size))
 
    # Compute targets using TARGET network
    Q_next = target_q.forward(S2)              # (batch, n_actions)
    max_Q_next = Q_next.max(axis=1)            # (batch,)
    targets_full = R + gamma * max_Q_next * (~Done)  # (batch,)
 
    # Forward pass main network
    Q_pred = main_q.forward(S)                 # (batch, n_actions)
 
    # Only update Q for the chosen action
    Q_target_mat = Q_pred.copy()
    Q_target_mat[np.arange(len(A)), A] = targets_full
 
    # Loss and backward
    loss = 0.5 * ((Q_pred - Q_target_mat)**2).mean()
    dQ   = (Q_pred - Q_target_mat) / len(A)
    grads = main_q.backward(dQ)
    optim.step(main_q.get_params(), grads)
    losses.append(loss)
 
    # Soft update target network
    for mp, tp in zip(main_q.get_params(), target_q.get_params()):
        tp[:] = tau_soft * mp + (1 - tau_soft) * tp
 
    # Collect new experience with ε-greedy
    s  = np.random.randint(1, 5)
    sv = state_vec(s)
    a  = np.random.randint(2) if np.random.rand() < eps_val else main_q.forward(sv).argmax()
    sn = next_state(s, a)
    r  = rewards[s]
    done = sn in [0, 5]
    if done: r += rewards[sn]
    buffer.store(sv, a, r, state_vec(sn), done)
    eps_val = max(0.05, eps_val * 0.9997)
 
# Evaluate learned Q table
print("DQN learned Q values vs true Q*:")
print("State  Q(s,left)  Q(s,right)  Best action  True best")
true_best_names = ['left','left','left','right','right','right']
for s in range(n_states):
    sv    = state_vec(s)
    q_dqn = main_q.forward(sv)
    best  = "left" if q_dqn[0] > q_dqn[1] else "right"
    print(f"  {s+1}    {q_dqn[0]:8.2f}   {q_dqn[1]:8.2f}     {best:<6}       {true_best_names[s]}")
 
import matplotlib.pyplot as plt
plt.figure(figsize=(7, 3))
window = 100
smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
plt.plot(smoothed)
plt.xlabel("Training step")
plt.ylabel("TD Loss (smoothed)")
plt.title("DQN training loss")
plt.tight_layout()
plt.savefig("learn_c3_05a_dqn_loss.png")
plt.show()
print("Saved: learn_c3_05a_dqn_loss.png\n")
 