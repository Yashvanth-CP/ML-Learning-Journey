"""
PROJECT 1: DEMAND PREDICTION
Step 4: Train Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("DEMAND PREDICTION: Step 4 - Train Model")
print("="*70)


#  COPY MODEL CLASS FROM PREVIOUS STEP

class RegressionNetwork:
    """Neural network for regression"""
    
    def __init__(self, input_size=6, hidden_sizes=[32, 16, 8], 
                 learning_rate=0.01, dropout_rate=0.3):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_sizes[0]))
        
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2 / hidden_sizes[0])
        self.b2 = np.zeros((1, hidden_sizes[1]))
        
        self.W3 = np.random.randn(hidden_sizes[1], hidden_sizes[2]) * np.sqrt(2 / hidden_sizes[1])
        self.b3 = np.zeros((1, hidden_sizes[2]))
        
        self.W4 = np.random.randn(hidden_sizes[2], 1) * np.sqrt(2 / hidden_sizes[2])
        self.b4 = np.zeros((1, 1))
    
    def forward(self, X, training=True):
        self.X = X
        self.training = training
        
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        if training:
            self.mask1 = np.random.binomial(1, 1 - self.dropout_rate, self.A1.shape)
            self.A1_dropped = (self.A1 * self.mask1) / (1 - self.dropout_rate)
        else:
            self.A1_dropped = self.A1
        
        self.Z2 = np.dot(self.A1_dropped, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2)
        
        if training:
            self.mask2 = np.random.binomial(1, 1 - self.dropout_rate, self.A2.shape)
            self.A2_dropped = (self.A2 * self.mask2) / (1 - self.dropout_rate)
        else:
            self.A2_dropped = self.A2
        
        self.Z3 = np.dot(self.A2_dropped, self.W3) + self.b3
        self.A3 = np.maximum(0, self.Z3)
        
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.output = self.Z4
        
        return self.output
    
    def backward(self, y):
        m = self.X.shape[0]
        
        dZ4 = self.output - y
        dW4 = (1/m) * np.dot(self.A3.T, dZ4)
        db4 = (1/m) * np.sum(dZ4, axis=0, keepdims=True)
        
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * (self.Z3 > 0)
        dW3 = (1/m) * np.dot(self.A2_dropped.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)
        
        dA2 = np.dot(dZ3, self.W3.T)
        if self.training:
            dA2 = dA2 * self.mask2 / (1 - self.dropout_rate)
        dZ2 = dA2 * (self.Z2 > 0)
        dW2 = (1/m) * np.dot(self.A1_dropped.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)
        
        dA1 = np.dot(dZ2, self.W2.T)
        if self.training:
            dA1 = dA1 * self.mask1 / (1 - self.dropout_rate)
        dZ1 = dA1 * (self.Z1 > 0)
        dW1 = (1/m) * np.dot(self.X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)
        
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4
    
    def compute_loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)
    
    def compute_r2(self, y_pred, y_true):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def predict(self, X):
        return self.forward(X, training=False)


print("\n1. LOADING DATA")
print("-"*70)

# Load prepared data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy').reshape(-1, 1)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy').reshape(-1, 1)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")


print("\n2. CREATE AND INITIALIZE MODEL")
print("-"*70)

model = RegressionNetwork(
    input_size=6,
    hidden_sizes=[32, 16, 8],
    learning_rate=0.01,  # Smaller learning rate
    dropout_rate=0.3
)

print("✓ Model initialized")


print("\n3. TRAINING CONFIGURATION")
print("-"*70)

epochs = 300
batch_size = 16

print(f"""
Training settings:
  Epochs: {epochs}
  Batch size: {batch_size}
  Learning rate: 0.001
  Dropout rate: 0.3
  
Optimization: Mini-batch gradient descent
  - Shuffle data each epoch
  - Update weights every batch
  - Monitor train and test loss
""")


print("\n4. TRAINING LOOP")
print("-"*70)

train_losses = []
test_losses = []
train_r2s = []
test_r2s = []

print(f"Training for {epochs} epochs...\n")

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    # Mini-batch gradient descent
    for i in range(0, len(X_train), batch_size):
        X_batch = X_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        # Forward pass
        model.forward(X_batch, training=True)
        
        # Backward pass
        model.backward(y_batch)
    
    # ===== EVALUATE =====
    # Training metrics
    y_pred_train = model.predict(X_train)
    train_loss = model.compute_loss(y_pred_train, y_train)
    train_r2 = model.compute_r2(y_pred_train, y_train)
    train_losses.append(train_loss)
    train_r2s.append(train_r2)
    
    # Test metrics
    y_pred_test = model.predict(X_test)
    test_loss = model.compute_loss(y_pred_test, y_test)
    test_r2 = model.compute_r2(y_pred_test, y_test)
    test_losses.append(test_loss)
    test_r2s.append(test_r2)
    
    # Print progress
    if epoch % 50 == 0:
        print(f"Epoch {epoch:3d}: "
              f"Train Loss={train_loss:.4f}, R²={train_r2:.4f} | "
              f"Test Loss={test_loss:.4f}, R²={test_r2:.4f}")

print(f"\n✓ Training complete!")


print("\n5. TRAINING RESULTS")
print("-"*70)

print(f"""
Final metrics:

Training:
  Loss: {train_losses[-1]:.6f}
  R²:   {train_r2s[-1]:.4f}

Test:
  Loss: {test_losses[-1]:.6f}
  R²:   {test_r2s[-1]:.4f}

Improvement:
  Loss: {train_losses[0] - train_losses[-1]:.6f} (decreased)
  R²:   {train_r2s[-1] - train_r2s[0]:+.4f}

R² Interpretation:
  R² = 1.0  → Perfect predictions
  R² = 0.8  → Very good (explains 80% of variance)
  R² = 0.5  → Decent (explains 50% of variance)
  R² = 0.0  → As good as mean prediction
  R² < 0.0  → Worse than mean!
""")


print("\n6. VISUALIZING TRAINING")
print("-"*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curves
ax = axes[0, 0]
ax.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
ax.plot(test_losses, 'r-', label='Test Loss', linewidth=2)
ax.set_title('Loss Over Epochs', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.legend()
ax.grid(True, alpha=0.3)

# R² curves
ax = axes[0, 1]
ax.plot(train_r2s, 'b-', label='Train R²', linewidth=2)
ax.plot(test_r2s, 'r-', label='Test R²', linewidth=2)
ax.set_title('R² Score Over Epochs', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('R² Score')
ax.set_ylim([0, 1.05])
ax.legend()
ax.grid(True, alpha=0.3)

# Predictions vs Actual (Train)
ax = axes[1, 0]
ax.scatter(y_train, y_pred_train, alpha=0.5, s=30, color='blue')
ax.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
        'r--', linewidth=2, label='Perfect')
ax.set_title('Train: Predictions vs Actual', fontsize=12, fontweight='bold')
ax.set_xlabel('Actual Revenue')
ax.set_ylabel('Predicted Revenue')
ax.legend()
ax.grid(True, alpha=0.3)

# Predictions vs Actual (Test)
ax = axes[1, 1]
ax.scatter(y_test, y_pred_test, alpha=0.5, s=30, color='red')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
        'k--', linewidth=2, label='Perfect')
ax.set_title('Test: Predictions vs Actual', fontsize=12, fontweight='bold')
ax.set_xlabel('Actual Revenue')
ax.set_ylabel('Predicted Revenue')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: training_results.png")
plt.show()


print("\n7. SAVE TRAINED MODEL")
print("-"*70)

# Save model weights
np.save('model_W1.npy', model.W1)
np.save('model_W2.npy', model.W2)
np.save('model_W3.npy', model.W3)
np.save('model_W4.npy', model.W4)
np.save('model_b1.npy', model.b1)
np.save('model_b2.npy', model.b2)
np.save('model_b3.npy', model.b3)
np.save('model_b4.npy', model.b4)

print("✓ Saved: model weights (model_W*.npy, model_b*.npy)")

# Save training history
np.save('train_losses.npy', np.array(train_losses))
np.save('test_losses.npy', np.array(test_losses))
np.save('train_r2s.npy', np.array(train_r2s))
np.save('test_r2s.npy', np.array(test_r2s))

print("✓ Saved: training history (losses and R² scores)")



print("✓ Model trained and saved!")
