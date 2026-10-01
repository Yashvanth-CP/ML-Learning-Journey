"""
06. COMPLETE TRAINING LOOP
"""

import numpy as np
import matplotlib.pyplot as plt

print("COMPLETE NEURAL NETWORK: Full Training Pipeline")

# NEURAL NETWORK CLASS

class NeuralNetwork:
    # complete neural network from scratch 

    def __init__(self, input_size=2, hidden_size=4, output_size=1, learning_rate=0.1):
        self.learning_rate = learning_rate

        # layer 1 input -> hidden
        self.W1 = np.random.rand(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1,hidden_size))

        # layer 2 hidden -> output

        self.W2 = np.random.rand(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        print("=" * 70)
        print("Neural network ")
        print(f" input = {input_size} features")
        print(f"hidden layer : {hidden_size} neuronns (ReLU)")
        print(f" output : {output_size} neuron (sigmoid)")
        print(f"\nParameters:")
        print(f"  W1 shape: {self.W1.shape}")
        print(f"  b1 shape: {self.b1.shape}")
        print(f"  W2 shape: {self.W2.shape}")
        print(f"  b2 shape: {self.b2.shape}")

    def forward(self, X):
        # X = Input data (m, input_size) 
        # A2 = Prediction (m, output_size)

        # layer 1 : linear and Activation 
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)
        
        # layer 2 linear : sigmoid 
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = 1 / (1 + np.exp(-np.clip(self.Z2, -500, 500)))

        # save for backward
        self.X = X
        return self.A2

    def backward(self, y):
        # y = Target values (m , output_size)

        m = self.X.shape[0]

        # layer 2 backward 
        dZ2 = self.A2 - y
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # layer 1 
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0) 
        dW1 = (1/m) * np.dot(self.X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # updated weight 
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def compute_loss(self, y_pred, y_true):
        # binary cross entropy 

        epsilon = 1e-8
        loss = -np.mean(y_true * np.log(y_pred + epsilon) + (1- y_true) * np.log(1-y_pred+ epsilon))
        return loss


    def compute_accuracy(self, y_pred, y_true):
        # binary cross accuracy 

        y_pred_binary = (y_pred > 0.5).astype(int)

        return np.mean(y_pred_binary == y_true)

    def train(self, X_train, y_train, X_test=None, y_test=None, epochs=100, batch_size=32):
        # Compute training loop
        """
         X_train: Training input
            y_train: Training target
            X_test: Test input (optional)
            y_test: Test target (optional)
            epochs: Number of epochs
            batch_size: Batch size for mini-batch gradient descent
        """
        train_losses =[]
        train_accs =[]
        test_losses =[]
        test_accs = []

        print(f"\n{'='*70}")
        print("STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Training samples: {len(X_train)}")

        for epoch in range(epochs):
            # shuffle data 
            indices = np.random.permutation(len(X_train)) 
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            # mini -batch gradient descent 
            for i in range(0, len(X_train), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # forward 
                self.forward(X_batch)

                # backward 
                self.backward(y_batch)

                # Evaluate 
                # Training metrics

                y_pred_train = self.forward(X_train)
                train_loss = self.compute_loss(y_pred_train, y_train)
                train_acc = self.compute_accuracy(y_pred_train, y_train)
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # Test metrics (if provided)

                if X_test is not None:
                    y_pred_test = self.forward(X_test)
                    test_loss = self.compute_loss(y_pred_test, y_test)
                    test_acc =self.compute_accuracy(y_pred_test, y_test)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)

                if epoch % 20 == 0:
                    if X_test is not None:
                        print(f"Epoch {epoch:4d}: Train Loss={train_loss:.4f} ")
                        print( f"Train Acc={train_acc:.4f} ")
                        print( f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.4f}")

                    else : 
                        print(f" Epoch {epoch:4d}: Train Loss = {train_loss:.4},"
                              f" train ACC = {train_acc:4f}")

    
        print(f"\n{'='*70}")
        print("TRAINING COMPLETE")
        print(f"{'='*70}")
        print(f"Final Train Loss: {train_losses[-1]:.4f}")
        print(f"Final Train Accuracy: {train_accs[-1]:.4f}")
        if X_test is not None:
            print(f"Final Test Loss: {test_losses[-1]:.4f}")
            print(f"Final Test Accuracy: {test_accs[-1]:.4f}")
        
        return train_losses, train_accs, test_losses, test_accs

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)


# ============ CREATE DATASET ============
 
print("\n1. CREATING DATASET")
print("-"*70)
 
# Classification problem
np.random.seed(42)
 
# Class 0 (circular region in center)
theta = np.random.uniform(0, 2*np.pi, 100)
r = np.random.uniform(0, 1, 100)
X0 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y0 = np.zeros((100, 1))
 
# Class 1 (annular region outside)
theta = np.random.uniform(0, 2*np.pi, 100)
r = np.random.uniform(1.5, 2.5, 100)
X1 = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
y1 = np.ones((100, 1))
 
# Combine
X = np.vstack([X0, X1])
y = np.vstack([y0, y1])
 
# Split
split = int(0.8 * len(X))
X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]
 
print(f"Dataset created:")
print(f"  Total samples: {len(X)}")
print(f"  Training: {len(X_train)} samples")
print(f"  Test: {len(X_test)} samples")
print(f"  Class 0 (center): {len(y0)} samples")
print(f"  Class 1 (outside): {len(y1)} samples")

# Create and train network 

network = NeuralNetwork(input_size=2, hidden_size=8, output_size=1, learning_rate=0.1)

train_losses, train_accs, test_losses, test_accs = network.train(
    X_train, y_train, X_test, y_test,
    epochs=200,
    batch_size=16
)

# ============ VISUALIZATION ============
 
print("\n4. VISUALIZING RESULTS")
print("-"*70)
 
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
 
# Plot 1: Decision Boundary
ax1 = fig.add_subplot(gs[0, :2])
 
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
 
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
 
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = network.predict(grid_points)
Z = Z.reshape(xx.shape)
 
ax1.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.3)
ax1.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax1.scatter(X_train[y_train[:, 0] == 0, 0], 
           X_train[y_train[:, 0] == 0, 1],
           color='blue', s=50, label='Class 0 (train)', alpha=0.7)
ax1.scatter(X_train[y_train[:, 0] == 1, 0], 
           X_train[y_train[:, 0] == 1, 1],
           color='red', s=50, label='Class 1 (train)', alpha=0.7)
ax1.scatter(X_test[y_test[:, 0] == 0, 0], 
           X_test[y_test[:, 0] == 0, 1],
           color='blue', s=100, marker='x', label='Class 0 (test)', linewidths=2)
ax1.scatter(X_test[y_test[:, 0] == 1, 0], 
           X_test[y_test[:, 0] == 1, 1],
           color='red', s=100, marker='x', label='Class 1 (test)', linewidths=2)
ax1.set_title('Decision Boundary', fontsize=12, fontweight='bold')
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
 
# Plot 2: Metrics summary
ax2 = fig.add_subplot(gs[0, 2])
summary_text = f"""Training Summary
 
Final Metrics:
━━━━━━━━━━━━━━━━
Train Loss: {train_losses[-1]:.4f}
Train Acc:  {train_accs[-1]:.1%}
 
Test Loss:  {test_losses[-1]:.4f}
Test Acc:   {test_accs[-1]:.1%}
 
Gap (overfitting):
△Loss:  {test_losses[-1] - train_losses[-1]:+.4f}
△Acc:   {test_accs[-1] - train_accs[-1]:+.1%}
"""
ax2.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax2.transAxes)
ax2.axis('off')
 
# Plot 3: Loss curves
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(train_losses, 'b-', linewidth=2, label='Train')
ax3.plot(test_losses, 'r-', linewidth=2, label='Test')
ax3.set_title('Loss Over Time', fontsize=12, fontweight='bold')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)
 
# Plot 4: Accuracy curves
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(train_accs, 'b-', linewidth=2, label='Train')
ax4.plot(test_accs, 'r-', linewidth=2, label='Test')
ax4.set_title('Accuracy Over Time', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Accuracy')
ax4.set_ylim([0, 1.05])
ax4.legend()
ax4.grid(True, alpha=0.3)
 
# Plot 5: Overfitting gap
ax5 = fig.add_subplot(gs[1, 2])
loss_gap = [test_losses[i] - train_losses[i] for i in range(len(train_losses))]
ax5.plot(loss_gap, 'purple', linewidth=2)
ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax5.set_title('Overfitting Gap (Test-Train)', fontsize=12, fontweight='bold')
ax5.set_xlabel('Epoch')
ax5.set_ylabel('Loss Difference')
ax5.grid(True, alpha=0.3)
 
# Plot 6: Hidden layer visualization
ax6 = fig.add_subplot(gs[2, 0])
hidden_output = np.maximum(0, np.dot(X, network.W1) + network.b1)
scatter = ax6.scatter(hidden_output[:, 0], hidden_output[:, 1], 
                     c=y.ravel(), cmap='RdYlBu', s=50, alpha=0.7)
ax6.set_title('Hidden Layer Features (First 2)', fontsize=12, fontweight='bold')
ax6.set_xlabel('Neuron 1')
ax6.set_ylabel('Neuron 2')
ax6.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax6, label='Class')
 
# Plot 7: Predictions distribution
ax7 = fig.add_subplot(gs[2, 1])
y_pred_all = network.predict(X)
ax7.hist(y_pred_all[y == 0], bins=20, alpha=0.5, label='Class 0 (true)', color='blue')
ax7.hist(y_pred_all[y == 1], bins=20, alpha=0.5, label='Class 1 (true)', color='red')
ax7.axvline(x=0.5, color='k', linestyle='--', linewidth=2, label='Decision boundary')
ax7.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
ax7.set_xlabel('Predicted Probability')
ax7.set_ylabel('Count')
ax7.legend()
ax7.grid(True, alpha=0.3)
 
# Plot 8: Confusion matrix
ax8 = fig.add_subplot(gs[2, 2])
y_pred_binary = (network.predict(X_test) > 0.5).astype(int)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_binary)
im = ax8.imshow(cm, cmap='Blues', aspect='auto')
ax8.set_title('Confusion Matrix (Test)', fontsize=12, fontweight='bold')
ax8.set_xlabel('Predicted')
ax8.set_ylabel('Actual')
ax8.set_xticks([0, 1])
ax8.set_yticks([0, 1])
 
# Add values
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax8.text(
            j, i, cm[i, j],
            ha="center",
            va="center",
            color="white" if cm[i, j] > cm.max()/2 else "black",
            fontsize=14,
            fontweight="bold"
        )
 
plt.colorbar(im, ax=ax8)
 
# Save
plt.savefig('complete_training_loop.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: complete_training_loop.png")
plt.show()
 
 
# ============ TEST ON NEW DATA ============
 
print("\n5. TESTING ON NEW DATA")
print("-"*70)
 
# New points
new_points = np.array([
    [0, 0],      # Center
    [0.5, 0.5],  # Near center
    [2, 0],      # Far right (class 1)
    [0, 2],      # Far up (class 1)
    [1, 0],      # Boundary
])
 
print("\nNew predictions:")
new_pred = network.predict(new_points)
 
for point, pred in zip(new_points, new_pred):
    conf = max(pred[0], 1 - pred[0]) * 100
    cls = "Class 1" if pred[0] > 0.5 else "Class 0"
    print(f"  Point {point}: {cls} ({conf:.1f}% confident)")


"""
What We Did:
 
1. ✓ Created neural network from scratch
   - No libraries (just NumPy)
   - 2 layers with ReLU and Sigmoid
 
2. ✓ Generated realistic dataset
   - 2D classification problem
   - Train/test split
 
3. ✓ Trained with full pipeline
   - Forward propagation
   - Backward propagation
   - Gradient descent with batches
   - Loss and accuracy tracking
 
4. ✓ Visualized results
   - Decision boundary
   - Loss curves (train vs test)
   - Hidden layer features
   - Confusion matrix
 
5. ✓ Made predictions on new data
   - Used trained network to classify
 
Key Insights:
 
• Network learned to separate classes
• No overfitting (test acc ≈ train acc)
• ReLU and sigmoid worked well
• Vectorization made it fast
• Backprop automatically found good weights!
 
This is a complete neural network!
Everything you see in TensorFlow/PyTorch,
just implemented from scratch to understand it.
 
NEXT STEPS:
→ Study neural network layers
→ Build demand prediction project
→ Tackle face recognition
 
You now understand neural networks! 🎉
"""