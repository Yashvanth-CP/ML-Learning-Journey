"""
03. TWO-LAYER NEURAL NETWORK
"""

import numpy as np
import matplotlib.pylab as plt


print("_"*70)
print("TWO-LAYER NEURAL NETWORK: Stacking Neurons into Layers")
print("_"*70)
"""
Single Neuron:
  - Can learn linear & simple non-linear patterns
  - Limited power
 
Two Layers:
  - Layer 1: Learns basic features (edges, patterns)
  - Layer 2: Combines features to make decisions
  - Much more powerful!
 
Example:
  Single neuron on image: "Is there a line?"
  Two-layer network: "Do those lines form a face?"
"""

"""
Dimensions:
- Input: (m, 2)        - m samples, 2 features
- Layer 1: 3 neurons   - (2, 3) weights + (1, 3) bias
- Layer 2: 1 neuron    - (3, 1) weights + (1, 1) bias
- Output: (m, 1)       - m predictions

"""

# TWO LAYER NETWORK CLASS 

class TwoLayerNetwork : # Neural network with 2 layer 

    def __init__(self, input_size=2, hidden_size=3, output_size=1, learning_rate=0.01):

        """
Args:
            input_size: Number of input features
            hidden_size: Number of hidden neurons
            output_size: Number of output neurons
            learning_rate: Learning rate for gradient descent
"""

        self.learning_rate = learning_rate

        # Layer 1: input  -> hidden
        self.w1 = np.random.rand(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        # Layer 2: Hidden -> output 
        self.w2 = np.random.rand(hidden_size , output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

        print("✓ Two-Layer Network initialized:")
        print(f"  Input size: {input_size}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Output size: {output_size}")
        print(f"  W1 shape: {self.w1.shape}")
        print(f"  b1 shape: {self.b1.shape}")
        print(f"  W2 shape: {self.w2.shape}")
        print(f"  b2 shape: {self.b2.shape}")


    def forward(self, X):
         """
        Forward propagation through both layers
        Args:
            X: Input data (m, input_size)
        
        Returns:
            a2: Output (m, output_size)
        """

         # layer 1 : Linear combination 

         self.z1 = np.dot(X, self.w1) + self.b1 # (m, hidden_size)

         #activation od ReLU 
         self.a1 = np.maximum(0, self.z1) # (m, hidden_size)

        # Layer 2  : Linear combination 
         self.z2 = np.dot(self.a1, self.w2) + self.b2 # (m, output_size)

         # activation of sigmoid classification 
         self.a2 = 1 /(1+ np.exp(-np.clip(self.z2, -500, 500))) # (m, output_size)

         #save for backward pass
         self.X = X
         return self.a2

    def backward(self, y):
        """
        Backward propagation through both layers
        
        Args:
            y: Target values (m, output_size)
        """
        m = self.X.shape[0]

        # Layer 2 : Backward 
        # error at output 

        dz2 = self.a2 - y #(m, hidden_size)

        # Gradients

        dw2 = (1/m) * np.dot(self.a1.T, dz2) #(hidden_size, output_size)
        db2 = (1/m) * np.sum(dz2, axis =0, keepdims=True) # (1, output_size)

        # Layer 1 backward 

        # error propagated back
        da1 = np.dot(dz2, self.w2.T) #(m, hidden_size)

        # ReLU derivative: 1 if z1 > 0,else 0

        dz1 = da1 * (self.z1 > 0).astype(float)

        # gradients 
        dw1 = (1/m) * np.dot(self.X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)

        # Update 

        self.w1 -= self.learning_rate * dw1
        self.b1 -= self.learning_rate * db1
        self.w2 -= self.learning_rate * dw2
        self.b2 -= self.learning_rate * db2

    def compute_losee(self, y_pred, y_true):
        """
        binnary cross entropy loss 
        """
        epsilon = 1e-8
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    def predict( self, X):
        """ Make prediction"""
        return self.forward(X)

print("\n3. CREATING DATASET")
print("-"*70)
 
print("\nProblem: Classify points in 2D space")
print("  Class 0: Points in bottom-left")
print("  Class 1: Points in top-right")
 
# Generate data
np.random.seed(42)
 
# Class 0 (negative)
X0 = np.random.randn(50, 2) - 1  # Mean at (-1, -1)
y0 = np.zeros((50, 1))
 
# Class 1 (positive)
X1 = np.random.randn(50, 2) + 1  # Mean at (1, 1)
y1 = np.ones((50, 1))
 
# Combine
X_train = np.vstack([X0, X1])
y_train = np.vstack([y0, y1])
 
print(f"\nTraining data:")
print(f"  Total samples: {len(X_train)}")
print(f"  Class 0: {len(y0)} samples")
print(f"  Class 1: {len(y1)} samples")
print(f"  Features: 2 (x, y coordinates)")

# Train network 

print("\n TRAINING THE NETWORK")
print("* *" * 25)

# create network 
network = TwoLayerNetwork(input_size=2, hidden_size=3, output_size=1,learning_rate=0.1)

print("\n TRAINING FOR 1000 EPOCH ..")

train_losses =[]
train_accuracies = []

for epoch in range(1000):

    # Forward pass 
    y_pred = network.forward(X_train)

    # claculate loss 
    loss = network.compute_losee(y_pred, y_train)
    train_losses.append(loss)

    # calculate the accuracy 

    y_pred_binary = (y_pred > 0.5 ).astype(int)
    accuracy = np.mean(y_pred_binary == y_train)
    train_accuracies.append(accuracy)

    # backward pass
    network.backward(y_train)

    # print progress

    if epoch % 100 == 0:
        print(f"Epoch { epoch:4d}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")


print(f"\n✓ Training complete!")
print(f"Final Loss: {train_losses[-1]:.4f}")
print(f"Final Accuracy: {train_accuracies[-1]:.4f}")


# VISUALIZE DECISION BOUNDARY

print("\n5. VISUALIZING DECISION BOUNDARY")
print("-"*70)
 
# Create grid for decision boundary
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
 
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
 
# Predict on grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = network.predict(grid_points)
Z = Z.reshape(xx.shape)
 
# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
 
# Plot 1: Decision Boundary
ax = axes[0, 0]
ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.3)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.scatter(X_train[y_train[:, 0] == 0, 0], X_train[y_train[:, 0] == 0, 1], 
          color='blue', s=100, label='Class 0', alpha=0.7)
ax.scatter(X_train[y_train[:, 0] == 1, 0], X_train[y_train[:, 0] == 1, 1], 
          color='red', s=100, label='Class 1', alpha=0.7)
ax.set_title('Decision Boundary - 2 Layer Network', fontsize=12, fontweight='bold')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
ax.grid(True, alpha=0.3)
 
# Plot 2: Training Loss
ax = axes[0, 1]
ax.plot(train_losses, 'b-', linewidth=2)
ax.set_title('Training Loss Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Binary Cross Entropy Loss')
ax.grid(True, alpha=0.3)
 
# Plot 3: Training Accuracy
ax = axes[1, 0]
ax.plot(train_accuracies, 'g-', linewidth=2)
ax.set_title('Training Accuracy Over Time', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1.05])
ax.grid(True, alpha=0.3)
 
# Plot 4: Network Predictions vs Actual
ax = axes[1, 1]
y_pred_final = network.predict(X_train)
ax.scatter(y_train, y_pred_final, alpha=0.6, s=50)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_title('Predictions vs Actual', fontsize=12, fontweight='bold')
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_xlim([-0.1, 1.1])
ax.set_ylim([-0.1, 1.1])
ax.legend()
ax.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('two_layer_network.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: two_layer_network.png")
plt.show()
 
 
# ============ TEST PREDICTIONS ============
 
print("\n6. TESTING ON NEW DATA")
print("-"*70)
 
# Test points
X_test = np.array([
    [-2, -2],  # Far bottom-left (should be class 0)
    [0, 0],    # Middle (boundary)
    [2, 2],    # Far top-right (should be class 1)
    [-1, 1],   # Mixed
    [1, -1]    # Mixed
])
 
print("\nTest predictions:")
test_pred = network.predict(X_test)
 
for i, (point, pred) in enumerate(zip(X_test, test_pred)):
    confidence = max(pred[0], 1 - pred[0]) * 100
    class_pred = "1 (Red)" if pred[0] > 0.5 else "0 (Blue)"
    print(f"  Point {i+1} {point}: Predict {class_pred} ({confidence:.1f}% confidence)")
 
 
# ============ LAYER ANALYSIS ============
 
print("\n7. ANALYZING HIDDEN LAYER")
print("-"*70)
 
print("\nHidden layer is where magic happens!")
print("It learns features that help classify")
 
# Get hidden layer output
hidden_output = np.maximum(0, np.dot(X_train, network.w1) + network.b1)
 
print(f"\nHidden layer outputs:")
print(f"  Shape: {hidden_output.shape}")
print(f"  3 neurons learned 3 different features")
print(f"  First 5 samples, 3 neurons:")
print(hidden_output[:5])
 
print("\nEach hidden neuron learned to detect different patterns!")
 
 
# ============ KEY INSIGHTS ============
 

print("-"*70)
"""
1. LAYER STRUCTURE
   - Layer 1: Input → Hidden
     Learns low-level features (lines, edges, patterns)
   - Layer 2: Hidden → Output
     Combines features to make final decision
 
2. FORWARD PASS FLOW
   X → Linear1 → ReLU → Linear2 → Sigmoid → Predictions
   (input) (Z1)    (A1)  (Z2)     (A2)
 
3. BACKWARD PASS FLOW
   Error flows backward
   ∂L/∂W2 ← ∂L/∂Z2 ← ∂L/∂A1 ← ∂L/∂Z1 ← ∂L/∂W1
   Gradients multiply through chain rule (backpropagation!)
 
4. POWER OF 2 LAYERS
   - Layer 1 extracts features
   - Layer 2 uses features to make decisions
   - Together: Can learn any pattern!
 
5. PARAMETERS TO TUNE
   - Hidden layer size: More neurons = more power (but slower)
   - Learning rate: Too small = slow, too large = diverges
   - Epochs: More = better fit (but risk overfitting)
"""