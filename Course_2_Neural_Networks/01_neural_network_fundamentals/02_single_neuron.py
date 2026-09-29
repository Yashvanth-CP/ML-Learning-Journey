"""
02. SINGLE NEURON
"""

import numpy as np
import matplotlib.pyplot as plt
 
print("="*60)
print("SINGLE NEURON: The Foundation")
print("="*60)

# what is a neuron ?
"""
A neuron is simply:
 
INPUT ──→ [LINEAR COMPUTATION] ──→ [ACTIVATION] ──→ OUTPUT
   x           z = w·x + b           a = f(z)           a
 
Components:
1. Weights (w): Strength of connection
2. Bias (b): Base value
3. Linear combination: z = w·x + b
4. Activation: a = f(z) (adds non-linearity)
5. Output: a (prediction or intermediate output)

"""

# SINGLE NEURON CLASS

class Neuron:
    # a single neuron that can learn fron data

    def __init__(self, input_size, activation='relu', learning_rate=0.01):
        self.input_size = input_size # Number of inputs 
        self.activation_type = activation # relu, sigmoid or linear
        self.learning_rate = learning_rate # how fast to learn

        # Initialize weight randomly (Small values)

        self.w = np.random.randn(input_size) * 0.01
        self.b = 0.0

        print(f"Neuron initialized:")
        print(f" Weight shape :{self.w.shape}")
        print(f" Bias : {self.b}")
        print(f" Activation: {activation}")


    def _activation(self, z):
        # apply activation function 

        if self.activation_type == 'relu':
            return np.maximum(0,z)
        elif self.activation_type == 'sigmoid':
            z = np.clip(z, -500, 500)
            return 1/(1 + np.exp(-z))
        elif self.activation_type == 'linear':
            return z
        else:
            raise ValueError(f"Unknown acivation: {self.activation_type}")

    def _activation_derivative(self, z, a):
        # Calculate gradient of activation

        if self.activation_type == 'relu':
            return (z > 0).astype(float)

        elif self.activation_type == 'sigmoid':
            return a * (1 -a)
        elif self.activation_type == 'linear':
            return np.ones_like(z)


    def forward(self, X):

        """
        forward pass: compute output 
        args : X + input data 
        returns : a : Output
        """

        if X.ndim == 1: # X shape : (1,) for single sample or (m,) for batch
            X = X.reshape(-1, 1)

        self.z = np.dot(X, self.w) + self.b # Linear Combination
        self.a = self._activation(self.z) # Activation 
        self.X = X # save this for backward pass 
        
        

        return self.a 

    def backward(self, y):
        """
        Backward pass :calculate gradient and update weights
        Args: Y: Actual output (target)
        """ 
        batch_size = self.X.shape[0]

        error = self.a - y # claculate error 
        a_prime = self._activation_derivative(self.z, self.a) # Gradient of activation
        dz = error * a_prime # gradient of loss w.r.t. z

        # Gradient for weight and bias
        dw = (1/ batch_size) * np.dot(self.X.T, dz)
        db = (1 / batch_size) * np.sum(dz)

        # Update weight 

        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def predict(self, X):
            # Make prediction on new data
            return self.forward(X)

    def get_weights(self):
            # return current weigth 
            return self.w.copy(), self.b

    

# EXAMPLE 1: Simple Binary Classification

print("\n2. EXAMPLE 1: Binary Classification")
print("-"*60)
"""
    Scenario: Predict if someone bought coffee
    Input: Temperature (°F)
    Output: Buy coffee? (1=yes, 0=no)
"""

# Generate data 

np.random.seed(42)
X_cold = np.random.uniform(40, 60, 20) # cold days
Y_cold = np.ones(20)# people buy coffee
 
X_hot = np.random.uniform(75, 95, 20) # Hot days
Y_hot = np.zeros(20) # People don't buy coffee

X_train  = np.concatenate([X_cold, X_hot])
Y_train = np.concatenate([Y_cold, Y_hot])

print(f"\n Triainig data : ")
print(f" Cold days (40- 60F): {len(X_cold)} sample -> Buy (1)")
print(f" Hot days (75-95F): {len(X_hot)} sample -> Don't buy (0)")

# Normalize input 
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)

# create and train neuron
 
neuron1 = Neuron(input_size=1, activation='sigmoid', learning_rate=0.1)

print("\n Training for 200 epochs...")
losses =[]

for epoch in range(200):

    prediction = neuron1.forward(X_train_norm)
    loss = -np.mean(Y_train * np.log(prediction + 1e-8) + (1 - Y_train) * np.log(1 - prediction + 1e-8))
    losses.append(loss)

    neuron1.backward(Y_train) # Backward pass

    if epoch % 40 == 0:
        acc = np.mean((prediction > 0.5) == Y_train)
        print(f" Epoch {epoch:3d}: loss = {loss:.4f}, Accuracy = {acc:.4f}")

print("\n Training complete !")

# test on new data 

X_test = np.array([45, 65, 85]) # Cold, medium hot
X_test_norm = (X_test - np.mean(X_test )) / np.std(X_train)
prediction_test = neuron1.predict(X_test_norm)


print("\nPredictions on new temperatures:")

for temp, pred in zip(X_test, prediction_test) :
     print(f" {temp:>2}F: {pred:.4f} (yes if > 0.5) -> {'BUY' if pred > 0.5 else 'No NUY'}")


w, b = neuron1.get_weights()
print(f"\n Learned weight: w = {w[0]:.4f}, b={b:.4f}")


# EXAMPLE 2 : Regrassion 

print("\n\n 3. Example 2 : Regression: ")
print("*" *60)

"""
    Scenario: Predict coffee price based on quality
    Input: Quality score (1-10)
    Output: Price ($)
"""

# Generate the data 

X_quality = np.random.uniform(1, 10, 20)
Y_price = 2 * X_quality + 1 + np.random.normal(0,2,20) # Price ~ 2 * quality + 1

print(f"\nData : {len(X_quality)} coffee samples")

# Normalize 

X_quality_norm = (X_quality - np.mean(X_quality)) / np.std(X_quality)
Y_price_norm = (Y_price - np.mean(Y_price) ) / np.std(Y_price)

# Train neuron (linear for regression)

neuron2 = Neuron(input_size=1, activation='linear', learning_rate=0.1)

print("Training for 200 epochs..")
losses2 = []

for epoch in range(200):
    predictions = neuron2.forward(X_quality_norm) # Forward

    loss = np.mean((predictions - Y_price_norm) ** 2) # MSE loss
    losses2.append(loss)

    neuron2.backward(Y_price_norm) # Backward

    if epoch % 40 == 0:
        print(f" Epoch {epoch:3d}: Loss(MSE) = {loss:.4f}")

print("\n Training complete!")

# Predict 
X_test_quality = np.array([3, 5, 8])
X_test_quality_norm = (X_test_quality - np.mean(X_test_quality)) / np.std(X_test_quality)

predictions_price = neuron2.predict(X_quality_norm)
preditions_price_denorm = predictions_price * np.std(Y_price) + np.mean(Y_price)

print("\n Price PRediction ")
for quality, price in zip(X_test_quality, preditions_price_denorm):
    print(f" Quality {quality}/10: ${price:.2f}")

# ============ VISUALIZATION ============
 
print("\n\n4. VISUALIZATION")
print("-"*60)
 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
 
# Plot 1: Classification Training Loss
axes[0, 0].plot(losses, 'b-', linewidth=2)
axes[0, 0].set_title('Classification: Training Loss', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss (Binary Cross Entropy)')
axes[0, 0].grid(True, alpha=0.3)
 
# Plot 2: Classification Decision Boundary
X_plot = np.linspace(-3, 3, 100)
y_pred_plot = neuron1.predict(X_plot)
 
axes[0, 1].scatter(X_train_norm[Y_train == 1], Y_train[Y_train == 1], 
                   color='red', s=100, label='Buy Coffee', alpha=0.7)
axes[0, 1].scatter(X_train_norm[Y_train == 0], Y_train[Y_train == 0], 
                   color='blue', s=100, label='No Coffee', alpha=0.7)
axes[0, 1].plot(X_plot, y_pred_plot, 'g-', linewidth=3, label='Neuron Output')
axes[0, 1].axhline(y=0.5, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Classification: Decision Boundary', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Temperature (normalized)')
axes[0, 1].set_ylabel('P(Buy Coffee)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
 
# Plot 3: Regression Training Loss
axes[1, 0].plot(losses2, 'r-', linewidth=2)
axes[1, 0].set_title('Regression: Training Loss', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss (MSE)')
axes[1, 0].grid(True, alpha=0.3)
 
# Plot 4: Regression Fit
X_plot_reg = np.linspace(-2.5, 2.5, 100)
y_pred_plot_reg = neuron2.predict(X_plot_reg)
y_pred_plot_reg_denorm = y_pred_plot_reg * np.std(Y_price) + np.mean(Y_price)
X_plot_reg_denorm = X_plot_reg * np.std(X_quality) + np.mean(X_quality)
 
axes[1, 1].scatter(X_quality, Y_price, color='blue', s=100, label='Training Data', alpha=0.7)
axes[1, 1].plot(X_plot_reg_denorm, y_pred_plot_reg_denorm, 'g-', linewidth=3, label='Neuron Fit')
axes[1, 1].set_title('Regression: Price Prediction', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Quality (1-10)')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('single_neuron_training.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: single_neuron_training.png")
plt.show()
 
 
# KEY INSIGHTS 
print("""
1. SINGLE NEURON POWER
   - With non-linear activation: Can learn any pattern
   - Can solve both classification and regression
   - Foundation for building larger networks
 
2. FORWARD PASS
   - Input → Linear (w·x + b) → Activation → Output
   - Simple but powerful!
 
3. BACKWARD PASS
   - Calculate error: actual - predicted
   - Flow gradient back through activation
   - Update weights to reduce error
 
4. ACTIVATION MATTERS
   - Sigmoid for classification (outputs probability)
   - Linear for regression (outputs any number)
   - ReLU for hidden layers
 
5. LEARNING RATE
   - Too small: Training is slow
   - Too large: Might overshoot
   - 0.01-0.1 usually works well
""")
 