"""
PROJECT 1: DEMAND PREDICTION
Step 3: Build Neural Network Model
"""

import numpy as np

# MOdel type  : Regression 

class RegressionNetwork : 
    # neuralNetwork for regression from scratch 

    def __init__(self, input_size=6, hidden_sizes=[32,16,8], learning_rate = 0.01, dropout_rate = 0.3):
        """
            Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            learning_rate: Learning rate for gradient descent
            dropout_rate: Dropout rate (0-1)
        """
        self.learning_rate = learning_rate
        self.dropoit_rate = dropout_rate

        # layer 1 : Input -> Hidden 1
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * 0.01
        self.b1 = np.zeros((1, hidden_sizes[0]))

        # Layer 2 : Hidden 1 -> Hidden 2
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * 0.01
        self.b2 = np.zeros((1, hidden_sizes[1]))

        # layer 3 : hidden 2 -> hidden -> 3
        self.W3 = np.random.randn(hidden_sizes[1], hidden_sizes[2]) * 0.01
        self.b3 = np.zeros((1, hidden_sizes[2]))

        # output : hidden -> output
        self.W4 = np.random.randn(hidden_sizes[2], 1) * 0.01
        self.b4 = np.zeros((1,1))

        print("Regression Network Created!")
        print(f"  Input size: {input_size}")
        print(f"  Hidden layers: {hidden_sizes}")
        print(f"  Output: 1 (regression)")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Dropout rate: {dropout_rate}")

        print(f"\n Parameters:")
        total_params = (input_size * hidden_sizes[0] + hidden_sizes[0] + 
                        hidden_sizes[0]*hidden_sizes[1] + hidden_sizes[1]+ 
                        hidden_sizes[1] * hidden_sizes[2] +hidden_sizes[2] +
                        hidden_sizes[2] * 1+1)
        print(f"total : {total_params} parameters")

    def forward(self, X, training=True) : 
        # forward propagation 
        self.X = X 
        self.training = training

        # later 2 
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1) # ReLU 

        # DropOut 1

        if training :
            self.mask1 = np.random.binomial(1, 1 - self.dropout_rate, self.A1.shape)
            self.A1_dropped = (self.A1 * self.mask1) / (1 - self.dropout_rate)

        else:
             self.A1_dropped = self.A1

        # layer 2

        self.Z2 = np.dot(self.A1_dropped, self.W2) + self.b2
        self.A2 = np.maximum(0, self.Z2) # ReLU 


        # Dropout 2 

        if training :
            self.mask2 = np.random.binomial(1, 1- self.dropout_rate, self.A2.shape)
            self.A2_dropped = (self.A2 * self.mask2) / (1 - self.dropoit_rate)

        else : 
            self.A2_dropped = self.A2

        # Layer 3 

        self.Z3 = np.dot(self.A2_dropped, self.W3) + self.b3
        self.A3 = np.maximum(0, self.Z3)  # ReLU
        
        # Output layer (NO activation for regression!)
        self.Z4 = np.dot(self.A3, self.W4) + self.b4
        self.output = self.Z4  # Linear output
        
        return self.output
    
    def backward(self, y):
        # Backward propagation 
        m = self.X.shape[0]

        # layer 4

        dZ4 = self.output - y # MSE defivative 
        dW4 = (1/m) * np.dot(self.A3.T, dZ4)
        db4 = (1/m) * np.sum(dZ4, axis=0, keepdims=True)

         # LAYER 3 BACKWARD 
        dA3 = np.dot(dZ4, self.W4.T)
        dZ3 = dA3 * (self.Z3 > 0)  # ReLU derivative
        dW3 = (1/m) * np.dot(self.A2_dropped.T, dZ3)
        db3 = (1/m) * np.sum(dZ3, axis=0, keepdims=True)

        # layer 2 Backward 
        A2 = np.dot(dZ3, self.W3.T)
        if self.training:
            dA2 = dA2 * self.mask2 / (1 - self.dropout_rate)
        dZ2 = dA2 * (self.Z2 > 0)  # ReLU derivative
        dW2 = (1/m) * np.dot(self.A1_dropped.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # layer 1 Backward 
        dA1 = np.dot(dZ2, self.W2.T)
        if self.training:
            dA1 = dA1 * self.mask1 / (1 - self.dropout_rate)
        dZ1 = dA1 * (self.Z1 > 0)  # ReLU derivative
        dW1 = (1/m) * np.dot(self.X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # Update Weights 
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W4 -= self.learning_rate * dW4
        self.b4 -= self.learning_rate * db4

    def compute_loss(self, y_pred, y_true):
        """Mean Squared Error (MSE)"""
        return np.mean((y_pred - y_true) ** 2)
    
    def compute_r2(self, y_pred, y_true):
        """R² score (how well predictions match)"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def predict(self, X):
        """Make predictions"""
        return self.forward(X, training=False)

# Load data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy').reshape(-1, 1)
X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy').reshape(-1, 1)

# Create model
model = RegressionNetwork(
    input_size=6,
    hidden_sizes=[32, 16, 8],
    learning_rate=0.01,
    dropout_rate=0.3
)

# Test forward pass
y_pred_test = model.forward(X_test[:5], training=False)
print(f"Sample predictions:")
print(f"  First 5 predictions: {y_pred_test.flatten()}")
print(f"  First 5 actual: {y_test[:5].flatten()}")

"""
Network Architecture:
 
Input (6)
    ↓
Dense(32) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(16) + ReLU
    ↓
Dropout(0.3)
    ↓
Dense(8) + ReLU
    ↓
Dense(1) + Linear (Output)
    ↓
Regression Output
 
Loss function: Mean Squared Error (MSE)
  MSE = mean((y_pred - y_true)²)
 
Metric: R² Score
  R² = 1 - (SS_res / SS_tot)
  Ranges from 0 to 1 (1 = perfect)
 
Next step: Train the model!
"""
