import numpy as np

# 1. Define the Activation Function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Single Neuron Class
class SingleNeuralNetwork:
    def __init__(self, input_size):
        # Initialize weights randomly with a mean of 0, and bias to 0
        np.random.seed(42)
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

    def forward(self, inputs):
        # Linear combination + Activation function: y = sigmoid(w*x + b)
        dot_product = np.dot(inputs, self.weights) + self.bias
        return sigmoid(dot_product)

    def train(self, training_inputs, training_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(training_inputs)

            # Calculate error (Target - Prediction)
            error = training_outputs - outputs

            # Backpropagation: gradient descent adjustment
            # Gradient = Error * Derivative of sigmoid at the output
            adjustments = error * sigmoid_derivative(outputs)

            # Update weights and bias
            self.weights += np.dot(training_inputs.T, adjustments) * learning_rate
            self.bias += np.sum(adjustments, axis=0, keepdims=True) * learning_rate

            # Print loss every 2000 epochs
            if (epoch + 1) % 2000 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")

# --- Example Usage (Learning the Logical AND gate) ---

# Training Data (Inputs)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target Outputs (AND gate logic)
y = np.array([[0], 
              [0], 
              [0], 
              [1]])

# Initialize neural network with 2 inputs
neuron = SingleNeuralNetwork(input_size=2)

print("Training the single neuron...")
neuron.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained neuron
print("\nPredictions after training:")
for test_input in X:
    prediction = neuron.forward(test_input)
    print(f"Input: {test_input} -> Predicted: {prediction[0][0]:.4f} (Rounded: {round(prediction[0][0])})")

    import numpy as np

# 1. Define the Activation Function (Sigmoid) and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. Single Neuron Class
class SingleNeuralNetwork:
    def __init__(self, input_size):
        # Initialize weights randomly with a mean of 0, and bias to 0
        np.random.seed(42)
        self.weights = np.random.randn(input_size, 1)
        self.bias = np.zeros((1, 1))

    def forward(self, inputs):
        # Linear combination + Activation function: y = sigmoid(w*x + b)
        dot_product = np.dot(inputs, self.weights) + self.bias
        return sigmoid(dot_product)

    def train(self, training_inputs, training_outputs, epochs, learning_rate):
        for epoch in range(epochs):
            # Forward pass
            outputs = self.forward(training_inputs)

            # Calculate error (Target - Prediction)
            error = training_outputs - outputs

            # Backpropagation: gradient descent adjustment
            # Gradient = Error * Derivative of sigmoid at the output
            adjustments = error * sigmoid_derivative(outputs)

            # Update weights and bias
            self.weights += np.dot(training_inputs.T, adjustments) * learning_rate
            self.bias += np.sum(adjustments, axis=0, keepdims=True) * learning_rate

            # Print loss every 2000 epochs
            if (epoch + 1) % 2000 == 0:
                loss = np.mean(np.square(error))
                print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss:.6f}")

# --- Example Usage (Learning the Logical AND gate) ---

# Training Data (Inputs)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Target Outputs (AND gate logic)
y = np.array([[0], 
              [0], 
              [0], 
              [1]])

# Initialize neural network with 2 inputs
neuron = SingleNeuralNetwork(input_size=2)

print("Training the single neuron...")
neuron.train(X, y, epochs=10000, learning_rate=0.1)

# Test the trained neuron
print("\nPredictions after training:")
for test_input in X:
    prediction = neuron.forward(test_input)
    print(f"Input: {test_input} -> Predicted: {prediction[0][0]:.4f} (Rounded: {round(prediction[0][0])})")