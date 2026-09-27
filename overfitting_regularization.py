"""
OVERFITTING & REGULARIZATION:  
Learn how to detect and prevent overfitting using regularization.
"""

import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split

print("=" * 50)
print("Part 1 : understadding OVERFITTING ")
print("=" * 50)

print("Example 1: OVERFITTING WITH POLYNOMIAL DEGREES\n")


# Generate data with noise 
np.random.seed(42)
X_train_raw = np.linspace(0, 10, 20)
Y_train_raw = 2 + 0.5*X_train_raw

# Generate test data (cleaner, represents true pattern)
X_test_raw = np.linspace(0, 10, 100)
Y_test_raw = 2 + 0.5*X_test_raw

#Reshape for sklearn

X_train = X_train_raw.reshape(-1,1)
Y_train = Y_train_raw
X_test = X_test_raw.reshape(-1, 1)
Y_test = Y_test_raw

print(f"Training data: {len(X_train)} samples (with noise)")
print(f"Test data : {len(X_test)} samples (clean, represent true pattern)")
print(f"True relationship : y = 2 + 0.5*x \n")

#  Train model with different polynomial  degrees 

degrees = [1, 3, 10, 15]
models = {}
train_errors = {}
test_errors = {}

print(f"{'Degree':<10} {'Train Error':<20} {'Test Error':<20} {'Overfitting?':<15}")
print("-"*65)

for degree in degrees:
    # creste polynomial feature 
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)

    #Train model 
    model = Ridge(alpha=0.0)
    model.fit(X_train_poly, Y_train)

    # Calculate error 
    Y_train_pred = model.predict(X_train_poly)
    Y_test_pred = model.predict(X_test_poly)

    train_error = np.mean((Y_train_pred - Y_train) ** 2)
    test_error = np.mean((Y_test_pred - Y_test) ** 2)

    models[degree] = (model, poly)
    train_errors[degree] = train_error
    test_errors[degree] = test_error
    
    # Detect overfitting 
    overfitting_ratio = test_error / train_error
    is_overfitting = "yes " if overfitting_ratio > 2 else " NO"

    print(f"{degree:<10} {train_error:<20.4f} {test_error:<20.4f} {is_overfitting:<15}")
 
