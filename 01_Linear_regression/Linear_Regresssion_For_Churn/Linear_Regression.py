"""
LINEAR REGRESSION ON TELECOM CUSTOMER CHURN

MATH Logic :
 - fit line into y = w*x + c
 - minimize: MSE = mean((y_pre - y_true)^2)
 - Normal equation
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from utils import RegressionMetrics

# Simple house price dataset

square_feet = np.array([1200, 1400, 1600, 1800, 2000, 
                        2200, 2400, 2600, 2800, 3000]).reshape(-1, 1)
price = np.array([300, 350, 400, 450, 500, 
                  550, 600, 650, 700, 750])
 
# SPLIT DATA 
X_train, X_test, y_train, y_test = train_test_split(
    square_feet, price, test_size=0.2, random_state=42
)
 
# FIT MODEL 
model = LinearRegression()
model.fit(X_train, y_train)

 
# ============ PREDICT ============
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
 
# ============ METRICS ============
train_metrics = RegressionMetrics.calculate(y_train, y_pred_train, "Linear Regression (Train)")
test_metrics = RegressionMetrics.calculate(y_test, y_pred_test, "Linear Regression (Test)")
 
# ============ MODEL PARAMETERS ============
print(f"Slope (w):          {model.coef_[0]:.4f}")
print(f"Intercept (b):      {model.intercept_:.4f}")
print(f"Equation: Price = {model.coef_[0]:.4f} * SquareFeet + {model.intercept_:.4f}")
 
# ============ PREDICTION EXAMPLE ============
new_area = np.array([[2500]])
predicted_price = model.predict(new_area)[0]
print(f"\n🏠 For 2500 sq.ft house → Predicted Price: ${predicted_price:.2f}k")
 
# ============ VISUALIZATION ============
plt.figure(figsize=(10, 6))
 
# Scatter: actual data
plt.scatter(X_train, y_train, color='blue', label='Training data', s=100, alpha=0.7)
plt.scatter(X_test, y_test, color='red', label='Test data', s=100, alpha=0.7)
 
# Line: fitted model
X_range = np.linspace(square_feet.min(), square_feet.max(), 100).reshape(-1, 1)
y_line = model.predict(X_range)
plt.plot(X_range, y_line, color='green', linewidth=2, label='Fitted line')
 
plt.xlabel('Square Feet (X)', fontsize=12)
plt.ylabel('Price (Y)', fontsize=12)
plt.title('Linear Regression: House Price Prediction', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_linear_regression_plot.png', dpi=100)
print("\n✅ Plot saved: 1_linear_regression_plot.png")
plt.show()
 
