# Linear Regression - My Notes 📈

This is where I learned linear regression from scratch. I went from "what's a cost function?" to actually understanding how models learn. Here's everything I figured out.


# My Files: 

File	:                    What I Learned
01_cost_function.py	         How to measure if predictions are wrong
02_gradient_descent.py	     How the model automatically learns
03_polynomial_regression.py	 Handling curved data (and why it can overfit)
04_all_in_one.py             putting it all together on the reral data

# What I Figured Out
Part 1: Cost Function

At first, I was confused: "How do we know if our model is good?"

Answer: The cost function! It measures how far off our predictions are.

Lower cost = better predictions
Higher cost = worse predictions

# The formula:

J(w,b) = (1/2m) * Σ(prediction - actual)²

I ran 01_cost_function.py and it showed me:

When we have bad parameters, cost is HIGH
When we have good parameters, cost is LOW
The 3D plot shows the cost landscape - like a valley

Why 3D? Because cost depends on 2 parameters: w (weight) and b (bias)

Part 2: Gradient Descent

This was the "aha!" moment for me.

Question: How do we find the best parameters?

Answer: We don't calculate them directly. Instead, we start with random numbers and slowly improve them. This is gradient descent!

# How it works:

Start with w = 0, b = 0
Calculate the gradient (direction of steepest increase)
Move in OPPOSITE direction (downhill)
Repeat until we reach the bottom

The update rule:

w = w - learning_rate * (slope)
b = b - learning_rate * (slope)

Key insight: The gradient tells us which direction to go, and learning rate tells us how big each step is.

# What I learned:

Learning rate too small → takes forever to learn
Learning rate too big → overshoots and diverges
Need to find the "just right" learning rate
Part 3: Polynomial Regression

After mastering linear regression, I tried curved data.

Problem: Real data isn't always a straight line. House prices don't increase linearly - they curve.

Solution: Add polynomial features!
y = w₁x + b                    (straight line)
y = w₁x + w₂x² + b            (curved)
y = w₁x + w₂x² + w₃x³ + b    (more curved)

How it works: I transform the features (x → [x, x², x³]) and then use regular linear regression.

Important: Higher degree = more complex curves, but BEWARE of overfitting!

# What I saw:

Degree 1: Simple line (doesn't fit curved data)
Degree 3: Nice curve (good fit)
Degree 10: Wiggly line (overfitting!)
Part 4: Putting It All Together

In 04_all_in_one.py, I used a real dataset: house prices.

# What I did:

Loaded the data
Created polynomial features (x² for curved relationships)
Split into training and test sets
Trained using gradient descent
Checked accuracy
Made visualizations

Result: I could predict house prices! (with some error, but it worked)


# Key Formulas
What	        Formula
Cost	        J = (1/2m)Σ(ŷ-y)²
Gradient for w	∂J/∂w = (1/m)Σ(ŷ-y)·x
Gradient for b	∂J/∂b = (1/m)Σ(ŷ-y)
Update w	    w ← w - α·∂J/∂w
Update b	    b ← b - α·∂J/∂b
Predict	        ŷ = w·x + b

#  How to Run My Code
bash
python 01_cost_function.py
python 02_gradient_descent.py
python 03_polynomial_regression.py
python 04_all_in_one.py

Each one prints output to console and sometimes saves graphs.

# 📈 What You'll See
Cost Function Output
Testing different parameters...
Cost at w=0, b=0: 2500.0
Cost at w=1, b=0: 150.2
Cost at w=2, b=0: 5.8
...generating cost landscape plot
Gradient Descent Output
Starting gradient descent...
Iteration 0: w = 0.00, b = 0.00, Cost = 2500.00
Iteration 100: w = 1.50, b = 0.20, Cost = 100.50
Iteration 500: w = 1.95, b = 0.05, Cost = 2.30
Iteration 1000: w = 1.98, b = 0.02, Cost = 0.15
✓ Converged!
Polynomial Regression Output
Testing polynomial degrees...
Degree 1: Train Error = 0.5, Test Error = 0.6 ✓
Degree 3: Train Error = 0.1, Test Error = 0.15 ✓
Degree 5: Train Error = 0.01, Test Error = 0.8 ⚠️ OVERFITTING
⚡ Things That Tricked Me

Mistake 1: Wrong learning rate

python
# I tried this and it diverged (cost kept increasing)
learning_rate = 1.0

# This worked better
learning_rate = 0.01

Mistake 2: Forgetting to normalize features

python
# Big numbers = big gradients = unstable
# Solution: Scale features to 0-1 range
from sklearn.preprocessing import StandardScaler

Mistake 3: Not checking if it converged

python
# Just assuming it worked
# Better: Plot cost over iterations and check it decreases smoothly
🧪 Experiments I Tried
Different learning rates: Saw which ones work best
Different polynomial degrees: Found the overfitting point
Different datasets: Tried it on real house price data
Adding regularization: Prevented overfitting
💡 Key Insights
Cost function = scoreboard → tells us how well we're doing
Gradient = direction → tells us which way to go
Learning rate = step size → controls how fast we learn
Iteration = repetition → we keep improving gradually
Polynomial = flexibility → can fit curved data
BUT → overfitting risk → too much flexibility breaks generalization
🎯 Quiz I Asked Myself
What does cost function measure? → Distance from predictions to actual
Why gradient descent? → Automatically finds good parameters
What if learning rate too large? → Diverges, overshoots
What if learning rate too small? → Too slow, takes forever
How do polynomial features help? → Can fit curved relationships
When does overfitting happen? → High degree, small dataset
📚 What I Needed to Know First
Basic Python (loops, functions, lists)
NumPy arrays and operations
Basic math (slope, derivatives, summation)
Matplotlib for plotting
🔗 Next Steps
Learned: How linear regression works
Next: Logistic regression (for classification)
After: Regularization (to prevent overfitting)
📝 Notes to Self
Linear regression is the foundation - understand it deeply
The math gets easier when you see the code and the plot
Always visualize your results - plots are your best friend
Small learning rate is safer than large learning rate
Polynomial features are cool but require regularization

Time spent: About 2 hours
Difficulty: Beginner-friendly once you see examples
Worth it? YES! Understanding the fundamentals is crucial
