My ML Learning Journey 🚀

Hey! This is where I'm documenting my journey learning Machine Learning. I'm taking the DeepLearning.AI course and building from fundamentals because I want to actually understand ML, not just copy code.

📌 About This Repo

I created this to keep track of all the ML concepts I'm learning - from basic linear regression to regularization. Each file has working code, clear explanations, and the graphs/visualizations I generated while learning.

My goal: Build a really strong foundation in ML before moving to Deep Learning and eventually Generative AI.

📚 What I've Covered So Far
1.Linear Regression = 	Cost functions, gradient descent, optimization
2.Classification =  	Sigmoid, logistic loss, decision boundaries
3.Overfitting & Regularization = 	L1, L2, preventing overfitting


ML-Learning-Journey/
│
├── 01_linear_regression/
│   ├── 01_cost_function.py          # How cost function works
│   ├── 02_gradient_descent.py        # Automatic optimization
│   ├── 03_polynomial_regression.py   # Curved relationships
│   ├── 04_all_in_one.py              # Real example (house prices)
│   └── README.md
│
├── 02_classification/
│   ├── 05_decision_boundary.py       # Separating classes
│   ├── 06_sigmoid_logistic.py        # Converting to probabilities
│   ├── 07_regularization.py          # Preventing overfitting
│   ├── 08_complete_pipeline.py       # Full pipeline on real data
│   ├── 09_quick_reference.py         # All formulas
│   └── README.md
│
├── 03_overfitting/
│   ├── overfitting_regularization.py # Complete demo
│   ├── graphs/                        # My generated plots
│   └── README.md
│
├── datasets/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── requirements.txt
├── .gitignore
└── README.md

🎯 Key Concepts I Learned
Linear Regression

Cost function tells us how wrong the model is. Gradient descent automatically adjusts the parameters (w and b) to minimize this cost. It's like going downhill on a mountain - we keep taking steps in the direction that reduces cost the most.

Formula: J(w,b) = (1/2m)Σ(ŷ-y)²

Classification

Instead of predicting numbers, we predict categories (0 or 1). The sigmoid function converts any number into a probability between 0 and 1. Logistic loss is special for classification - it heavily punishes wrong predictions.

Key insight: The gradient update formula is the same as linear regression! Only the prediction function changes.

Overfitting & Regularization

When the model memorizes training data instead of learning patterns, it fails on new data. Regularization adds a penalty to the cost function to keep weights small and prevent overfitting. L2 (Ridge) is for general use, L1 (Lasso) is for removing unnecessary features.

Tech Stack : 
Python 3.8+
├── NumPy          # Numbers & arrays
├── Pandas         # Data handling
├── Matplotlib     # Making plots
├── Scikit-learn   # ML algorithms
└── SciPy          # Math stuff

📖 How to Use : 

Install everything : pip install -r requirements.txt


Run any script:
python 01_linear_regression/01_cost_function.py
python 02_classification/06_sigmoid_logistic.py
python 03_overfitting/overfitting_regularization.py


Check the outputs: 

Graphs get saved in the graphs/ folders. Console output shows

💡 How I Explain Things

Every concept follows this structure:

Simple meaning → What is it?
Real example → Why do we need it?
Math explanation → How does it work?
Python code → Let's implement it
Visualization → Let's see it in action
Practice → Try it yourself
📈 Code Quality

✅ Well-commented code
✅ Following PEP 8 (clean Python)
✅ Fixed random seeds (reproducible)
✅ Saves graphs automatically

🎓 My Learning Goals

Short term: Master ML fundamentals + DSA
Mid term: Build real AI/ML projects
Long term: Deep Learning + Generative AI expert

⚠️ Things I Learned the Hard Way
Learning rate matters! Too small = slow, too large = diverges
Polynomial features can overfit - high degree looks perfect on training data but fails on new data
Order of operations matters - plot first, THEN save
Regularization is essential - most real models need it
🔗 Useful Links
DeepLearning.AI
Scikit-learn
NumPy Docs
📝 Notes to Future Me
Don't skip the math - it actually makes sense once you understand it
Visualization is your friend - plot everything
Small learning rate is usually better than guessing
Check train vs test error always

Last Updated: September 2026
Status: 🟢 Still Learning!