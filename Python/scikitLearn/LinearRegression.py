from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
reg = LinearRegression()

X = [[0, 0],
     [1, 1],
     [2, 2],
     [0, 1]]

y = [0, 1, 2, 1]

reg.fit(X, y)
print("Weights:", reg.coef_)
print("Bias:", reg.intercept_)

print(reg.predict([[10, 10]]))

# Non-Negative Least Squares

# Ridge 
print("Ridge : ")
reg = Ridge(alpha= 0.5)
Ridge(alpha=0.5)
reg.fit(X,y)
print("Weight : ",reg.coef_)
print("Biase : ", reg.intercept_)

# Ridge classifier 

