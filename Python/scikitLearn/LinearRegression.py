from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
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