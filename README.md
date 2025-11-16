# Linear-Regression
# A Linear Regression Model that predict y from x
#predicting y from x
import numpy as np,pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LinearRegression

np.random.seed(0)
x=np.random.rand(100,1)*10
a=np.random.randn(100,1)*2
# y=a+bx 
y=a+2.5*x

model=LinearRegression()
model.fit(x,y)

y_pred=model.predict(x)
print("Intercept",model.intercept_)
print("slope",model.coef_)

plt.scatter(x,y,color='red',label="data points")
plt.plot(x,y_pred,color='blue',linewidth=2,label="Regression line")
plt.xlabel("x")
plt.ylabel("Y")
plt.legend()
plt.show()
