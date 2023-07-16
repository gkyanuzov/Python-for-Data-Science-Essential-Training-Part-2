import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import scale

""""

Simple  linear regression

"""
rcParams['figure.figsize'] = 10, 8

rooms = 2 * np.random.rand(100, 1) + 3
print(rooms[1:10])

price = 265 + 6* rooms + abs(np.random.randn(100,1))
print(price[1:10])

plt.plot(rooms, price, 'r^')
plt.xlabel('# of rooms, 2019 average')
plt.ylabel('average home price, 1000s USD')
plt.show()

x = rooms
y = price

linreg = LinearRegression()
linreg.fit(x, y)
print(linreg.intercept_, linreg.coef_)

""""

Simple Algebra notes

"""
# y = mx + b - equation for simple linear regression
# b = intercept
# linreg.coef_ = estimated coefficients for the terms in the lin reg problem

# R^2 - close to 1 = > model performs well
print(linreg.score(x,y))
