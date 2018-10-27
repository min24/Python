import numpy as np 
import matplotlib.pyplot as plt 

# height (cm)
x = np.array([[147, 150, 153, 158, 163, 165, 168, 170, 173, 175, 178, 180, 183]]).T

# weight (kg)
y = np.array([[49, 50, 51, 54, 58, 59, 60, 62, 63, 64, 66, 67, 68]]).T

# Visualize data
plt.plot(x, y, '+')
plt.axis([140, 190, 45, 75])
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
#plt.show()

# Building Xbar
one = np.ones((x.shape[0], 1))
Xbar = np.concatenate((one, x), axis = 1)

"""
one = [[1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]
 [1.]]
"""
"""
Xbar = [[  1. 147.]
 [  1. 150.]
 [  1. 153.]
 [  1. 158.]
 [  1. 163.]
 [  1. 165.]
 [  1. 168.]
 [  1. 170.]
 [  1. 173.]
 [  1. 175.]
 [  1. 178.]
 [  1. 180.]
 [  1. 183.]]
"""

# Calculating weights of thr fitting line
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.pinv(A), b)

"""
w = [[-33.73541021]
 [  0.55920496]]
 """
print("w = ", w)


# Preparing the fiting line
w_0 = w[0][0]
w_1 = w[1][0]
x0 = np.linspace(145, 185, 2)
y0 = w_0 + w_1*x0

# Drawing the fitting line
plt.plot(x.T, y.T, '+')  # data
plt.plot(x0, y0)         # the fitting line
plt.axis([145, 190, 45, 75])
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
#plt.show()

y1 = w_1*155 + w_0
y2 = w_1*160 + w_0

#print( u'Predict weight of person with height 155 cm: %.2f (kg), real number: 52 (kg)'  %(y1) )
#print( u'Predict weight of person with height 160 cm: %.2f (kg), real number: 56 (kg)'  %(y2) )
#==================================================================

from sklearn import datasets, linear_model
# fit the model by linear Regression
regr = linear_model.LinearRegression(fit_intercept=False)  # fit_intercept = False for calculaitng bias
regr.fit(Xbar, y)

# Compare two results
print("Solution found bu scikit-learn : ", regr.coef_)
print("Solution found by (5): ", w.T)
