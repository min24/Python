# Declare libraries
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn import neighbors, datasets

# Load data and show some samples
iris = datasets.load_iris()
iris_x = iris.data 
iris_y = iris.target
print("Number of classes: %d" %len(np.unique(iris_y)))
print("Number of data points: %d" %len(iris_y))

x0 = iris_x[iris_y == 0, :]
print("\nSamples from class 0:\n", x0[:5, :])

x1 = iris_x[iris_y == 1,:]
print('\nSamples from class 1:\n', x1[:5,:])

x2 = iris_x[iris_y == 2,:]
print('\nSamples from class 2:\n', x2[:5,:])

# Separate training and test sets
# 100 points for training set and 50 points for test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=50)

print()
print("Training size: %d" %len(y_train))
print("Test size    : %d" %len(y_test))

# Consider situation K = 1, it means with each point of test data
# we consider only 1 training data nearest and take its label to forecast
# test point
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
# "p" is norm of distance https://machinelearningcoban.com/math/#norm2
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print()
print("Print results for 20 test data points")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])
print()

# Evaluation method
from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# Method "major voting": we don't just see on 1 point nearest,
# we see on 10 points nearest
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))

# Score for neighboring points
# A point which is nearer, needs to be scored larger
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = "distance")
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# Another way to score for neighboring points
def myweight(distances):
	sigma2 = 0.5 # we can change this number
	return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
