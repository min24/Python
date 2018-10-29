# This is an exmple of kmeans
"""
Task:
To check level effectivity of a algorithm, we do an example.
Firstly, we choose center for each cluster and create data for each cluster
by picking sample according to standard distribution is expected to center
of that cluster and covariance matrix is unit matrix
"""


"""
We need to declare neccessary libraries. 
We need numpy and matplotlib for computatiing matrix and display data.
We need one more library scipy.spatial.distance to computate distance 
between pairs point in two set of points efectively
"""


import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist
np.random.seed(11)


"""
Then, we create dât by taking points according to standard distribution
is expected to such points (2, 2), (8, 3), (3, 6),
covariance matrixs are the same and are unit matrix.
Each cluster has 500 points
(Note: Each point is a row of matrix)
"""

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T

"""
Display data on plot:
We need a function kmeans_display to show data. Then, show data under original labels.
"""
def kmeans_display(X, label):
	K = np.amax(label) + 1
	X0 = X[label == 0, :]
	X1 = X[label == 1, :]
	X2 = X[label == 2, :]

	plt.plot(X0[:, 0], X0[:, 1], 'b^', markersize = 4, alpha = .8)
	plt.plot(X1[:, 0], X0[:, 1], 'go', markersize = 4, alpha = .8)
	plt.plot(X2[:, 0], X0[:, 1], 'rs', markersize = 4, alpha = .8)

	plt.axis('equal')
	plt.plot()
	plt.show()

kmeans_display(X, original_label)
