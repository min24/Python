import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from mnist.loader import MNIST
from display_network import *
import imp

mndata = MNIST('./MNIST/') # path to folder MNIST
mndata.load_testing()
X = mndata.test_images

kmeans = KMeans(n_clusters=K).fit(X)
pred_label = kmeans.predict(X)
