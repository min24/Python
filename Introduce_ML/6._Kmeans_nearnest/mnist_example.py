# %reset
from past import autotranslate

from mnist import MNIST # "require pip install python-mnist"
# https://pypi.python.org/pypi/python-mnist/

import matplotlib.pyplot as plt 
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time

# you need to download the MNIST dataset first
# at: http://yann.lecun.com/exdb/mnist/


mndata = MNIST('./MNIST/') # path to your MNIST folder
mndata.load_testing()
mndata.load_training()