# Dump data to file pickle
"""
You may have noticed that it is quite slow to read in the data from the csv files.

We will save the data in binary format with the dump function from the pickle module:
"""

import pickle
with open("./MNIST/pickle_mnist.pkl", "bw") as fh:
    data = (train_imgs, 
            test_imgs, 
            train_labels,
            test_labels,
            train_labels_one_hot,
            test_labels_one_hot)
    pickle.dump(data, fh)


# Load data from file pickle
# We are able now to read in the data by using pickle.load. This is a lot faster than using loadtxt on the csv files:

import pickle
with open("./MNIST/pickle_mnist.pkl", "br") as fh:
    data = pickle.load(fh)
train_imgs = data[0]
test_imgs = data[1]
train_labels = data[2]
test_labels = data[3]
train_labels_one_hot = data[4]
test_labels_one_hot = data[5]
image_size = 28
no_of_different_labels =  10
image_pixels = image_size*image_size
