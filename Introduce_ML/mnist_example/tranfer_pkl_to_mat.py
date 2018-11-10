# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a temporary script file.
"""

import pickle
import scipy.io
name = 'pickle_mnist'  #probe without 1
p = open('./MNIST/%s.pkl'%name,'rb')
data = pickle.load(p)
dict = {}
dict['data'] = data
scipy.io.savemat(name,dict)