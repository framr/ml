#!/usr/bin/python
import os
import sys
import numpy as np
import random

from wordvec import back_prop1, back_prop2, gradcheck_naive


if __name__ == '__main__':

     # Sanity check for the gradient checker
    quad = lambda x: (np.sum(x ** 2), x * 2)

    gradcheck_naive(quad, np.array(123.456))      # scalar test
    gradcheck_naive(quad, np.random.randn(3,))    # 1-D test
    gradcheck_naive(quad, np.random.randn(4,5))   # 2-D test


    # Set up fake data and parameters for the neural network
    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], )
    print "Dimensionality of parameter vector", params.shape

    # Perform gradcheck on your neural network
    print "=== Neural network gradient check 1==="
    check_res = gradcheck_naive(lambda params: back_prop1(data, labels, params, dimensions), params)

    print "=== Neural network gradient check 1===" 
    check_res = gradcheck_naive(lambda params: back_prop2(data, labels, params, dimensions), params)
  
