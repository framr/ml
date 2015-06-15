#!/usr/bin/env python

import random
import numpy as np
from itertools import izip


def softmax(x):
    y = x
    if len(x.shape) == 1:
        y = x[np.newaxis, :]
    
    renormed = np.exp(y - y.max(1)[:, np.newaxis])
    result = renormed / renormed.sum(1)[:, np.newaxis]   
    return result

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    res = f * (1 - f)   
    return res


def gradcheck_naive(f, x, step=1e-4, tolerance=1e-5):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = step


    print "grad check cost", fx
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        #print ix, x
        x[ix] += 0.5 * h
        random.setstate(rndstate)
        f2, _ = f(x)
        x[ix] -= h
        random.setstate(rndstate)
        f1, _ = f(x)
        numgrad = (f2 - f1) / h
        x[ix] += 0.5 * h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > tolerance:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            return False

        it.iternext() # Step to next dimension

    return True


def back_prop1(X, Y, params, dim):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    X - array of size N (number of data points in a chunk) x dim0 (input vec dimension)
    Y - array of size N x dim2 (output layer size)
    dim0 - input vec (not taking into account bias)
    dim1 - hidden layer size (not taking into account bias)
    dim2 - output layer size
    """
    ### Unpack network parameters
    t = 0
    W1 = np.reshape(params[t:t+dim[0]*dim[1]], (dim[0], dim[1]))
    t += dim[0]*dim[1]
    b1 = np.reshape(params[t:t+dim[1]], (1, dim[1]))
    t += dim[1]
    W2 = np.reshape(params[t:t+dim[1]*dim[2]], (dim[1], dim[2]))
    t += dim[1]*dim[2]
    b2 = np.reshape(params[t:t+dim[2]], (1, dim[2]))
    
    z = np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2
    s = softmax(z)
    cost = -np.sum(np.log(s[Y.nonzero()]))

    gradW1 = 0
    gradb1 = 0
    gradW2 = 0
    gradb2 = 0
    for x_vec, y_vec in izip(X, Y):
        h = sigmoid(np.dot(x_vec, W1) + b1)
        z = np.dot(h, W2) + b2
        s = softmax(z)
        ds = s - y_vec

        gradW2 += np.outer(h, ds)
        gradb2 += ds
        
        h_grad = sigmoid_grad(h)
        r = np.dot(h_grad.T * W2, ds.T)
        gradW1 += np.outer(x_vec, r)
        gradb1 += r

    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad


def back_prop2(X, Y, params, dim):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    X - array of size N (number of data points in a chunk) x dim0 (input vec dimension)
    Y - array of size N x dim2 (output layer size)
    dim0 - input vec (not taking into account bias)
    dim1 - hidden layer size (not taking into account bias)
    dim2 - output layer size
    """
    ### Unpack network parameters
    t = 0
    W1 = np.reshape(params[t:t+dim[0]*dim[1]], (dim[0], dim[1]))
    t += dim[0]*dim[1]
    b1 = np.reshape(params[t:t+dim[1]], (1, dim[1]))
    t += dim[1]
    W2 = np.reshape(params[t:t+dim[1]*dim[2]], (dim[1], dim[2]))
    t += dim[1]*dim[2]
    b2 = np.reshape(params[t:t+dim[2]], (1, dim[2]))
    
    z = np.dot(sigmoid(np.dot(X, W1) + b1), W2) + b2
    s = softmax(z)
    cost = -np.sum(np.log(s[Y.nonzero()]))

    gradW1 = 0
    gradb1 = 0
    gradW2 = 0
    gradb2 = 0
    for x_vec, y_vec in np.nditer([X, Y], flags=['external_loop'], op_axes=[[1], [1]]):

        print x_vec.shape
        print y_vec.shape
        h = sigmoid(np.dot(x_vec, W1) + b1)
        z = np.dot(h, W2) + b2
        s = softmax(z)
        ds = s - y_vec

        gradW2 += np.outer(h, ds)
        gradb2 += ds
        
        h_grad = sigmoid_grad(h)
        r = np.dot(h_grad.T * W2, ds.T)
        gradW1 += np.outer(x_vec, r)
        gradb1 += r

    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad







