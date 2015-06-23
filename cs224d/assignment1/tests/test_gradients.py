#!/usr/bin/env python

import os
import sys
import numpy as np
import random
import pytest
import string

print sys.path

from assignment1.cs224d.data_utils import StanfordSentiment
from assignment1.wordvec import back_prop1, back_prop2
from assignment1.word2vec import word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows, neg_sampling_cost_and_gradient, softmax_cost_and_gradient, gradcheck_naive

from attrdict import AttrDict

random.seed(31415)
np.random.seed(9265)


def assert_close(first, second, tolerance=1e-4, norm=1.0):
    print first, second
    if type(first) is np.ndarray and len(first.shape) > 0:
        diff = np.abs(first - second) / np.maximum(norm, first, second)
        violated = diff > tolerance
        if violated.any():
            print "input data: checking \n%s vs \n%s" % (first, second)
            print "diff\n", diff
            #print "gradient violated at indices %s" % violated
        assert not violated.any()

    else:
        violated = abs(first - second) / max(norm, first, second) > tolerance
        print "input data: checking \n%s vs \n%s" % (first, second)
        assert not violated  

      
def empirical_grad(f, x, step=1e-4, verbose=False):

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, _ = f(x) # Evaluate function value at original point

    numgrad = np.zeros_like(x)
    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        if verbose:
            print "Gradient check at dimension %s" % str(ix)
        
        x[ix] += 0.5 * step
        random.setstate(rndstate)
        f2, _ = f(x)
        x[ix] -= step
        random.setstate(rndstate)
        f1, _ = f(x)
        numgrad[ix] = (f2 - f1) / step
        x[ix] += 0.5 * step
        it.iternext() # Step to next dimension
    return numgrad


@pytest.fixture(scope='module')
def gradcheck_data():
    return (lambda x: (np.sum(x ** 2), x * 2), 
        [np.array(123.456), np.random.randn(3, ), np.random.randn(4, 5)]
    )
def test_empirical_grad(gradcheck_data):
    func, points = gradcheck_data
    for p in points:
        f, grad = func(p)
        assert_close(empirical_grad(func, p), grad)
 

@pytest.fixture(scope='module', params=[1, 10, 20, 100])
def nn(request):

    N = request.param # number of data samples
    dimensions = [10, 5, 10] # input, hidden, output layer dimensions
    data = np.random.randn(N, dimensions[0])
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0, dimensions[2]-1)] = 1

    w = np.random.randn((dimensions[0] + 1) * dimensions[1] + (dimensions[1] + 1) * dimensions[2], ) 
    par = AttrDict(
        {'N': N, 'dimensions': dimensions, 'data': data, 'labels': labels, 'parameters': w}
    )
    return par

def test_neuralnet_grad(nn):
    f, grad = back_prop1(nn.data, nn.labels, nn.parameters, nn.dimensions)
    emp_grad = empirical_grad(lambda params: back_prop1(nn.data, nn.labels, params, nn.dimensions), nn.parameters)
    assert_close(grad, emp_grad)

def test_normalize_rows():
    first = normalize_rows(np.array([[3.0, 4.0], [1.0, 2.0]]))
    second = np.array([[0.6, 0.8], [0.4472, 0.8944]])

    print first
    print second
    assert_close(
        first,
        second
    )


@pytest.fixture(params=[1, 5, 10])
def tokens(request):
    """
    Output: array of tokens
    """
    return random.sample(string.letters, request.param)

class DummyDataset(object):
    def __init__(self, tokens):
        self.tokens = tokens
    def sample_token_idx(self):
        return random.randint(0, len(self.tokens) - 1)
    def get_context(size, self):
        center = tokens[self.sample_token_idx()]
        context = [tokens[self.sample_token_idx()] for i in xrange(2 * size)]
        return center, context


@pytest.fixture(scope='module')
def model_parameters():
    return AttrDict({'context_size' : 5,'sgd' : {'batch_size': 50}, 'dataset' : {}})

@pytest.fixture(scope='module')
def vectors():
    return normalize_rows(np.random.randn(10, 3))

@pytest.fixture(scope='module')
def input_vectors(vectors):
    return vectors[0]
@pytest.fixture(scope='module')
def output_vectors(vectors):
    return vectors[1:]

@pytest.fixture(params=[softmax_cost_and_gradient, neg_sampling_cost_and_gradient]) 
def cost_grad_func(request):
    return request.param

def test_cost_and_grad_func_inputvec(cost_grad_func, input_vectors, output_vectors, tokens, model_parameters):
    dataset = DummyDataset(tokens)
    target = 0
    cost, grad_input, grad_output = cost_grad_func(input_vectors, target, output_vectors, dataset,
        parameters=model_parameters)

    grad_func_input = lambda w: cost_grad_func(w, target, output_vectors, dataset,
        parameters=model_parameters)
    grad_func_output = lambda w: cost_grad_func(input_vectors, target, w, dataset,
        parameters=model_parameters)

    empirical_grad_in = empirical_grad(grad_func_input, input_vectors)
    empirical_grad_out = empirical_grad(grad_func_output, output_vectors)




"""  
parameters = AttrDict(
{
'context_size' : 1,
'sgd' : {'batch_size': 1},
'dataset' : {}
}
)
random.seed(31415)
np.random.seed(9265)
dummy_vectors = normalize_rows(np.random.randn(10,3))
dummy_tokens = dict([("a",0), ("b",1), ("c",2), ("d",3), ("e",4)])
"""


