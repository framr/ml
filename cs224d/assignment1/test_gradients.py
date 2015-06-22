#!/usr/bin/env python

import os
import sys
import numpy as np
import random
import pytest
import string

from cs224d.data_utils import StanfordSentiment
from wordvec import back_prop1, back_prop2
from word2vec import normalize_rows, word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows, neg_sampling_cost_and_gradient, softmax_cost_and_gradient, gradcheck_naive

from attrdict import AttrDict


def assert_close(first, second, tolerance=1e-4, norm=1.0):
    diff = (np.abs(first - second) / np.maximum(norm, first, second)).max()
    violated = diff > tolerance
    if violated:
        print "input data: checking \n%s vs \n%s" % (first, second)
        print "gradient violated at indices %s" % violated
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




if __name__ == '__main__':

    random.seed(31415)
    np.random.seed(9265)
 
    @pytest.fixture(scope='module')
    def gradcheck_data():
        return (lambda x: (np.sum(x ** 2), x * 2), 
            [np.array(123.456), np.random.randn(3, ), np.randn((4, 5))]
        )
    def test_empirical_grad(gradcheck_data):
        func, points = gradcheck_data
        for p in points:
            f, grad = func(p)
            assert_close(empirical_grad(func, p), grad)
     

    @pytest.fixture
    def nn(scope='module', params=[1, 10, 20, 100]):

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
        f, grad = back_prop1(nn.data, nn.labels, nn.params, dd.dimensions)
        emp_grad = empirical_grad(lambda params: back_prop1(data, labels, params, dimensions), params)
        assert_close(grad, emp_grad)

    def test_normalize_rows():
        assert normalize_rows(np.array([[3.0, 4.0],[1, 2]])), [[0.6, 0.8], [0.4472, 0.8944]]

    
    @pytest.fixture(params=[1, 5, 10])
    def tokens():
        return random.sample(string.letters, param)

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
    def parameters():
        return AttrDict({'context_size' : 5,'sgd' : {'batch_size': 50}, 'dataset' : {}})

    @pytest.fixture:
    def vectors():
        return normalize_rows(np.random.randn(10, 3))

    @pytest.fixture:
    def input_vectors(vectors):
        return vectors[0]
    @pytest.fixture:
    def output_vectors(vectors):
        return vectors[1:]

    @pytest.fixture(params=[softmax_cost_and_gradient, neg_sampling_cost_and_gradient]) 
    def test_cost_and_grad_func_inputvec(input_vectors, output_vectors, parameters):
        grad_func = lambda w: param(w, )


    print "==== Gradient check for soft_max_cost_and_gradient ===="
    def g_func_wrapper1(f, *params, **kws):
        cost, grad_pred, grad = f(*params, **kws)
        return cost, grad_pred
    def g_func_wrapper2(f, *params, **kws):
        cost, grad_pred, grad = f(*params, **kws)
        return cost, grad

    gradcheck_naive(lambda vec: g_func_wrapper1(softmax_cost_and_gradient, vec, 0, dummy_vectors[1:], dataset, parameters=parameters), 
        dummy_vectors[0])
    gradcheck_naive(lambda vec: g_func_wrapper2(softmax_cost_and_gradient, dummy_vectors[0], 0, vec, dataset, parameters=parameters), 
        dummy_vectors[1:])

    print "==== Gradient check for neg_sampling_max_cost_and_gradient ===="
    print "test 1"
    gradcheck_naive(lambda vec: g_func_wrapper1(neg_sampling_cost_and_gradient, vec, 0, dummy_vectors[1:], dataset, parameters=parameters), 
        dummy_vectors[0], verbose=False)
    print "test 2"
    gradcheck_naive(lambda vec: g_func_wrapper2(neg_sampling_cost_and_gradient, dummy_vectors[0], 0, vec, dataset, parameters=parameters), 
        dummy_vectors[1:])

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



