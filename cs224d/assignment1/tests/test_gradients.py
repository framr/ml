#!/usr/bin/env python
import os
import sys
import numpy as np
import random
import pytest
import string
from attrdict import AttrDict


from assignment1.wordvec import back_prop1, back_prop2
from assignment1.word2vec import word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows, neg_sampling_cost_and_gradient, softmax_cost_and_gradient, gradcheck_naive


from .conftest import empirical_grad, assert_close


random.seed(31415)
np.random.seed(9265)



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
    assert_close(
        first,
        second
    )


@pytest.fixture(params=[1, 5, 10, 20])
def tokens(request):
    """
    Output: array of tokens - a vocabulary
    Params: size of vocabulary
    """
    return random.sample(string.letters, request.param)

class DummyDataset(object):
    def __init__(self, tokens, size):
        self.data = np.random.choice(tokens, size)
        self.tokens = np.unique(self.data)
        self.dim = len(self.tokens)
    def sample_token_idx(self):
        """ return random word index from dataset """
        return random.randint(0, self.dim - 1)
    def sample_word_pos(self):
        """ return random word index from dataset """
        return random.randint(0, len(self.data) - 1)
    def get_context(self, size):
        """ get random word and context from dataset"""
        center = self.data[self.sample_word_pos()]
        context = [self.data[self.sample_word_pos()] for i in xrange(2 * size)]
        return center, context

@pytest.fixture(params=[10, 100])
def dataset(request, tokens):
    """ 
    Generate random dataset
    Params: dataset size
    """
    return DummyDataset(tokens, request.param)

@pytest.fixture
def model_parameters():
    return AttrDict({'context_size' : 5, 'sgd' : {'batch_size': 50}, 'dataset' : {}})

@pytest.fixture(params=[3, 10, 50])
def vectors(request, dataset):
    """
    Generate random word vectors
    Params: dimensionality
    """
    output_vec = normalize_rows(np.random.randn(dataset.dim, request.param))
    input_vec = normalize_rows(np.random.randn(1, request.param))
    return input_vec, output_vec

@pytest.fixture(params=[softmax_cost_and_gradient, neg_sampling_cost_and_gradient]) 
def cost_grad_func(request):
    return request.param

def test_cost_and_grad_func_inputvec(cost_grad_func, vectors, dataset, model_parameters):
    target = 0
    input_vectors, output_vectors = vectors
    cost, grad_input, grad_output = cost_grad_func(input_vectors[0], target, output_vectors, dataset,
        parameters=model_parameters)

    #XXX
    #WTF???? why it succedes without using saving and restoring random state???
    grad_func_input = lambda w: cost_grad_func(w, target, output_vectors, dataset,
        parameters=model_parameters)
    grad_func_output = lambda w: cost_grad_func(input_vectors[0], target, w, dataset,
        parameters=model_parameters)

    empirical_grad_in = empirical_grad(grad_func_input, input_vectors[0])
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


