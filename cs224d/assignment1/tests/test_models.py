#!/usr/bin/env python
import sys
import os
import numpy as np
import random
import pytest
import string
from attrdict import AttrDict


from assignment1.cs224d.data_utils import StanfordSentiment
from assignment1.wordvec import back_prop1, back_prop2
from assignment1.word2vec import word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows, neg_sampling_cost_and_gradient, softmax_cost_and_gradient, gradcheck_naive

from .conftest import empirical_grad, assert_close


random.seed(31415)
np.random.seed(9265)

"""
def skipgram(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None, verbose=False):
def cbow(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None, verbose=False):
return cost, grad_in, grad_out
"""

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
        self.tokens = dict((w, i) for i, w in enumerate(np.unique(self.data)))
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

@pytest.fixture(params=[5, 25])
def dataset(request, tokens):
    """ 
    Generate random dataset
    Params: dataset size
    """
    return DummyDataset(tokens, request.param)

@pytest.fixture
def model_parameters():
    return AttrDict({'context_size' : 5, 'sgd' : {'batch_size': 50}, 'dataset' : {}, 'noise_sample_size': 5})

@pytest.fixture(params=[3, 10, 20])
def vectors(request, dataset):
    """
    Generate random word vectors
    Params: dimensionality
    """
    output_vec = normalize_rows(np.random.randn(dataset.dim, request.param))
    input_vec = normalize_rows(np.random.randn(dataset.dim, request.param))
    return input_vec, output_vec

@pytest.fixture(params=[softmax_cost_and_gradient, neg_sampling_cost_and_gradient]) 
def cost_grad_func(request):
    return request.param

#@pytest.fixture
#def random_context(dataset):
    
def model_gradients(grad_func, dataset, vectors, model_parameters):
    input_vectors, output_vectors = vectors
    center_word, context_words = dataset.get_context(model_parameters.context_size)

    rndstate = random.getstate()
    random.setstate(rndstate)
    cost, grad_in, grad_out = skipgram(center_word, len(context_words), context_words, dataset.tokens, input_vectors, output_vectors, 
        cost_grad_func=grad_func, dataset=dataset, parameters=model_parameters, verbose=False)

    grad_in_func = lambda w: skipgram(center_word, len(context_words), context_words, dataset.tokens, w, output_vectors, 
        cost_grad_func=grad_func, dataset=dataset, parameters=model_parameters, verbose=False)
    grad_out_func = lambda w: skipgram(center_word, len(context_words), context_words, dataset.tokens, input_vectors, w, 
        cost_grad_func=grad_func, dataset=dataset, parameters=model_parameters, verbose=False)

    random.setstate(rndstate) 
    empirical_grad_in = empirical_grad(grad_in_func, input_vectors)
    random.setstate(rndstate) 
    empirical_grad_out = empirical_grad(grad_out_func, output_vectors)

    assert_close(grad_in, empirical_grad_in)
    assert_close(grad_out, empirical_grad_out)

def test_skipgram_gradients(dataset, vectors, model_parameters):
    model_gradients(softmax_cost_and_gradient, dataset, vectors, model_parameters)
 
def test_negsampling_gradients(dataset, vectors, model_parameters):
    model_gradients(neg_sampling_cost_and_gradient, dataset, vectors, model_parameters)
 


def test_cbow_gradients(dataset, vectors, model_parameters):
    input_vectors, output_vectors = vectors
    center_word, context_words = dataset.get_context(model_parameters.context_size)

    rndstate = random.getstate()
    random.setstate(rndstate)
    cost, grad_in, grad_out = cbow(center_word, len(context_words), context_words, dataset.tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=dataset, parameters=model_parameters, verbose=False)

    grad_in_func = lambda w: cbow(center_word, len(context_words), context_words, dataset.tokens, w, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=dataset, parameters=model_parameters, verbose=False)
    grad_out_func = lambda w: cbow(center_word, len(context_words), context_words, dataset.tokens, input_vectors, w, 
        cost_grad_func=softmax_cost_and_gradient, dataset=dataset, parameters=model_parameters, verbose=False)

    random.setstate(rndstate) 
    empirical_grad_in = empirical_grad(grad_in_func, input_vectors)
    random.setstate(rndstate) 
    empirical_grad_out = empirical_grad(grad_out_func, output_vectors)
    assert_close(grad_in, empirical_grad_in)
    assert_close(grad_out, empirical_grad_out)


    
def test_sgd_wrapper(dataset, vectors, model_parameters):

    vec = np.hstack(vectors)
    rndstate = random.getstate()
    cost, grad = word2vec_sgd_wrapper(skipgram, dataset.tokens, vec, dataset, parameters=model_parameters, verbose=False)

    func = lambda vec: word2vec_sgd_wrapper(skipgram, dataset.tokens, vec, dataset, parameters=model_parameters, verbose=False)
    random.setstate(rndstate)
    emp_grad = empirical_grad(func, vec)
    assert_close(grad, emp_grad)


'''
print "==== Gradient check for skip-gram ===="
print "test 1"
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, parameters=parameters, verbose=False),
    dummy_vectors, verbose=False)
print "test 2"
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, parameters=parameters,
    cost_grad_func=neg_sampling_cost_and_gradient), dummy_vectors)

print "\n==== Gradient check for CBOW      ===="
print "test 1"
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, parameters=parameters),
    dummy_vectors)
print "test 2"
gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, parameters=parameters, 
    cost_grad_func=neg_sampling_cost_and_gradient), dummy_vectors)

print "\n=== For autograder ==="
print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:],
    parameters=parameters)
print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], 
    parameters=parameters, cost_grad_func=neg_sampling_cost_and_gradient)
print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:],
    parameters=parameters)
print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:],
    parameters=parameters, cost_grad_func=neg_sampling_cost_and_gradient)
'''


