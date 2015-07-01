#!/usr/bin/env python
import sys
import os
import numpy as np
import random
import pytest
import string


from assignment1.word2vec import normalize_rows
from assignment1.sentiment import softmax_regression, softmax_wrapper, precision, get_sentence_feature
from .conftest import assert_close, empirical_grad

random.seed(314159)
np.random.seed(265)


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
    def get_random_train_sentence(self):
        size = random.randint(1, 8)
        words = [w for w in self.data[self.sample_word_pos()]]
        return words, random.randint(0, 4)


@pytest.fixture(params=[(2, 2), (5, 10), (10, 50)])
def dataset(request):
    return DummyDataset(random.sample(string.letters, request.param[0]), request.param[1])

@pytest.fixture
def dummy_vectors(dataset):
    
    dim = 10
    weights = 0.1 * np.random.randn(dim, 5)
    features = np.zeros((dataset.dim, dim))
    labels = np.zeros((dataset.dim,), dtype=np.int32)    
    word_vectors = normalize_rows(np.random.randn(dataset.dim, dim))

    for i in xrange(dataset.dim):
        words, labels[i] = dataset.get_random_train_sentence()
        features[i, :] = get_sentence_feature(dataset.tokens, word_vectors, words)

    return weights, features, labels

def test_softmax_regression(dummy_vectors):

    weights, features, labels = dummy_vectors
    rndstate = random.getstate()
    random.setstate(rndstate)

    ll = 1.0
    cost, grad = softmax_regression(features, labels, weights, regularization=ll, nopredictions=True)
    random.setstate(rndstate)
    grad_func = lambda w: softmax_regression(features, labels, w, regularization=ll, nopredictions=True)
    emp_grad = empirical_grad(grad_func, weights)
    assert_close(grad, emp_grad)
 


