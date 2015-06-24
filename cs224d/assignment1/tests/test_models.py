#!/usr/bin/env python
import sys
import os
import numpy as np

from assignment1.cs224d.data_utils import StanfordSentiment
from assignment1.wordvec import back_prop1, back_prop2
from assignment1.word2vec import word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows, neg_sampling_cost_and_gradient, softmax_cost_and_gradient, gradcheck_naive

from .conftest import empirical_grad, assert_close


random.seed(31415)
np.random.seed(9265)

"""
Testing here skipgram, cbow
"""


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



