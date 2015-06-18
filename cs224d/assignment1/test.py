#!/usr/bin/python
import os
import sys
import numpy as np
import random

from cs224d.data_utils import StanfordSentiment
from wordvec import back_prop1, back_prop2, gradcheck_naive
from word2vec import normalize_rows, word2vec_sgd_wrapper, sgd, cbow, skipgram, normalize_rows

from attrdict import AttrDict


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

    print "=== normalize rows ==="
    print normalize_rows(np.array([[3.0, 4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]


    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)
    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext



    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, neg_sampling_cost_and_gradient), dummy_vectors)

    print "\n=== For autograder ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], neg_sampling_cost_and_gradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], neg_sampling_cost_and_gradient)







    # **Show time! Now we are going to load some real data and train word vectors with everything you just implemented!**
    # 
    # We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task.
    # Load some data and initialize word vectors

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    num_words = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dim_vectors = 10

    # Context size
    context_size = 5

    print "Training word vectors"

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)
    word_vectors = np.concatenate((
        (np.random.rand(num_words, dim_vectors) - 0.5) / dim_vectors, 
        np.zeros((num_words, dimVectors))), 
        axis=0
    )
    word_vectors0 = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, neg_sampling_cost_and_gradient), 
        word_vectors, 0.3, 40000, posprocessing=normalize_rows, use_saved=True, print_every=10, tolerance=1e-8)

    # sanity check: cost at convergence should be around or below 10

    # sum the input and output word vectors
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])

    print "\n=== For autograder ==="
    check_words = ["the", "a", "an", "movie", "ordinary", "but", "and"]
    checkIdx = [tokens[word] for word in check_words]
    checkVecs = word_vectors[checkIdx, :]
    print checkVecs


    # In[ ]:

    # Visualize the word vectors you trained

    _, word_vectors0, _ = load_saved_params()
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])
    visualize_words = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"]
    visualizeIdx = [tokens[word] for word in visualize_words]
    visualizeVecs = word_vectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2]) 

    for i in xrange(len(visualize_words)):
        plt.text(coord[i,0], coord[i,1], visualize_words[i], bbox=dict(facecolor='green', alpha=0.1))
        
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))



