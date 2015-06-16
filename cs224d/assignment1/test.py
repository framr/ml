#!/usr/bin/python
import os
import sys
import numpy as np
import random

from wordvec import back_prop1, back_prop2, gradcheck_naive
from word2vec import normalize_rows

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

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    print "\n==== Gradient check for CBOW      ===="
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)

    print "\n=== For autograder ==="
    print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)
    print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:])
    print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], negSamplingCostAndGradient)







    # **Show time! Now we are going to load some real data and train word vectors with everything you just implemented!**
    # 
    # We are going to use the Stanford Sentiment Treebank (SST) dataset to train word vectors, and later apply them to a simple sentiment analysis task.

    # In[ ]:

    # Load some data and initialize word vectors

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    # We are going to train 10-dimensional vectors for this assignment
    dimVectors = 10

    # Context size
    C = 5


    # In[ ]:

    # Train word vectors (this could take a while!)

    # Reset the random seed to make sure that everyone gets the same results
    random.seed(31415)
    np.random.seed(9265)
    wordVectors = np.concatenate(((np.random.rand(nWords, dimVectors) - .5) / dimVectors, 
                                  np.zeros((nWords, dimVectors))), axis=0)
    wordVectors0 = sgd(lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negSamplingCostAndGradient), 
                       wordVectors, 0.3, 40000, None, True, PRINT_EVERY=10)
    # sanity check: cost at convergence should be around or below 10

    # sum the input and output word vectors
    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])

    print "\n=== For autograder ==="
    checkWords = ["the", "a", "an", "movie", "ordinary", "but", "and"]
    checkIdx = [tokens[word] for word in checkWords]
    checkVecs = wordVectors[checkIdx, :]
    print checkVecs


    # In[ ]:

    # Visualize the word vectors you trained

    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
    visualizeWords = ["the", "a", "an", ",", ".", "?", "!", "``", "''", "--", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying"]
    visualizeIdx = [tokens[word] for word in visualizeWords]
    visualizeVecs = wordVectors[visualizeIdx, :]
    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2]) 

    for i in xrange(len(visualizeWords)):
        plt.text(coord[i,0], coord[i,1], visualizeWords[i], bbox=dict(facecolor='green', alpha=0.1))
        
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))



