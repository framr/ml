#!/usr/bin/env python

import sys
import os
import numpy as np
import random
import pytest
import string


from word2vec import normalize_rows
from sentiment import softmax_regression, softmax_wrapper, precision, get_sentence_feature


if __name__ == '__main__':

    regularization = 0.0 # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best

    _, word_vectors0, _ = load_saved_params()
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])
 
    dim_vectors = word_vectors.shape[1]

    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dim_vectors, 5)  # D x NUM_LABELS array

    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    num_words = len(tokens)


    trainset = dataset.getTrainSentences()
    num_train = len(trainset)
    train_features = np.zeros((num_train, dim_vectors))
    train_labels = np.zeros((num_train,), dtype=np.int32)

    for i in xrange(nTrain):
        words, train_labels[i] = trainset[i]
        train_features[i, :] = getSentenceFeature(tokens, word_vectors, words)

    # We will do batch optimization
    params = AttrDict({'sgd' : {'batch_size': 50}, 'dataset' : {}})
    params['sgd']['step'] = 3.0
    params['sgd']['iterations'] = 10000
    params['sgd']['tolerance'] = 1e-48

    print "Starting SGD..."
    weights = sgd(lambda weights: softmax_wrapper(train_features, train_labels, weights, regularization),
        weights, params, postprocessing=None, use_saved=True, print_every=100, save_params_every=1000)


    # Prepare dev set features
    devset = dataset.getDevSentences()
    num_dev = len(devset)
    dev_features = np.zeros((num_dev, dim_vectors))
    dev_labels = np.zeros((num_dev,), dtype=np.int32)

    for i in xrange(num_dev):
        words, dev_labels[i] = devset[i]
        dev_features[i, :] = getSentenceFeature(tokens, word_vectors, words)
        
    _, _, pred = softmax_regression(dev_features, dev_labels, weights)
    print "Dev precision (%%): %f" % precision(dev_labels, pred)


    # Write down the best regularization and accuracy you found
    # sanity check: your accuracy should be around or above 30%

    ### YOUR CODE HERE

    BEST_REGULARIZATION = 1
    BEST_ACCURACY = 0.0

    ### END YOUR CODE

    print "=== For autograder ===\n%g\t%g" % (BEST_REGULARIZATION, BEST_ACCURACY)


    # In[ ]:

    # Test your findings on the test set

    testset = dataset.getTestSentences()
    nTest = len(testset)
    test_features = np.zeros((num_test, dim_vectors))
    test_labels = np.zeros((num_test,), dtype=np.int32)

    for i in xrange(num_test):
        words, test_labels[i] = testset[i]
        test_features[i, :] = getSentenceFeature(tokens, word_vectors, words)
        
    _, _, pred = softmax_regression(test_features, test_labels, weights)
    print "=== For autograder ===\nTest precision (%%): %f" % precision(test_labels, pred)


# #### Extra Credit
# 
# Train your own classifier for sentiment analysis! We will not provide any starter code for this part, but you can feel free to reuse the code you've written before, or write some new code for this task. Also feel free to refer to the code we provided you with to see how we scaffolded training for you.
# 
# Try to contain all of your code in one code block. You could start by using multiple blocks, then paste code together and remove unnecessary blocks. Report, as the last two lines of the output of your block, the dev set accuracy and test set accuracy you achieved, in the format we used above.
# 
# *Note: no credits will be given for this part if you use the dev or test sets for training, or if you fine-tune your regularization or other hyperparameters on the test set.*

# In[ ]:

### YOUR CODE HERE

### END YOU CODE


    _, _, pred = softmax_regression(dev_features, dev_labels, weights)
    print "=== For autograder ===\num_dev precision (%%): %f" % precision(dev_labels, pred)
    _, _, pred = softmax_regression(test_features, test_labels, weights)
    print "Test precision (%%): %f" % precision(test_labels, pred)

