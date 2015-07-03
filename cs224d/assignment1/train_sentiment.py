#!/usr/bin/env python
import sys
import os
import numpy as np
import random
import pytest
import string

from argparse import ArgumentParser


from word2vec import normalize_rows
from sentiment import softmax_regression, softmax_wrapper, precision, get_sentence_feature


def get_data(dataset, word_vectors, dtype='train'):

    #tokens = dataset.tokens()
    #num_words = len(tokens)

    if dtype == 'train':
        data = dataset.getTrainSentences()
    elif dtype == 'dev':
        data = dataset.getTrainSentences()
    elif dtype = 'test':
        data = dataset.getTestSentences()
    else:
        raise ValueError('Wrong dataset type requested')

    num_train = len(data)
    train_features = np.zeros((num_train, dim_vectors))   # N x D
    train_labels = np.zeros((num_train,), dtype=np.int32) # N x 1 

    for i in xrange(num_train):
        words, train_labels[i] = data[i]
        train_features[i, :] = get_sentence_feature(tokens, word_vectors, words)

    return train_features, train_labels 

def read_vectors(infile):

    _, word_vectors0, _ = load_saved_params(infile)
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])
    #dim_vectors = word_vectors.shape[1]
    return word_vectors


if __name__ == '__main__':

    random.seed(3141)
    np.random.seed(59265)

    argparser = ArgumentParser()
    argparser.add_argument('--reg', dest='regularization', default=0.0, help='regularization constant')
    argparser.add_argument('--vectors', dest='vectors', default=None, help='word vectors file')

    args = agparser.parse_args()

    # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best
    regularization = args.regularization   

    word_vectors = read_vectors(args.vectors)
    dim_vectors = word_vectors.shape[1]

    dataset = StanfordSentiment()
    train_features, train_labels = get_data(dataset, word_vectors, dtype='train')

    weights = np.random.randn(dim_vectors, 5)  # D x NUM_LABELS array
    # We will do batch optimization
    params = AttrDict({
        'sgd' : {'batch_size': 50, 'step': 3.0, 'iterations': 10000, 'tolerance': 1e-48},
        'dataset' : {}
    })

    print "Starting SGD..."
    weights = sgd(lambda weights: softmax_wrapper(train_features, train_labels, weights, regularization),
        weights, params, postprocessing=None, use_saved=True, print_every=100, save_params_every=1000)


    print "Testing on dev dataset"
    dev_features, dev_labels = get_data(dataset, word_vectors, dtype='dev')
       
    _, _, pred = softmax_regression(dev_features, dev_labels, weights)
    print "Dev precision (%%): %f" % precision(dev_labels, pred)

    test_features, test_labels = get_data(dataset, word_vectors, dtype='test')
       
    _, _, pred = softmax_regression(test_features, test_labels, weights)
    print "=== For autograder ===\nTest precision (%%): %f" % precision(test_labels, pred)


# #### Extra Credit
# 
# Train your own classifier for sentiment analysis! We will not provide any starter code for this part, but you can feel free to reuse the code you've written before, or write some new code for this task. Also feel free to refer to the code we provided you with to see how we scaffolded training for you.
# 
# Try to contain all of your code in one code block. You could start by using multiple blocks, then paste code together and remove unnecessary blocks. Report, as the last two lines of the output of your block, the dev set accuracy and test set accuracy you achieved, in the format we used above.
# 
# *Note: no credits will be given for this part if you use the dev or test sets for training, or if you fine-tune your regularization or other hyperparameters on the test set.*

#    _, _, pred = softmax_regression(dev_features, dev_labels, weights)
#    print "=== For autograder ===\num_dev precision (%%): %f" % precision(dev_labels, pred)
#    _, _, pred = softmax_regression(test_features, test_labels, weights)
#    print "Test precision (%%): %f" % precision(test_labels, pred)

