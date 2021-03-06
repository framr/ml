#!/usr/bin/env python
import sys
import os
import numpy as np
import random
from argparse import ArgumentParser
from attrdict import AttrDict

from cs224d.data_utils import StanfordSentiment
from word2vec import load_saved_params, sgd
from sentiment import softmax_regression, softmax_wrapper, precision, get_sentence_feature

def save_data(words, labels, pred, filename):

    with open(filename, 'w') as outfile:
        outfile.write("sentence\tlabels\tpred\n")
        for i in range(len(words)):
            outfile.write("%s\t%s\t%s\n" % (' '.join(words[i]), labels[i], pred[i]))
     
def get_data(dataset, word_vectors, dtype='train'):

    tokens = dataset.tokens()
    #num_words = len(tokens)

    if dtype == 'train':
        data = dataset.getTrainSentences()
    elif dtype == 'dev':
        data = dataset.getDevSentences()
    elif dtype == 'test':
        data = dataset.getTestSentences()
    else:
        raise ValueError('Wrong dataset type requested')

    num_train = len(data)
    features = np.zeros((num_train, dim_vectors))   # N x D
    labels = np.zeros((num_train,), dtype=np.int32) # N x 1 

    words = []
    for i in xrange(num_train):
        sent_words, labels[i] = data[i]
        words.append(sent_words)
        features[i, :] = get_sentence_feature(tokens, word_vectors, sent_words)

    return features, labels, words

def read_vectors(infile):

    _, word_vectors0, _ = load_saved_params(infile)
    num_words = word_vectors0.shape[0] / 2
    print "loaded vectors for %d words" %  num_words
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])
    #dim_vectors = word_vectors.shape[1]
    return word_vectors


if __name__ == '__main__':

    random.seed(3141)
    np.random.seed(59265)

    argparser = ArgumentParser()
    argparser.add_argument('--reg', dest='regularization', default=0.0, help='regularization constant', type=float)
    argparser.add_argument('-i', dest='iter', default=10000, help='iterations', type=int)
    argparser.add_argument('--vectors', dest='vectors', default=None, help='word vectors file')

    args = argparser.parse_args()
   
    # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best
    regularization = args.regularization   
    iterations = args.iter

    word_vectors = read_vectors(args.vectors)
    dim_vectors = word_vectors.shape[1]

    dataset = StanfordSentiment()
    train_features, train_labels, words = get_data(dataset, word_vectors, dtype='train')

    weights = np.random.randn(dim_vectors, 5)  # D x NUM_LABELS array
    # We will do batch optimization
    params = AttrDict({
        'sgd' : {'batch_size': 50, 'step': 3.0, 'iterations': iterations, 'tolerance': 0,
            'anneal_every': 10000, 'anneal_factor': 0.5},
        'dataset' : {}
    })

    print "Starting SGD..."
 
    weights = sgd(lambda weights: softmax_wrapper(train_features, train_labels, weights, regularization),
        weights, params, postprocessing=None, use_saved=False, print_every=500, save_params_every=1000)

    _, _, pred = softmax_regression(train_features, train_labels, weights)
    print "Train precision (%%): %f" % precision(train_labels, pred)
    save_data(words, train_labels, pred, 'data_train.txt')
     
    print "Testing on dev dataset"
    dev_features, dev_labels, dev_words = get_data(dataset, word_vectors, dtype='dev')

    print dev_features.shape, weights.shape
    _, _, pred = softmax_regression(dev_features, dev_labels, weights)
    print "Dev precision (%%): %f" % precision(dev_labels, pred)
    save_data(dev_words, dev_labels, pred, 'data_dev.txt')


    test_features, test_labels, test_words = get_data(dataset, word_vectors, dtype='test')
    _, _, pred = softmax_regression(test_features, test_labels, weights)
    print "=== For autograder ===\nTest precision (%%): %f" % precision(test_labels, pred)
    save_data(test_words, test_labels, pred, 'data_test.txt')



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

