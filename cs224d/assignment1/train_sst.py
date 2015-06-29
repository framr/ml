#!/usr/bin/env python
import sys
import random
import numpy as np
from cs224d.data_utils import StanfordSentiment
import matplotlib.pyplot as plt
from attrdict import AttrDict

from word2vec import sgd, word2vec_sgd_wrapper, skipgram, cbow, neg_sampling_cost_and_gradient, normalize_rows, load_saved_params

if __name__ == '__main__':

    
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    num_words = len(tokens)

    params = AttrDict({'context_size' : 5, 'sgd' : {'batch_size': 50}, 'dataset' : {}, 'noise_sample_size': 10})
    dim_vectors = 10

    random.seed(31415)
    np.random.seed(9265)
    word_vectors = np.concatenate((
        (np.random.rand(num_words, dim_vectors) - .5) / dim_vectors, 
        np.zeros((num_words, dim_vectors))), 
        axis=0
    )

    params['sgd']['step'] = 0.2
    params['sgd']['iterations'] = 40000
    params['sgd']['tolerance'] = 1e-48
    params['sgd']['anneal_every'] = 20000
    params['sgd']['anneal_factor'] = 0.5

    word_vectors0 = sgd(
        lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, params, neg_sampling_cost_and_gradient), 
        word_vectors, params, postprocessing=normalize_rows, use_saved=True, print_every=100, save_params_every=5000)

    # sanity check: cost at convergence should be around or below 10
    # sum the input and output word vectors
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])

    print "\n=== For autograder ==="
    check_words = ["the", "a", "an", "movie", "ordinary", "but", "and"]
    check_idx = [tokens[word] for word in check_words]
    check_vecs = word_vectors[check_idx, :]
    print check_vecs


    # Visualize the word vectors you trained

    _, word_vectors0, _ = load_saved_params()
    word_vectors = (word_vectors0[:num_words,:] + word_vectors0[num_words:,:])
    visualize_words = ["the", "a", "an", "good", "great", "cool", "brilliant", "wonderful", "well", "amazing", "worth", "sweet", "enjoyable", "boring", "bad", "waste", "dumb", "annoying", "london", "england", "york", "yorker", "winter", "car", "automatically"]
    visualize_idx = [tokens[word] for word in visualize_words]
    visualize_vecs = word_vectors[visualize_idx, :]
    temp = (visualize_vecs - np.mean(visualize_vecs, axis=0))
    covariance = 1.0 / len(visualize_idx) * temp.T.dot(temp)
    U, S, V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2]) 

    print "SVD coordinates"
    for i in xrange(len(visualize_words)):
        plt.text(coord[i,0], coord[i,1], visualize_words[i], bbox=dict(facecolor='green', alpha=0.1))
        
    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.show()

