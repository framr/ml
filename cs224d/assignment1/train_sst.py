#!/usr/bin/env python
import sys
import random
import numpy as np
from cs224d.data_utils import StanfordSentiment
import matplotlib.pyplot as plt

from attrict import AttrDict


if __name__ == '__main__':

    
    random.seed(314)
    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    params = AttrDict({'
context_size' : 5, 'sgd' : {'batch_size': 50}, 'dataset' : {}, 'noise_sample_size': 5})


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



