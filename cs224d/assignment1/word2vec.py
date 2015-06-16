#!/usr/bin/python

import sys
import os


import random
import numpy as np

# Interface to the dataset for negative sampling
dataset = type('dummy', (), {})()
def dummySampleTokenIdx():
    return random.randint(0, 4)
def getRandomContext(C):
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in xrange(2*C)]
dataset.sampleTokenIdx = dummySampleTokenIdx
dataset.getRandomContext = getRandomContext



def softmax_cost_and_gradient(predicted, target, output_vectors, dataset=None):
    """ 
    Softmax cost function for word2vec models 
    Input:
        - predicted: numpy ndarray, vector representation for input word
        assuming 1 x D dim dummy_vectors
        - target: index of the target word
        - output_vectors: word representations for all output vectors
        assuming V x D matrix, where V - vocabulary size, D - vector space dim
    Output:
        - cost
        - grad_pred - grad with respect to input word 1 x D vector
        - grad - grad with respect to all other words, V x D matrix 
    """ 
    score = np.dot(output_vectors, predicted.T).T
    s = np.exp(score - score.max()) / np.exp(score - score.max()).sum()
    cost = -np.log(s[target])

    grad_pred = np.dot(s, output_vectors) - output_vectors[target]
    s[target] -= 1
    grad = s.T[:, np.newaxis] * predicted[np.newaxis, :]


    return cost, grad_pred, grad

def neg_sampling_cost_and_gradient(predicted, target, output_vectors, dataset=None, parameters=None):
    """ 
    Negative sampling cost function for word2vec models 
    Input:
        - predicted: vector 1 x D, vector for input word
    Output:
        output_vectors: size V x D array
    """
    parameters = parameters if parameters else {}
    noise_sample_size = parameters.get('noise_sample_size', 10)

    score = numpy.dot(predicted, output_vectors[target])

    # sample noise objects
    if dataset is None:
        indices = np.random.choice(output_vectors.shape[0], size=noise_sample_size)
    else:
        indices = np.asarray([dataset.sample_token_idx() for _ in xrange(noise_sample_size)])
 
    w = output_vectors[indices]
    score_noise = numpy.dot(w, predicted).T # 1 x K vector
    cost = -np.log(sigmoid(score)) - np.log(sigmoid(-score_noise)).sum()

    grad_pred = numpy.dot(sigmoid(score_noise), w) - (1 - sigmoid(score)) * predicted

    grad_out_noise = sigmoid(score_noise).T[:, np.newaxis] * pred[np.newaxis, :]
    grad_out = - (1 - sigmoid(output_vectors[target], pred)) * pred

    # memory inefficient, lots of zeros
    grad = numpy.zeros(output_vectors.shape)
    grad[indices] = grad_out_noise
    grad[target] = grad_out

    return cost, grad_pred, grad

def skipgram(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None):
    """
    Calculate skipgram cost and gradients for one context window
    Inputs:                                                         
        - current_word: a string of the current center word           
        - context_size: integer, context size                                    
        - context_words: list of no more than 2*context_size strings, the context 
                 words                                               
        - tokens: a dictionary that maps words to their indices in    
                 the word vector list 
        - input_vectors: "input" word vectors for all tokens V x D    
        - output_vectors: "output" word vectors for all tokens V x D     
        - cost_grad_func: the cost and gradient function for 
                 a prediction vector given the target word vectors,  
                 could be one of the two cost functions you          
                 implemented above                                   
    Outputs:                                                        
        - cost: the cost function value for the skip-gram model
        - grad: the gradient with respect to the word vectors
    """

    #XXX: what for input context_size here??? it should be already inside context_words

    current_vec = input_vectors[tokens[current_word]]
    total_cost = 0
    grad_in = np.zeros_like(input_vectors)
    grad_out = np.zeros_like(output_vectors)
    for word in context_words:
        cost, grad_pred, grad = cost_grad_func(current_vec, tokens[word], output_vectors, 
                dataset=dataset, parameters=parameters)
        grad_in[tokens[current_word]] += grad_pred
        grad_out += grad
        total_cost += cost

    return total_cost, grad_in, grad_out


def cbow(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None):
    """
    Calculate cbow cost and gradients for one context window
    Inputs:                                                         
        - current_word: a string of the current center word           
        - context_size: integer, context size                                    
        - context_words: list of no more than 2*context_size strings, the context 
                 words                                               
        - tokens: a dictionary that maps words to their indices in    
                 the word vector list 
        - input_vectors: "input" word vectors for all tokens V x D    
        - output_vectors: "output" word vectors for all tokens V x D     
        - cost_grad_func: the cost and gradient function for 
                 a prediction vector given the target word vectors,  
                 could be one of the two cost functions you          
                 implemented above                                   
    Outputs:                                                        
        - cost: the cost function value for the skip-gram model
        - grad: the gradient with respect to the word vectors
    """

    #XXX: what for input context_size here???

    # Here we are calculatinng input vector as average over all input context words
    indices = [tokens[w] for w in context_words]
    current_vec = input_vectors[indices].sum(0)
    current_vec /= len(context_words)

    cost = 0
    grad_in = np.zeros_like(input_vectors)
    cost, grad_pred, grad_out = cost_grad_func(current_vec, tokens[current_word], output_vectors, 
                dataset=dataset, parameters=parameters)

    grad_in[indices] = grad_pred
    return cost, grad_in, grad_out

# In[ ]:

# Implement a function that normalizes each row of a matrix to have unit length
def normalizeRows(x):
    """ Row normalization function """
    
    ### YOUR CODE HERE
    
    ### END YOUR CODE
    
    return x

# Test this function
print "=== For autograder ==="
print normalizeRows(np.array([[3.0,4.0],[1, 2]]))  # the result should be [[0.6, 0.8], [0.4472, 0.8944]]


# In[ ]:

# Gradient check!

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:]
    outputVectors = wordVectors[N/2:,:]
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)
        
        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1
        
        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom
        
    return cost, grad

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


# In[ ]:

# Now, implement SGD

# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000

import glob
import os.path as op
import cPickle as pickle

def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(op.splitext(op.basename(f))[0].split("_")[2])
        if (iter > st):
            st = iter
            
    if st > 0:
        with open("saved_params_%d.npy" % st, "r") as f:
            params = pickle.load(f)
            state = pickle.load(f)
        return st, params, state
    else:
        return st, None, None
    
def save_params(iter, params):
    with open("saved_params_%d.npy" % iter, "w") as f:
        pickle.dump(params, f)
        pickle.dump(random.getstate(), f)

def sgd(f, x0, step, iterations, postprocessing = None, useSaved = False, PRINT_EVERY=10):
    """ Stochastic Gradient Descent """
    ###################################################################
    # Implement the stochastic gradient descent method in this        #
    # function.                                                       #
    # Inputs:                                                         #
    #   - f: the function to optimize, it should take a single        #
    #        argument and yield two outputs, a cost and the gradient  #
    #        with respect to the arguments                            #
    #   - x0: the initial point to start SGD from                     #
    #   - step: the step size for SGD                                 #
    #   - iterations: total iterations to run SGD for                 #
    #   - postprocessing: postprocessing function for the parameters  #
    #        if necessary. In the case of word2vec we will need to    #
    #        normalize the word vectors to have unit length.          #
    #   - PRINT_EVERY: specifies every how many iterations to output  #
    # Output:                                                         #
    #   - x: the parameter value after SGD finishes                   #
    ###################################################################
    
    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    
    if useSaved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= 0.5 ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    
    for iter in xrange(start_iter + 1, iterations + 1):
        ### YOUR CODE HERE
        ### Don't forget to apply the postprocessing after every iteration!
        ### You might want to print the progress every few iterations.
        
        ### END YOUR CODE
        
        if iter % SAVE_PARAMS_EVERY == 0 and useSaved:
            save_params(iter, x)
            
        if iter % ANNEAL_EVERY == 0:
            step *= 0.5
    
    return x


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



