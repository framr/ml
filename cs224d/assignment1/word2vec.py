#!/usr/bin/env python

import sys
import os
import glob
import cPickle as pickle

import random
import numpy as np



# Save parameters every a few SGD iterations as fail-safe
SAVE_PARAMS_EVERY = 1000


def gradcheck_naive(f, x, step=1e-4, tolerance=1e-5, verbose=False):
    """ 
    Gradient check for a function f 
    - f should be a function that takes a single argument and outputs the cost and its gradients
    - x is the point (numpy array) to check the gradient at
    """ 

    rndstate = random.getstate()
    random.setstate(rndstate)  
    fx, grad = f(x) # Evaluate function value at original point
    h = step

    # Iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index
        if verbose:
            print "Gradient check at dimension %s" % str(ix)
        
        x[ix] += 0.5 * h
        random.setstate(rndstate)
        f2, _ = f(x)
        x[ix] -= h
        random.setstate(rndstate)
        f1, _ = f(x)
        numgrad = (f2 - f1) / h
        x[ix] += 0.5 * h

        # Compare gradients
        reldiff = abs(numgrad - grad[ix]) / max(1, abs(numgrad), abs(grad[ix]))

        if reldiff > tolerance:
            print "Gradient check failed."
            print "First gradient error found at index %s" % str(ix)
            print "Your gradient: %f \t Numerical gradient: %f" % (grad[ix], numgrad)
            raise ValueError

        it.iternext() # Step to next dimension



def softmax(x):
    y = x
    if len(x.shape) == 1:
        y = x[np.newaxis, :]
    
    renormed = np.exp(y - y.max(1)[:, np.newaxis])
    result = renormed / renormed.sum(1)[:, np.newaxis]   
    return result

def sigmoid(x):
    result = 1 / (1 + np.exp(-x))
    return result

def sigmoid_grad(f):
    """ Sigmoid gradient function """
    res = f * (1 - f)   
    return res



def softmax_cost_and_gradient(predicted, target, output_vectors, dataset=None, parameters=None):
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
        - output_vectors: size V x D array
        - 
    """
    parameters = parameters if parameters else {}
    noise_sample_size = parameters.get('noise_sample_size', 5)

    score = np.dot(predicted, output_vectors[target])

    # sample noise objects
    if dataset is None:
        indices = np.random.choice(output_vectors.shape[0], size=noise_sample_size)
    else:
        indices = np.asarray([dataset.sample_token_idx() for _ in xrange(noise_sample_size)])

    indices = np.unique(indices)

    w = output_vectors[indices] # K x D vector
    score_noise = np.dot(w, predicted).T # 1 x K vector
    #print "indices\n", indices
    #print "w\n", w 
    #print "predicted\n", predicted 
    #print "score_noise\n", score_noise
    #print "output vectors:\n", w
    #print "input vector:\n", predicted
    #print "score: %s" % score
    #print "noise score: %s " % score_noise

    cost = -np.log(sigmoid(score)) - np.log(1 - sigmoid(score_noise)).sum()

    grad_pred = np.dot(sigmoid(score_noise), w) - (1 - sigmoid(score)) * output_vectors[target]
    grad_out_noise = sigmoid(score_noise).T[:, np.newaxis] * predicted[np.newaxis, :]

    grad_out = -(1 - sigmoid(np.dot(output_vectors[target], predicted))) * predicted

    # memory inefficient, lots of zeros
    grad = np.zeros(output_vectors.shape)
    grad[indices] = grad_out_noise
    grad[target] += grad_out   # add gradient here since noise sample and target can be the same in principle

    return cost, grad_pred, grad

def skipgram(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None, verbose=False):
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
        if verbose:
            print "skipgram: considering %s in the context of %s" % (current_word, word)
        cost, grad_pred, grad = cost_grad_func(current_vec, tokens[word], output_vectors, 
                dataset=dataset, parameters=parameters)
        grad_in[tokens[current_word]] += grad_pred
        grad_out += grad
        total_cost += cost
        if verbose:
            print "skipgram cost, grad_pred, grad:\n", cost, "\n", grad_pred, "\n", grad



    return total_cost, grad_in, grad_out


def cbow(current_word, context_size, context_words, tokens, input_vectors, output_vectors, 
        cost_grad_func=softmax_cost_and_gradient, dataset=None, parameters=None, verbose=False):
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


# Implement a function that normalizes each row of a matrix to have unit length
def normalize_rows(x):
    """ Row normalization function """   
    return x / np.sqrt((x * x).sum(1))[:, np.newaxis] 


def word2vec_sgd_wrapper(model, tokens, word_vectors, dataset, parameters, cost_grad_func=softmax_cost_and_gradient, verbose=False):

    #XXX batch provides no vectorization currently

    context_size = parameters.context_size
    batchsize = parameters.sgd.batch_size

    total_cost = 0.0
    grad = np.zeros(word_vectors.shape)
    N = word_vectors.shape[0]
    input_vectors = word_vectors[:N/2,:]
    output_vectors = word_vectors[N/2:,:]

    if verbose:
        print 'Wrapper'
    for i in xrange(batchsize):
        C1 = random.randint(1, context_size)
        centerword, context = dataset.get_context(C1, parameters.dataset)
        
        #WTF??
        if model == skipgram:
            denom = 1
        else:
            denom = 1
       
        if verbose:
            print "sgd wrapper: Batch element %d" % i
            print "sgd wrapper: Evaluating model at center word %s, inside context %s" % (centerword, ' '.join(context))
        cost, grad_in, grad_out = model(centerword, C1, context, tokens, input_vectors, output_vectors,
            cost_grad_func, dataset=dataset, parameters=parameters, verbose=verbose)
        total_cost += cost / batchsize / denom
        grad[:N/2, :] += grad_in / batchsize / denom
        grad[N/2:, :] += grad_out / batchsize / denom
        if verbose:
            print "sgd wrapper: input vectors\n", input_vectors
            print "sgd wrapper: output vectors\n", output_vectors
            print "sgd wrapper: Total cost = %s, current cost = %s" % (total_cost, cost)
            print "sgd wrapper: Updated gradient\n", grad 
    return total_cost, grad



def load_saved_params():
    """ A helper function that loads previously saved parameters and resets iteration start """
    st = 0
    for f in glob.glob("saved_params_*.npy"):
        iter = int(os.path.splitext(os.path.basename(f))[0].split("_")[2])
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

def sgd(f, x0, step, iterations, postprocessing=None, use_saved=False, tolerance=1e-5, print_every=10):
    """ 
    Stochastic Gradient Descent                                               
    Inputs:                                                         
        - f: the function to optimize, it should take a single        
            argument and yield two outputs, a cost and the gradient  
            with respect to the arguments                            
        - x0: the initial point to start SGD from                     
        - step: the step size for SGD                               
        - iterations: total iterations to run SGD for                 
        - postprocessing: postprocessing function for the parameters  
            if necessary. In the case of word2vec we will need to    
            normalize the word vectors to have unit length.          
        - PRINT_EVERY: specifies every how many iterations to output  
    Output:                                                         
        - x: the parameter value after SGD finishes                   
    """

    # Anneal learning rate every several iterations
    ANNEAL_EVERY = 20000
    ANNEAL_FACTOR = 0.5

    if use_saved:
        start_iter, oldx, state = load_saved_params()
        if start_iter > 0:
            x0 = oldx;
            step *= ANNEAL_FACTOR ** (start_iter / ANNEAL_EVERY)
            
        if state:
            random.setstate(state)
    else:
        start_iter = 0
    
    x = x0
    
    if not postprocessing:
        postprocessing = lambda x: x
    
    expcost = None
    old_cost = None
    for iter in xrange(start_iter + 1, iterations + 1):
        
        cost, grad = f(x)
        x -= step * grad
        posprocessing(x)
        
        if iter % SAVE_PARAMS_EVERY == 0 and use_saved:
            save_params(iter, x)
            
        if iter % ANNEAL_EVERY == 0:
            step *= ANNEAL_FACTOR
    
        if iter % print_every == 0:
            print "%d  %e" % (iter, cost)
        if old_cost is not None and abs((cost - old_cost) / old_cost) < tolerance:
            print "Relative cost change %e below tolerance threshold, exiting" % (cost - old_cost) / old_cost

        old_cost = cost

    return x



