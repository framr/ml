import random
import numpy as np

from word2vec import softmax

# ## 4. Sentiment Analysis
# 
# Now, with the word vectors you trained, we are going to perform a simple sentiment analysis.
# 
# For each sentence in the Stanford Sentiment Treebank dataset, we are going to use the average of all the word vectors 
#in that sentence as its feature, and try to predict the sentiment level of the said sentence. 
#The sentiment level of the phrases are represented as real values in the original dataset, here we'll just use five classes:
# 
#     "very negative", "negative", "neutral", "positive", "very positive"
#     
# which are represented by 0 to 4 in the code, respectively.
# 
# For this part, you will learn to train a softmax regressor with SGD, and perform train/dev validation to improve generalization of your regressor.

# In[ ]:

# Now, implement some helper functions

def get_sentence_feature(tokens, word_vectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its word vectors
    Inputs:                                                         
       - tokens: a dictionary that maps words to their indices in    
                 the word vector list                                
       - word_vectors: word vectors for all tokens                    
       - sentence: a list of words in the sentence of interest       
    Output:                                                         
       - sent_vector: feature vector for the sentence                 
    """    
    indices = [tokens[i] for i in sentence]
    sent_vector = np.average(word_vectors[indices], axis=0)
    #sent_vector = np.zeros((word_vectors.shape[1],))
       
    return sent_vector


def softmax_regression(features, labels, weights, regularization=0.0, nopredictions=False):
    """ 
    Softmax Regression
    Implement softmax regression with weight regularization.       
    Inputs:                                                        
      - features: feature vectors, each row is a feature vector     
      - labels: labels corresponding to the feature vectors         
      - weights: weights of the regressor                           
      - regularization: L2 regularization constant                  
    Output:                                                         
      - cost: cost of the regressor                                 
      - grad: gradient of the regressor cost with respect to its   
              weights                                              
      - pred: label predictions of the regressor (you might find    
              np.argmax helpful)                                    
    """

    z = features.dot(weights)
    prob = softmax(z)
    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N 
    cost += 0.5 * regularization * np.sum(weights ** 2)

    y = np.zeros(z.shape)
    y[np.arange(N), labels] = 1.0
    grad = np.dot(features.T, prob - y) / N + regularization * weights
    pred = np.argmax(prob, axis=1) 
       
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def precision(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization=0.0):
    cost, grad, _ = softmax_regression(features, labels, weights, regularization)
    return cost, grad



