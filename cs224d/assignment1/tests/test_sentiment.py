#!/usr/bin/env python
import sys
import os
import numpy as np
import random

from assignment1.cs224d.sentiment import softmax_regression, softmax_wrapper, precision, get_sentence_feature

random.seed(314159)
np.random.seed(265)


class DummyDataset(object):
    def __init__(self, tokens, size):
        self.data = np.random.choice(tokens, size)
        self.tokens = np.unique(self.data)
        self.dim = len(self.tokens)
    def sample_token_idx(self):
        """ return random word index from dataset """
        return random.randint(0, self.dim - 1)
    def sample_word_pos(self):
        """ return random word index from dataset """
        return random.randint(0, len(self.data) - 1)
    def get_context(self, size):
        """ get random word and context from dataset"""
        center = self.data[self.sample_word_pos()]
        context = [self.data[self.sample_word_pos()] for i in xrange(2 * size)]
        return center, context
    def get_random_train_sentence(self):
        


@pytest.fixture(params=[(2, 2), (5, 10), (10, 50)])
def dataset(request):
    return DummyDataset(random.sample(string.letters, request.params[0]), request.params[1])

dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
dummy_features = np.zeros((10, dimVectors))
dummy_labels = np.zeros((10,), dtype=np.int32)    
for i in xrange(10):
    words, dummy_labels[i] = dataset.getRandomTrainSentence()
    dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
print "==== Gradient check for softmax regression ===="
gradcheck_naive(lambda weights: softmaxRegression(dummy_features, dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

print "\n=== For autograder ==="
print softmax_regression(dummy_features, dummy_labels, dummy_weights, 1.0)


# In[ ]:

# Try different regularizations and pick the best!

### YOUR CODE HERE

regularization = 0.0 # try 0.0, 0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01 and pick the best

### END YOUR CODE

random.seed(3141)
np.random.seed(59265)
weights = np.random.randn(dimVectors, 5)

trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)

for i in xrange(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
# We will do batch optimization
weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels, weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)

for i in xrange(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(devFeatures, devLabels, weights)
print "Dev precision (%%): %f" % precision(devLabels, pred)


# In[ ]:

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
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)

for i in xrange(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)
    
_, _, pred = softmaxRegression(testFeatures, testLabels, weights)
print "=== For autograder ===\nTest precision (%%): %f" % precision(testLabels, pred)


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


_, _, pred = softmaxRegression(devFeatures, devLabels, weights)
print "=== For autograder ===\nDev precision (%%): %f" % precision(devLabels, pred)
_, _, pred = softmaxRegression(testFeatures, testLabels, weights)
print "Test precision (%%): %f" % precision(testLabels, pred)

