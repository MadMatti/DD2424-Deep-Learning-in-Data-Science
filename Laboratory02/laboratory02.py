import pickle 
import numpy as np
import matplotlib.pyplot as plt


data_dir = 'datasets/'

N = 10000
d = 3072
K = 10

def unpickle(file):
    with open(data_dir + file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_labels(labels):
    one_hot = np.zeros((N, K))
    for i in range(len(labels)):
        one_hot[i, labels[i]] = 1
    return one_hot

def load_batch(file):
    batch = unpickle(file)
    data = batch[b'data'].T / 255 # normalization
    labels = batch[b'labels']
    labels_one_hot = one_hot_labels(labels).T
    return data, labels_one_hot, labels

def preprocess_data(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

def load_data():
    trainSet = {}
    validSet = {}
    testSet = {}

    trainSet['data'], trainSet['one_hot'], trainSet['labels'] = load_batch('data_batch_1')
    validSet['data'], validSet['one_hot'], validSet['labels'] = load_batch('data_batch_2')
    testSet['data'], testSet['one_hot'], testSet['labels'] = load_batch('test_batch')

    # normalization
    trainSet['data'] = preprocess_data(trainSet['data'])
    validSet['data'] = preprocess_data(validSet['data'])
    testSet['data'] = preprocess_data(testSet['data'])

    return trainSet, validSet, testSet


class Classifier():

    def __init__(self, hidden_nodes, regularization):
        self.hidden_nodes = hidden_nodes

        self.W1 = np.zeros((hidden_nodes, d))
        self.W2 = np.zeros((K, hidden_nodes))
        self.b1 = np.zeros((hidden_nodes, 1))
        self.b2 = np.zeros((K, 1))

        self.lambda_reg = regularization

        np.random.seed(42)
        self.initialization()


    def initialization(self):
        mu = 0
        sigma1 = 1 / np.sqrt(d)
        sigma2 = 1 / np.sqrt(self.hidden_nodes)

        self.W1 = np.random.normal(mu, sigma1, (self.W1.shape))
        self.W2 = np.random.normal(mu, sigma2, (self.W2.shape))
        self.b1 = np.zeros((self.b1.shape))
        self.b2 = np.zeros((self.b2.shape))

    def softmax(self, s):
        return np.exp(s) / np.sum(np.exp(s), axis=0)

    def evaluateClassifier(self, X, W1, b1, W2, b2):
        s1 = np.dot(W1, X) + b1
        h = np.maximum(s1, 0)
        s2 = np.dot(W2, h) + b2
        P = self.softmax(s2)
        return P

    def cross_entropy(self, X, Y, W1, b1, W2, b2):
        P = self.evaluateClassifier(X, W1, b1, W2, b2)
        loss = -np.sum(Y * np.log(P)) / X.shape[1]
        return loss

    def computeCost(self, X, Y, W1, b1, W2, b2):
        loss = self.cross_entropy(X, Y, W1, b1, W2, b2)
        reg = self.lambda_reg * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        loss_reg = loss + reg
        return loss, loss_reg

    def computeAccuracy(self, X, y):
        P = self.evaluateClassifier(X, self.W1, self.b1, self.W2, self.b2)
        y_pred = np.argmax(P, axis=0)
        acc = np.sum(y_pred == y) / X.shape[1]
        return acc

    
    