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

    def __init__(self, hidden_nodes):
        self.hidden_nodes = hidden_nodes

        self.W1 = np.zeros((hidden_nodes, d))
        self.W2 = np.zeros((K, hidden_nodes))
        self.b1 = np.zeros((hidden_nodes, 1))
        self.b2 = np.zeros((K, 1))

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

    
    