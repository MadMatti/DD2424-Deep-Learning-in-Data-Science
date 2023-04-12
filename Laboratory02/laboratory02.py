import pickle 
import numpy as np
import matplotlib.pyplot as plt


data_dir = 'dataset/'

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

        self.gradW1 = np.zeros((hidden_nodes, d))
        self.gradW2 = np.zeros((K, hidden_nodes))
        self.gradb1 = np.zeros((hidden_nodes, 1))
        self.gradb2 = np.zeros((K, 1))

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

    def computeGradients(self, X, Y, P, W1, b1, W2, b2):
        s1 = np.dot(W1, X) + b1
        h = np.maximum(s1, 0)
        G = -(Y - P)

        self.gradb2 = np.sum(G, axis=1, keepdims=True) / X.shape[1]
        self.gradW2 = np.dot(G, h.T) / X.shape[1] + 2 * self.lambda_reg * W2

        s1 = np.where(s1 > 0, 1, 0)
        G = np.dot(W2.T, G) * s1

        self.gradb1 = np.sum(G, axis=1, keepdims=True) / X.shape[1]
        self.gradW1 = np.dot(G, X.T) / X.shape[1] + 2 * self.lambda_reg * W1

    def computeGradsNum(self, X, Y, W1, b1, W2, b2, h):
        gradW1 = np.zeros(W1.shape)
        gradb1 = np.zeros(b1.shape)
        gradW2 = np.zeros(W2.shape)
        gradb2 = np.zeros(b2.shape)

        c = self.computeCost(X, Y, W1, b1, W2, b2)[0]

        for i in range(len(b1)):
            b1_try = np.array(b1)
            b1_try[i] += h
            c2 = self.computeCost(X, Y, W1, b1_try, W2, b2)[0]
            gradb1[i] = (c2 - c) / h

        for i in range(W1.shape[0]):
            for j in range(W1.shape[1]):
                W1_try = np.array(W1)
                W1_try[i, j] += h
                c2 = self.computeCost(X, Y, W1_try, b1, W2, b2)[0]
                gradW1[i, j] = (c2 - c) / h

        for i in range(len(b2)):
            b2_try = np.array(b2)
            b2_try[i] += h
            c2 = self.computeCost(X, Y, W1, b1, W2, b2_try)[0]
            gradb2[i] = (c2 - c) / h

        for i in range(W2.shape[0]):
            for j in range(W2.shape[1]):
                W2_try = np.array(W2)
                W2_try[i, j] += h
                c2 = self.computeCost(X, Y, W1, b1, W2_try, b2)[0]
                gradW2[i, j] = (c2 - c) / h

        return gradW1, gradb1, gradW2, gradb2



def check_gradients(X_train, y_train_oh):
    X_train = trainSet['data']
    y_train_oh = trainSet['one_hot']

    X = X_train[0:20,[0]]
    Y = y_train_oh[:,[0]]
    
    network = Classifier(50, 0)
    network.initialization()

    P = network.evaluateClassifier(X, network.W1[:,0:20], network.b1, network.W2, network.b2)
    
    network.computeGradients(X, Y, P, network.W1[:, 0:20], network.b1, network.W2, network.b2)
    gradW1, gradb1, gradW2, gradb2 = network.computeGradsNum(X, Y, network.W1[:, 0:20], network.b1, network.W2, network.b2, 1e-5)

    # absolute difference between the numerical and analytical gradient
    diffW1 = np.abs(gradW1 - network.gradW1)
    diffb1 = np.abs(gradb1 - network.gradb1)
    diffW2 = np.abs(gradW2 - network.gradW2)
    diffb2 = np.abs(gradb2 - network.gradb2)

    # print
    print('For W1: '+str(np.mean(diffW1<1e-6)*100)+"% of absolute errors below 1e-6"+" and maximum absolute error is "+str(diffW1.max()))
    print('For b1: '+str(np.mean(diffb1<1e-6)*100)+"% of absolute errors below 1e-6"+" and maximum absolute error is "+str(diffb1.max()))
    print('For W2: '+str(np.mean(diffW2<1e-6)*100)+"% of absolute errors below 1e-6"+" and maximum absolute error is "+str(diffW2.max()))
    print('For b2: '+str(np.mean(diffb2<1e-6)*100)+"% of absolute errors below 1e-6"+" and maximum absolute error is "+str(diffb2.max()))
    print("\n")

    # relative error between the numerical and analytical gradient
    diffW1_rel = np.abs(gradW1 - network.gradW1) / np.maximum(1e-6, np.abs(gradW1) + np.abs(network.gradW1))
    diffb1_rel = np.abs(gradb1 - network.gradb1) / np.maximum(1e-6, np.abs(gradb1) + np.abs(network.gradb1))
    diffW2_rel = np.abs(gradW2 - network.gradW2) / np.maximum(1e-6, np.abs(gradW2) + np.abs(network.gradW2))
    diffb2_rel = np.abs(gradb2 - network.gradb2) / np.maximum(1e-6, np.abs(gradb2) + np.abs(network.gradb2))

    # print
    print('For W1: '+str(np.mean(diffW1_rel<1e-6)*100)+"% of relative errors below 1e-6"+" and maximum relative error is "+str(diffW1_rel.max()))
    print('For b1: '+str(np.mean(diffb1_rel<1e-6)*100)+"% of relative errors below 1e-6"+" and maximum relative error is "+str(diffb1_rel.max()))
    print('For W2: '+str(np.mean(diffW2_rel<1e-6)*100)+"% of relative errors below 1e-6"+" and maximum relative error is "+str(diffW2_rel.max()))
    print('For b2: '+str(np.mean(diffb2_rel<1e-6)*100)+"% of relative errors below 1e-6"+" and maximum relative error is "+str(diffb2_rel.max()))



if __name__ == '__main__':
    trainSet, validSet, testSet = load_data()

    check_gradients(trainSet['data'], trainSet['one_hot'])


    
    
    