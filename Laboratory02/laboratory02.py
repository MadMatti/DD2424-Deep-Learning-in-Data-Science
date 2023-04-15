import math
import pickle 
import numpy as np
import matplotlib.pyplot as plt
import csv

import warnings
warnings.filterwarnings("ignore")


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

def load_data_more():
    trainSet = {}
    validSet = {}
    testSet = {}

    images_1, labels_one_hot_1, labels_1 = load_batch('data_batch_1')
    images_2, labels_one_hot_2, labels_2 = load_batch('data_batch_2')
    images_3, labels_one_hot_3, labels_3 = load_batch('data_batch_3')
    images_4, labels_one_hot_4, labels_4 = load_batch('data_batch_4')
    images_5, labels_one_hot_5, labels_5 = load_batch('data_batch_5')
    testSet['data'], testSet['one_hot'], testSet['labels'] = load_batch('test_batch')

    images = np.concatenate((images_1, images_2, images_3, images_4, images_5), axis=1)
    labels_one_hot = np.concatenate((labels_one_hot_1, labels_one_hot_2, labels_one_hot_3, labels_one_hot_4, labels_one_hot_5), axis=1)
    labels = np.concatenate((labels_1, labels_2, labels_3, labels_4, labels_5), axis=0)

    # subset for validation
    np.random.seed(100)
    indices = np.random.choice(images.shape[1], 1000, replace=False)
    validSet['data'] = images[:, indices]
    validSet['one_hot'] = labels_one_hot[:, indices]
    validSet['labels'] = labels[indices]

    # subset for training
    trainSet['data'] = np.delete(images, indices, axis=1)
    trainSet['one_hot'] = np.delete(labels_one_hot, indices, axis=1)
    trainSet['labels'] = np.delete(labels, indices, axis=0)

    # normalization
    trainSet['data'] = preprocess_data(trainSet['data'])
    validSet['data'] = preprocess_data(validSet['data'])
    testSet['data'] = preprocess_data(testSet['data'])

    return trainSet, validSet, testSet


class Classifier():

    def __init__(self, hidden_nodes, regularization, batch_size, n_epochs, learning_rate, cyclical, eta_min, eta_max ,step_size, n_cycles):
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
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = learning_rate
        self.cyclical = cyclical
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.step_size = step_size
        self.n_cycles = n_cycles

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

    # adaptation of matlab code to calculate numerical gradients
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

    def fit(self, X, Y, validSet):
        N = X.shape[1]
        metrics = {
            'train_loss': [],
            'train_cost': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_cost': [],
            'valid_acc': []
        }

        n_batch = N // self.batch_size

        if self.cyclical:
            iterations = self.n_cycles * 2 * self.step_size # each cycle correspond to two steps
            num_epochs = iterations // n_batch
            list_eta = []
        else:
            num_epochs = self.n_epochs

        for epoch in range(num_epochs):

            for j in range(n_batch):
                j_start = j * self.batch_size
                j_end = (j + 1) * self.batch_size

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                P_batch = self.evaluateClassifier(X_batch, self.W1, self.b1, self.W2, self.b2)
                self.computeGradients(X_batch, Y_batch, P_batch, self.W1, self.b1, self.W2, self.b2)

                # implement formulas 14 and 15 for cyclical learning rate
                if self.cyclical:
                    t = epoch * n_batch + j
                    cycle = np.floor(t / (2 * self.step_size))
                    if 2 * cycle * self.step_size <= t <= (2 * cycle + 1) * self.step_size:
                        self.eta = self.eta_min + (t - 2 * cycle * self.step_size) / self.step_size * (self.eta_max - self.eta_min)
                    elif (2 * cycle + 1) * self.step_size <= t <= 2 * (cycle + 1) * self.step_size:
                        self.eta = self.eta_max - (t - (2 * cycle + 1) * self.step_size) / self.step_size * (self.eta_max - self.eta_min)

                list_eta.append(self.eta)

                self.W1 -= self.eta * self.gradW1
                self.b1 -= self.eta * self.gradb1
                self.W2 -= self.eta * self.gradW2
                self.b2 -= self.eta * self.gradb2

            metrics['train_loss'].append(self.computeCost(X, Y, self.W1, self.b1, self.W2, self.b2)[0])
            metrics['train_cost'].append(self.computeCost(X, Y, self.W1, self.b1, self.W2, self.b2)[1])
            metrics['train_acc'].append(self.computeAccuracy(X, np.argmax(Y, axis=0)))
            metrics['valid_loss'].append(self.computeCost(validSet['data'], validSet['one_hot'], self.W1, self.b1, self.W2, self.b2)[0])
            metrics['valid_cost'].append(self.computeCost(validSet['data'], validSet['one_hot'], self.W1, self.b1, self.W2, self.b2)[1])
            metrics['valid_acc'].append(self.computeAccuracy(validSet['data'], np.argmax(validSet['one_hot'], axis=0)))

        return metrics, list_eta

    def random_search(self, X, Y, validSet, lambda_values, n_random):
        search_metrics = {
            'lambda_search' : [],
            'accuracy_train' : [],
            'accuracy_val' : []
        }
        best_metrics = {
            'accuracy_train' : 0,
            'accuracy_val' : 0,
            'lamba_' : np.copy(self.lambda_reg)
        }

        for lmb in lambda_values:
            list_accuracies_train = []
            list_accuracies_val = []
            self.lambda_reg = lmb
            search_metrics['lambda_search'].append(lmb)

            for rnd in range(n_random):
                np.random.seed(rnd)

                self.initialization()
            
                metrics, _ = self.fit(X, Y, validSet)
                
                list_accuracies_train.append(metrics['train_acc'][-1])
                list_accuracies_val.append(metrics['valid_acc'][-1])

            search_metrics['accuracy_train'].append(np.mean(list_accuracies_train))
            search_metrics['accuracy_val'].append(np.mean(list_accuracies_val))
            
            print('lambda = ', lmb, 'accuracy = ', np.mean(list_accuracies_val))

            if np.mean(list_accuracies_val) > best_metrics['accuracy_val']:
                best_metrics['accuracy_train'] = np.mean(list_accuracies_train)
                best_metrics['accuracy_val'] = np.mean(list_accuracies_val)
                best_metrics['lamba_'] = lmb

        return search_metrics, best_metrics

 

def check_gradients(X_train, y_train_oh):
    X_train = trainSet['data']
    y_train_oh = trainSet['one_hot']
    X = X_train[0:20,[0]]
    Y = y_train_oh[:,[0]]

    network = Classifier(50, 0, 20, 200, 0.001, False, 1e-5, 1e-1,500, 1)
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


def plot_curves(metrics, title):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=3.0)

    ax[0].plot(metrics['train_cost'], label='train', color='seagreen')
    ax[0].plot(metrics['valid_cost'], label='valid', color='indianred')
    ax[0].set_title('Cost Plot')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Cost')
    ax[0].legend()
    ax[0].grid(True)
    #ax[0].set_ylim(0, 3)

    ax[1].plot(metrics['train_loss'], label='train', color='seagreen')
    ax[1].plot(metrics['valid_loss'], label='valid', color='indianred')
    ax[1].set_title('Loss Plot')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)
    #ax[1].set_ylim(0, 3)

    ax[2].plot(metrics['train_acc'], label='train', color='seagreen')
    ax[2].plot(metrics['valid_acc'], label='valid', color='indianred')
    ax[2].set_title('Accuracy Plot')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()
    ax[2].grid(True)
    #ax[2].set_ylim(0, 1)

    plt.suptitle("Learning curves for " + title, y=0.98)
    plt.subplots_adjust(top=0.85)
    plt.show()

def test_accuracy(network, testSet):
    P = network.evaluateClassifier(testSet['data'], network.W1, network.b1, network.W2, network.b2)
    y_pred = np.argmax(P, axis=0)
    y_true = np.argmax(testSet['one_hot'], axis=0)
    acc = np.mean(y_pred == y_true)
    return acc

def call_random_search():
    trainSet, validSet, testSet = load_data_more()

    # random lambdas to search
    l_max, l_min = -1, -5
    l = l_min + (l_max - l_min) * np.random.rand(10)
    print(l)
    list_lambda = [10**i for i in l]
    list_lambda.sort(reverse=True)
    print(list_lambda)

    network_search = Classifier(50, 0.01, 100, 200, 1e-5, True, 1e-5, 1e-1, 900, 2)

    search, best = network_search.random_search(trainSet['data'], trainSet['one_hot'], validSet, list_lambda, 10)

    print(best)

    print(search)

    # plot train and test accuracy on the y axis and lambda on the x axis
    plt.plot(list_lambda, search['accuracy_train'], label='train', color='seagreen')
    plt.plot(list_lambda, search['accuracy_val'], label='valid', color='indianred')
    plt.title('Accuracy Plot')
    plt.xlabel('lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def call_random_search_fine():
    trainSet, validSet, testSet = load_data_more()

    # random lambdas to search
    list_lambda = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001]
    print(list_lambda)

    network_search = Classifier(50, 0.01, 100, 200, 1e-5, True, 1e-5, 1e-1, 900, 4)

    search, best = network_search.random_search(trainSet['data'], trainSet['one_hot'], validSet, list_lambda, 10)

    print(best)

    print(search)

    # plot train and test accuracy on the y axis and lambda on the x axis
    plt.plot(list_lambda, search['accuracy_train'], label='train', color='seagreen')
    plt.plot(list_lambda, search['accuracy_val'], label='valid', color='indianred')
    plt.title('Accuracy Plot')
    plt.xlabel('lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()




if __name__ == '__main__':
    # trainSet, validSet, testSet = load_data()

    #check_gradients(trainSet['data'], trainSet['one_hot'])

    trainSet, validSet, testSet = load_data_more()

    network = Classifier(50, 0.001, 100, 200, 1e-5, True, 1e-5, 1e-1,900, 3)
    network.initialization()

    metrics, list_eta = network.fit(trainSet['data'], trainSet['one_hot'], validSet)

    # plot list eta
    # plt.plot(list_eta)
    # plt.title('List of eta')
    # plt.xlabel('iteration')
    # plt.ylabel('eta')
    # plt.show()

    # plot curves
    plot_curves(metrics, '3 cycle with n_s = 900, lambda = 0.001')
    print(test_accuracy(network, testSet))

    # call_random_search()
    # call_random_search_fine()
    

    
    
    