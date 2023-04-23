import numpy as np
import matplotlib.pyplot as plt
import pickle

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

def preprocess_data(data, mean, std):
    return (data - mean) / std

def load_data():
    trainSet = {}
    validSet = {}
    testSet = {}

    trainSet['data'], trainSet['one_hot'], trainSet['labels'] = load_batch('data_batch_1')
    validSet['data'], validSet['one_hot'], validSet['labels'] = load_batch('data_batch_2')
    testSet['data'], testSet['one_hot'], testSet['labels'] = load_batch('test_batch')

    # normalization
    mean = np.mean(trainSet['data'], axis=1, keepdims=True)
    std = np.std(trainSet['data'], axis=1, keepdims=True)
    trainSet['data'] = preprocess_data(trainSet['data'], mean, std)
    validSet['data'] = preprocess_data(validSet['data'], mean, std)
    testSet['data'] = preprocess_data(testSet['data'], mean, std)

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
    mean = np.mean(trainSet['data'], axis=1, keepdims=True)
    std = np.std(trainSet['data'], axis=1, keepdims=True)
    trainSet['data'] = preprocess_data(trainSet['data'], mean, std)
    validSet['data'] = preprocess_data(validSet['data'], mean, std)
    testSet['data'] = preprocess_data(testSet['data'], mean, std)

    return trainSet, validSet, testSet


class Classifier():
    def __init__(self, layer_dims, lambda_reg, batch_size, n_epochs, eta, cyclical, eta_min, eta_max, step_size, n_cycles, init_mode):
        self.layer_dims = layer_dims
        self.lambda_reg = lambda_reg
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.eta = eta
        self.cyclical = cyclical
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.step_size = step_size
        self.n_cycles = n_cycles
        self.init_mode = init_mode

        self.n_layers = len(layer_dims)
        
        self.W = [np.zeros((self.layer_dims[i+1], self.layer_dims[i])) for i in range(self.n_layers-1)]
        self.b = [np.zeros((self.layer_dims[i+1], 1)) for i in range(self.n_layers-1)]
        self.gradW = [np.zeros(w.shape for w in self.W)]
        self.gradb = [np.zeros(b.shape for b in self.b)]

        np.random.seed(42)
        self.initialization()

    def initialization(self):
        if self.init_mode == 'normal':
            mu = 0
            for i in range(1, len(self.layer_dims)):
                sigma = 1 / np.sqrt(self.layer_dims[i-1])
                self.W[i-1] = np.random.normal(mu, sigma, (self.layer_dims[i], self.layer_dims[i-1]))
                self.b[i-1] = np.zeros((self.layer_dims[i], 1))

        elif self.init_mode == 'he':
            for i in range(1, len(self.layer_dims)):
                self.W[i-1] = np.random.randn(self.layer_dims[i], self.layer_dims[i-1]) * np.sqrt(2 / self.layer_dims[i-1])
                self.b[i-1] = np.zeros((self.layer_dims[i], 1))

        elif self.init_mode == 'xavier':
            for i in range(1, len(self.layer_dims)):
                self.W[i-1] = np.random.randn(self.layer_dims[i], self.layer_dims[i-1]) * np.sqrt(1 / self.layer_dims[i-1])
                self.b[i-1] = np.zeros((self.layer_dims[i], 1))

        else:
            raise ValueError("Invalid initialization mode. Choose from 'normal', 'he', or 'xavier'.")

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def evaluateClassifier(self, X):
        activation = [X]
        for i in range(len(self.W)):
            s = np.dot(self.W[i], activation[-1]) + self.b[i]
            if i == len(self.W) - 1:
                activation.append(self.softmax(s))
            else:
                activation.append(np.maximum(0, s))
        return activation

    def cross_entropy(self, X, Y):
        P = self.evaluateClassifier(X)[-1]
        loss = -np.sum(Y * np.log(P)) / X.shape[1]
        return loss

    def computeCost(self, X, Y):
        loss = self.cross_entropy(X, Y)
        reg = self.lambda_reg * np.sum([np.sum(np.square(w)) for w in self.W])
        loss_reg =  loss + reg
        return loss, loss_reg

    def computeAccuracy(self, X, y):
        P = self.evaluateClassifier(X)[-1]
        y_pred = np.argmax(P, axis=0)
        acc =  np.sum(y_pred == y) / y.shape[0]
        return acc

    def computeGradients(self, X, Y):
        activations = self.evaluateClassifier(X)
        N = X.shape[1]
        G = -(Y - activations[-1])

        self.gradW = np.dot(G, activations[-2].T) / N + 2 * self.lambda_reg * self.W[-1]
        self.gradb = np.sum(G, axis=1, keepdims=True) / N

        for i in range(len(self.W) -2, -1, -1):
            G = np.dot(self.W[i+1].T, G) * np.where(activations[i+1] > 0, 1, 0)
            self.gradW[i] = np.dot(G, activations[i].T) / N + 2 * self.lambda_reg * self.W[i]
            self.gradb[i] = np.sum(G, axis=1, keepdims=True) / N

    def fit(self, X, Y, validSet):
        N = X.shape[1]
        metrics = {
            'train_loss': [],
            'train_cost': [],
            'train_acc': [],
            'valid_loss': [],
            'valid_cost': [],
            'valid_acc': [],
        }

        n_batch = N // self.batch_size

        if self.cyclical:
            iterations = self.n_epochs * 2 * self.step_size
            num_epochs = iterations // n_batch
        else:
            num_epochs = self.n_epochs

        for epoch in range(num_epochs):

            # shuffle data
            idx = np.arange(N)
            np.random.shuffle(idx)
            X = X[:, idx]
            Y = Y[:, idx]

            for j in range(n_batch):
                j_start = j * self.batch_size
                j_end = (j+1) * self.batch_size

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                self.computeGradients(X_batch, Y_batch)

                if self.cyclical:
                    t = epoch * n_batch + j
                    cycle = np.floor(t / (2 * self.step_size))
                    if 2 * cycle * self.step_size <= t <= (2 * cycle + 1) * self.step_size:
                        self.eta = self.eta_min + (t - 2 * cycle * self.step_size) / self.step_size * (self.eta_max - self.eta_min)
                    elif (2 * cycle + 1) * self.step_size <= t <= 2 * (cycle + 1) * self.step_size:
                        self.eta = self.eta_max - (t - (2 * cycle + 1) * self.step_size) / self.step_size * (self.eta_max - self.eta_min)

                for i in range(len(self.W)):
                    self.W[i] -= self.eta * self.gradW[i]
                    self.b[i] -= self.eta * self.gradb[i]

            metrics['train_loss'].append(self.computeCost(X, Y)[0])
            metrics['train_cost'].append(self.computeCost(X, Y)[1])
            metrics['train_acc'].append(self.computeAccuracy(X, np.argmax(Y, axis=0)))
            metrics['valid_loss'].append(self.computeCost(validSet['data'], validSet['one_hot'])[0])
            metrics['valid_cost'].append(self.computeCost(validSet['data'], validSet['one_hot'])[1])
            metrics['valid_acc'].append(self.computeAccuracy(validSet['data'], np.argmax(validSet['one_hot'], axis=0)))

            print('Epoch: %d, Train accuracy: %.2f, Validation accuracy: %.2f' % (epoch, metrics['train_acc'][-1], metrics['valid_acc'][-1]))

        return metrics

        