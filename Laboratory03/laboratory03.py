import numpy as np
import matplotlib.pyplot as plt
import pickle

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
        self.gradW = [np.zeros_like(w) for w in self.W]
        self.gradb = [np.zeros_like(b) for b in self.b]

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

        temp_gradW = [None] * len(self.W)
        temp_gradb = [None] * len(self.b)

        temp_gradW[-1] = np.dot(G, activations[-2].T) / N + 2 * self.lambda_reg * self.W[-1]
        temp_gradb[-1] = np.sum(G, axis=1, keepdims=True) / N

        for i in range(len(self.W) - 2, -1, -1):
            G = np.dot(self.W[i + 1].T, G) * np.where(activations[i + 1] > 0, 1, 0)
            temp_gradW[i] = np.dot(G, activations[i].T) / N + 2 * self.lambda_reg * self.W[i]
            temp_gradb[i] = np.sum(G, axis=1, keepdims=True) / N

        self.gradW = temp_gradW
        self.gradb = temp_gradb

    def set_params(self, W=None, b=None):
        if W is not None:
            self.W = W
        if b is not None:
            self.b = b

    def computeGradsNum(self, X, Y, h):
        grads = {"W": [], "b": []}
        for i in range(len(self.W)):
            grads["W"].append(np.zeros_like(self.W[i]))
            grads["b"].append(np.zeros_like(self.b[i]))

        for i in range(len(self.b)):
            for j in range(self.b[i].shape[0]):
                b_try = np.copy(self.b)
                b_try[i][j, 0] -= h
                self.set_params(b=b_try)
                c1 = self.computeCost(X, Y)[1]
                b_try = np.copy(self.b)
                b_try[i][j, 0] += h
                self.set_params(b=b_try)
                c2 = self.computeCost(X, Y)[1]
                grads["b"][i][j, 0] = (c2 - c1) / (2 * h)

            for j in range(self.W[i].shape[0]):
                for k in range(self.W[i].shape[1]):
                    W_try = np.copy(self.W)
                    W_try[i][j, k] -= h
                    self.set_params(W=W_try)
                    c1 = self.computeCost(X, Y)[1]
                    W_try = np.copy(self.W)
                    W_try[i][j, k] += h
                    self.set_params(W=W_try)
                    c2 = self.computeCost(X, Y)[1]
                    grads["W"][i][j, k] = (c2 - c1) / (2 * h)

        self.set_params(W=self.W, b=self.b)
        return grads

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
            iterations = self.n_cycles * 2 * self.step_size
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


def check_gradients(X_train, y_train_oh, params):
    # check differences between analytical and numerical gradients usign the first 20 input samples
    X = X_train[0:20,0:5]
    Y = y_train_oh[:,0:5]
    net = Classifier(**params)
    net.computeGradients(X, Y)
    analytic_grads = {"W": net.gradW, "b": net.gradb}
    num_grads = net.computeGradsNum(X, Y, 1e-7)

    # Compute the % of absolute errors below 1e-6 and maximum absolute error for weights by layers
    print("For weights, the % of absolute errors below 1e-6 by layers is:")
    weights_perc_err = [np.mean(np.abs(analytic_grads["W"][i] - num_grads["W"][i]) < 1e-6) * 100 for i in range(len(analytic_grads["W"]))]
    print(weights_perc_err)
    print("and the maximum absolute error by layers is:")
    weights_max_err = [np.max(np.abs(analytic_grads["W"][i] - num_grads["W"][i])) for i in range(len(analytic_grads["W"]))]
    print(weights_max_err)

    # Compute the % of absolute errors below 1e-6 and maximum absolute error for bias by layers
    print("\nFor bias, the % of absolute errors below 1e-6 by layers is:")
    bias_perc_err = [np.mean(np.abs(analytic_grads["b"][i] - num_grads["b"][i]) < 1e-6) * 100 for i in range(len(analytic_grads["b"]))]
    print(bias_perc_err)
    print("and the maximum absolute error by layers is:")
    bias_max_err = [np.max(np.abs(analytic_grads["b"][i] - num_grads["b"][i])) for i in range(len(analytic_grads["b"]))]
    print(bias_max_err)

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

def test_accuracy(MLP, testSet):
    return MLP.computeAccuracy(testSet['data'], np.argmax(testSet['one_hot'], axis=0))*100


if __name__ == '__main__':
    trainSet, validSet, testSet = load_data_more()
    
    params = {
        'layer_dims': [d, 50, 30, 20, 20, 10, 10, 10, 10, K],
        'lambda_reg': 0.005,
        'batch_size': 100,
        'n_epochs': 200,
        'eta': 0.01,
        'cyclical': True,
        'eta_min': 1e-5,
        'eta_max': 1e-1,
        'step_size': 2250,
        'n_cycles': 2,
        'init_mode': 'xavier'
    }

    #check_gradients(trainSet['data'], trainSet['one_hot'], params)

    MLP = Classifier(**params)
    metrics = MLP.fit(trainSet['data'], trainSet['one_hot'], validSet)

    title = '2 cycle with n_s = 2250, lambda = 0.005'
    plot_curves(metrics, title)
    print("Final test accuracy: %.2f" % test_accuracy(MLP, testSet))
        