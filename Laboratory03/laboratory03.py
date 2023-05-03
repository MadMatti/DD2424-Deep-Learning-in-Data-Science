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
    def __init__(self, layer_dims, lambda_reg, batch_size, n_epochs, eta, cyclical, eta_min, eta_max, step_size, n_cycles, init_mode, batch_norm):
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
        self.batch_norm = batch_norm
        self.eps = 1e-10
        self.momentum = 0.9
        self.test_sensibility = True
        self.sigma = 1e-4

        self.n_layers = len(layer_dims)
        
        self.W = [np.zeros((self.layer_dims[i+1], self.layer_dims[i])) for i in range(self.n_layers-1)]
        self.b = [np.zeros((self.layer_dims[i+1], 1)) for i in range(self.n_layers-1)]
        self.gradW = [np.zeros_like(w) for w in self.W]
        self.gradb = [np.zeros_like(b) for b in self.b]

        if self.batch_norm:
            self.gamma = [np.ones((self.layer_dims[i+1], 1)) for i in range(self.n_layers-2)]
            self.beta = [np.zeros((self.layer_dims[i+1], 1)) for i in range(self.n_layers-2)]
            self.gradGamma = [np.zeros_like(g) for g in self.gamma]
            self.gradBeta = [np.zeros_like(b) for b in self.beta]

        np.random.seed(42)
        self.initialization()

    def initialization(self):
        if self.init_mode == 'normal':
            mu = 0
            for i in range(1, len(self.layer_dims)):
                if self.test_sensibility:
                    sigma = self.sigma
                else: sigma = 1 / np.sqrt(self.layer_dims[i-1])
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

    def relu(self, x):
        return np.maximum(0, x)

    def evaluateClassifier(self, X, mean=None, var=None):
        k = len(self.W)
        X_layers, S, S_bn = [X.copy()]+[None]*(k-1), [None]*(k-1), [None]*(k-1)

        if self.batch_norm:
            if mean is None and var is None:
                return_values = True
                mean, var = [None]*(k-1), [None]*(k-1)
            else:
                return_values = False

        # Iterate layers
        for i in range(k-1):
            S[i] = self.W[i]@X_layers[i] + self.b[i]
            if self.batch_norm:
                if return_values:
                    mean[i] = np.mean(S[i], axis=1, keepdims=True)
                    var[i] = np.var(S[i], axis=1, keepdims=True)
                S_bn[i] = (S[i] - mean[i]) / np.sqrt(var[i] + self.eps)
                S_bn_scaled = self.gamma[i] * S_bn[i] + self.beta[i]
                X_layers[i+1] = self.relu(S_bn_scaled)
            else:
                X_layers[i+1] = self.relu(S[i])

        # Output layer
        P = self.softmax(self.W[k-1]@X_layers[k-1] + self.b[k-1])

        if self.batch_norm:
            if return_values:
                return P, S_bn, S, X_layers[1:] , mean, var
            else:
                return P, S_bn, S, X_layers[1:]
        else:
            return P, X_layers[1:]

    def computeCost(self, X, Y, mean=None, var=None):

        if self.batch_norm:
            if mean is None and var is None:
                P, S_bn, S, X_layers, mean, var = self.evaluateClassifier(X, mean, var)
            else:
                P, S_bn, S, X_layers = self.evaluateClassifier(X, mean, var)
        else:
            P, X_layers = self.evaluateClassifier(X)

        loss = sum(-np.log((Y*P).sum(axis=0))) / X.shape[1]

        loss_reg = 0
        for w in self.W:
            loss_reg += self.lambda_reg * np.sum(w**2)

        return loss, loss + loss_reg

    def computeAccuracy(self, X, y, mean=None, var=None):

        if self.batch_norm:
            if mean is None and var is None:
                P, S_bn, S, X_layers, mean, var = self.evaluateClassifier(X, mean, var)
            else:
                P, S_bn, S, X_layers = self.evaluateClassifier(X, mean, var)
        else:
            P, X_layers = self.evaluateClassifier(X)

        accuracy = np.mean(np.argmax(P, axis=0) == y)

        return accuracy

    def batchNorm_backward(self, G, S, mean, var):
        n = S.shape[1]
        G1 = G * (((var + 1e-15) ** (-0.5))@np.ones((1,n)))
        G2 = G * (((var + 1e-15) ** (-1.5))@np.ones((1,n)))
        D = S - mean@np.ones((1,n))
        c = (G2 * D)@np.ones((n,1))
        G_batch = G1 - (G1@np.ones((n,1))) / n - D * (c@np.ones((1,n))) / n

        return G_batch

    def computeGradients(self, X, Y, P, S_bn, S, X_layers, mean, var):
        k = len(self.W)
        n = X.shape[1]

        temp_gradW, temp_gradb = [None]*k, [None]*k
        if self.batch_norm:
            temp_gradGamma, temp_gradBeta = [None]*(k-1), [None]*(k-1)

        # output layer
        X_layers = [X.copy()] + X_layers
        G = -(Y - P)
        temp_gradW[k-1] = G@X_layers[k-1].T / n + 2 * self.lambda_reg * self.W[k-1]
        temp_gradb[k-1] = G@np.ones((n,1)) / n

        # layer k-1
        G = self.W[k-1].T@G 
        G = G * (X_layers[k-1] > 0)

        # hidden layers
        for i in range(k-2, -1, -1):
            if self.batch_norm:
                temp_gradGamma[i] = (G*S_bn[i])@np.ones((n,1)) / n
                temp_gradBeta[i] = G@np.ones((n,1)) / n
                G = G * (self.gamma[i]@np.ones((1,n)))
                G = self.batchNorm_backward(G, S[i], mean[i], var[i])

            temp_gradW[i] = G@X_layers[i].T / n + 2 * self.lambda_reg * self.W[i]
            temp_gradb[i] = G@np.ones((n,1)) / n

            if i > 0:
                G = self.W[i].T@G
                G = G * (X_layers[i] > 0)

        self.gradW, self.gradb = temp_gradW, temp_gradb
        if self.batch_norm:
            self.gradGamma, self.gradBeta = temp_gradGamma, temp_gradBeta

    # to do grads numerical
    def computeGradsNum(self, X, Y, mean, var, h=1e-7):

        grad_W = [w.copy() for w in self.W]
        grad_b = [b.copy() for b in self.b]
        if self.batch_norm:
            grad_gamma = [g.copy() for g in self.gamma]
            grad_beta = [b.copy() for b in self.beta]

        c = self.computeCost(X, Y, mean, var)[1]
        k = len(self.W)

        for i in range(k):

            # biases
            for j in range(self.b[i].shape[0]):
                self.b[i][j,0] += h
                c2 = self.computeCost(X, Y, mean, var)[1]
                grad_b[i][j,0] = (c2 - c) / h
                self.b[i][j,0] -= h

            # weights
            for j in range(self.W[i].shape[0]):
                for l in range(self.W[i].shape[1]):
                    self.W[i][j,l] += h
                    c2 = self.computeCost(X, Y, mean, var)[1]
                    grad_W[i][j,l] = (c2 - c) / h
                    self.W[i][j,l] -= h

            if self.batch_norm and i < k-1:
                # gamma
                for j in range(self.gamma[i].shape[0]):
                    self.gamma[i][j,0] += h
                    c2 = self.computeCost(X, Y, mean, var)[1]
                    grad_gamma[i][j,0] = (c2 - c) / h
                    self.gamma[i][j,0] -= h

                # beta
                for j in range(self.beta[i].shape[0]):
                    self.beta[i][j,0] += h
                    c2 = self.computeCost(X, Y, mean, var)[1]
                    grad_beta[i][j,0] = (c2 - c) / h
                    self.beta[i][j,0] -= h

        if self.batch_norm:
            return grad_W, grad_b, grad_gamma, grad_beta
        else:
            return grad_W, grad_b       

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

            for j in range(n_batch):
                j_start = j * self.batch_size
                j_end = (j+1) * self.batch_size

                X_batch = X[:, j_start:j_end]
                Y_batch = Y[:, j_start:j_end]

                if self.batch_norm:
                    P, S_bn, S, X_layers, mean, var = self.evaluateClassifier(X_batch)
                else:
                    P, X_layers = self.evaluateClassifier(X_batch)
                
                #initialize average mean and var
                if epoch==0 and j==0 and self.batch_norm:
                    self.mean_avg = mean
                    self.var_avg = var
                elif self.batch_norm:
                    self.mean_avg = [self.momentum * self.mean_avg[i] + (1-self.momentum) * mean[i] for i in range(len(mean))]
                    self.var_avg = [self.momentum * self.var_avg[i] + (1-self.momentum) * var[i] for i in range(len(var))]
                else:
                    self.mean_avg = None
                    self.var_avg = None

                # gradients
                if self.batch_norm:
                    self.computeGradients(X_batch, Y_batch, P, S_bn, S, X_layers, mean, var)
                else:
                    self.computeGradients(X_batch, Y_batch, P, S_bn=None, S=None, X_layers=X_layers, mean=None, var=None)

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
                if self.batch_norm:
                    for i in range(len(self.gamma)):
                        self.gamma[i] -= self.eta * self.gradGamma[i]
                        self.beta[i] -= self.eta * self.gradBeta[i]

            metrics['train_loss'].append(self.computeCost(X, Y, self.mean_avg, self.var_avg)[0])
            metrics['train_cost'].append(self.computeCost(X, Y, self.mean_avg, self.var_avg)[1])
            metrics['train_acc'].append(self.computeAccuracy(X, np.argmax(Y, axis=0), self.mean_avg, self.var_avg))
            metrics['valid_loss'].append(self.computeCost(validSet['data'], validSet['one_hot'], self.mean_avg, self.var_avg)[0])
            metrics['valid_cost'].append(self.computeCost(validSet['data'], validSet['one_hot'], self.mean_avg, self.var_avg)[1])
            metrics['valid_acc'].append(self.computeAccuracy(validSet['data'], np.argmax(validSet['one_hot'], axis=0), self.mean_avg, self.var_avg))

            print('Epoch: %d, Train accuracy: %.2f, Validation accuracy: %.2f' % (epoch, metrics['train_acc'][-1], metrics['valid_acc'][-1]))

            # shuffle data
            idx = np.arange(N)
            np.random.shuffle(idx)
            X = X[:, idx]
            Y = Y[:, idx]

        return metrics

    def grid_search(self, X, Y, validSet, lambda_values):

        search_metrics = {
            'lambda_search': [],
            'train_acc': [],
            'valid_acc': [],
        }

        best_metrics = {
            'accuracy_train' : 0,
            'accuracy_val' : 0,
            'lambda_' : np.copy(self.lambda_reg)
        }

        # perform grid search
        for lmb in lambda_values:
            self.lambda_reg = lmb
            search_metrics['lambda_search'].append(lmb)

            metrics = self.fit(X, Y, validSet)
            search_metrics['train_acc'].append(metrics['train_acc'][-1])
            search_metrics['valid_acc'].append(metrics['valid_acc'][-1])

            if metrics['valid_acc'][-1] > best_metrics['accuracy_val']:
                best_metrics['accuracy_train'] = metrics['train_acc'][-1]
                best_metrics['accuracy_val'] = metrics['valid_acc'][-1]
                best_metrics['lambda_'] = lmb

            print('Lambda: %f, Train accuracy: %.2f, Validation accuracy: %.2f' % (lmb, metrics['train_acc'][-1], metrics['valid_acc'][-1]))

        return search_metrics, best_metrics



def check_gradients(X_train, y_train_oh, params):
    # check differences between analytical and numerical gradients usign the first 20 input samples
    X = X_train[0:20,0:5]
    Y = y_train_oh[:,0:5]
    net = Classifier(**params)

    if net.batch_norm:
        P, S_bn, S, X_layers , mean, var = net.evaluateClassifier(X)
        net.computeGradients(X, Y, P, S_bn=S_bn, S=S, X_layers=X_layers, mean=mean, var=var)
        analytic_grads = {"W": net.gradW, "b": net.gradb, "gamma": net.gradGamma, "beta": net.gradBeta}
        num_grads = net.computeGradsNum(X, Y, mean=None, var=None, h=1e-7)
        num_grads = {"W": num_grads[0], "b": num_grads[1], "gamma": num_grads[2], "beta": num_grads[3]}
    else:
        P, X_layers = net.evaluateClassifier(X)
        net.computeGradients(X, Y, P, S_bn=None, S=None, X_layers=X_layers, mean=None, var=None)
        analytic_grads = {"W": net.gradW, "b": net.gradb}
        num_grads = net.computeGradsNum(X, Y, mean=None, var=None, h=1e-7)
        num_grads = {"W": num_grads[0], "b": num_grads[1]}

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
    if MLP.batch_norm:
        return MLP.computeAccuracy(testSet['data'], np.argmax(testSet['one_hot'], axis=0), MLP.mean_avg, MLP.var_avg)*100
    else:
        return MLP.computeAccuracy(testSet['data'], np.argmax(testSet['one_hot'], axis=0))*100

def call_grid_search(X, Y, validSet, params):

    # random lambdas to search
    # l_max, l_min = -1, -5
    # l = l_min + (l_max - l_min) * np.random.rand(10)
    # list_lambda = [10**i for i in l]
    # list_lambda.sort(reverse=True)
    # print(list_lambda)

    list_lambda = [0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002, 0.003, 0.004, 0.005]
    print(list_lambda)

    MLP_search = Classifier(**params)

    search, best = MLP_search.grid_search(X, Y, validSet, list_lambda)

    print(search)
    print("Best lambda: ", best['lambda_'])
    print("Best train accuracy: ", best['accuracy_train'])
    print("Best valid accuracy: ", best['accuracy_val'])

    # plot train and test accuracy on the y axis and lambda on the x axis
    plt.plot(list_lambda, search['train_acc'], label='train', color='seagreen')
    plt.plot(list_lambda, search['valid_acc'], label='valid', color='indianred')
    plt.title('Accuracy Plot')
    plt.xlabel('lambda')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    trainSet, validSet, testSet = load_data_more()

    params = {
        'layer_dims': [d, 150, 150, K],
        'lambda_reg': 0.005,
        'batch_size': 100,
        'n_epochs': 200,
        'eta': 0.01,
        'cyclical': True,
        'eta_min': 1e-5,
        'eta_max': 1e-1,
        'step_size': 2250,
        'n_cycles': 2,
        'init_mode': 'he',
        'batch_norm': True,
    }

    # check_gradients(trainSet['data'], trainSet['one_hot'], params)
    # exit()

    MLP = Classifier(**params)

    # grid search
    # call_grid_search(trainSet['data'], trainSet['one_hot'], validSet, params)
    # exit()
    # metrics = MLP.fit(trainSet['data'], trainSet['one_hot'], validSet)

    title = '2 cycle with n_s = 2250, lambda = 0.005'
    # plot_curves(metrics, title)
    # print("Final test accuracy: %.2f" % test_accuracy(MLP, testSet))

    hinned_units = [100, 250, 500]
    all_metrics = {}
    for i in range(len(hinned_units)):
        params['layer_dims'] = [d, hinned_units[i], hinned_units[i], K]
        MLP = Classifier(**params)
        metrics = MLP.fit(trainSet['data'], trainSet['one_hot'], validSet)
        all_metrics[i] = metrics

    # plot cost, loss and accuracy for each configurations on different colors
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.tight_layout(pad=3.0)

    ax[0].plot(all_metrics[0]['train_cost'], label='100_train', color='seagreen')
    ax[0].plot(all_metrics[0]['valid_cost'], label='100_valid', color='seagreeen', linestyle='dashed')
    ax[0].plot(all_metrics[1]['train_cost'], label='250_train', color='indianred')
    ax[0].plot(all_metrics[1]['valid_cost'], label='250_valid', color='indianred', linestyle='dashed')
    ax[0].plot(all_metrics[2]['train_cost'], label='500_train', color='yellowgreen')
    ax[0].plot(all_metrics[2]['valid_cost'], label='500_valid', color='yellowgreen', linestyle='dashed')
    ax[0].set_title('Cost Plot')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Cost')
    ax[0].legend()

    ax[1].plot(all_metrics[0]['train_loss'], label='100_train', color='seagreen')
    ax[1].plot(all_metrics[0]['valid_loss'], label='100_valid', color='seagreeen', linestyle='dashed')
    ax[1].plot(all_metrics[1]['train_loss'], label='250_train', color='indianred')
    ax[1].plot(all_metrics[1]['valid_loss'], label='250_valid', color='indianred', linestyle='dashed')
    ax[1].plot(all_metrics[2]['train_loss'], label='500_train', color='yellowgreen')
    ax[1].plot(all_metrics[2]['valid_loss'], label='500_valid', color='yellowgreen', linestyle='dashed')
    ax[1].set_title('Loss Plot')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Loss')
    ax[1].legend()

    ax[2].plot(all_metrics[0]['train_acc'], label='100_train', color='seagreen')
    ax[2].plot(all_metrics[0]['valid_acc'], label='100_valid', color='seagreeen', linestyle='dashed')
    ax[2].plot(all_metrics[1]['train_acc'], label='250_train', color='indianred')
    ax[2].plot(all_metrics[1]['valid_acc'], label='250_valid', color='indianred', linestyle='dashed')
    ax[2].plot(all_metrics[2]['train_acc'], label='500_train', color='yellowgreen')
    ax[2].plot(all_metrics[2]['valid_acc'], label='500_valid', color='yellowgreen', linestyle='dashed')
    ax[2].set_title('Accuracy Plot')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Accuracy')
    ax[2].legend()

    plt.suptitle("Learning curves for " + title, y=0.98)
    plt.subplots_adjust(top=0.85)
    plt.show()

    
    
        