import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys

def load_file(filename):
    with open('Laboratory01/dataset/'+filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def one_hot_labels(labels, num_distincts=10):
    one_hot = np.zeros((num_distincts, len(labels)))
    for i, label in enumerate(labels):
        one_hot[label, i] = 1
    return one_hot

def LoadBatch(filename):
    dict = load_file(filename)
    images = dict[b'data'].T
    labels = dict[b'labels']
    labels_one_hot = one_hot_labels(labels)
    return images, labels_one_hot ,labels

def preprocess_data(data):
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True)
    return (data - mean) / std

def initialize_weights(input_dim, output_dim, seed=42, mean=0, std=0.01):
    #np.random.seed(seed)
    W = np.random.normal(mean, std, (output_dim, input_dim))
    #np.random.seed(seed)
    b = np.random.normal(mean, std, (output_dim, 1))

    return W, b

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def EvaluateClassifier(X, W, b):
    # eq 1,2
    s = W@X + b
    p = softmax(s)
    
    return p

def computeCost(X, y, W, b, lamda):
    # eq 5
    p = EvaluateClassifier(X, W, b)
    # loss function term
    loss_cross = sum(-np.log((y*p).sum(axis=0)))
    # regularization term
    regularization = lamda * np.sum(W**2)
    # total cost
    J = loss_cross/X.shape[1] + regularization
    
    return J

def computeAccuracy(X, y, W, b):
    # eq 4
    p = EvaluateClassifier(X, W, b)
    predictions = np.argmax(p, axis=0)
    accuracy = np.mean(predictions == y)
    
    return accuracy

def computeGradients(X, y, p, W, lamda):
    # eq 10, 11
    n = X.shape[1]
    C = y.shape[0]
    G = -(y-p)
    grad_W = (G@X.T)/n+2*lamda*W
    grad_b = (G@np.ones(shape=(n,1))/n).reshape(C,1)
    
    return grad_W, grad_b

def MiniBatchGD(X, Y, y, GDparams, W, b, X_val, Y_val, y_val, lamda=0):
    n = X.shape[1]
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']

    W = W.copy()
    b = b.copy()
    X_orig = X.copy()
    Y_orig = Y.copy()
    y_orig = y.copy()

    metrics = {'epochs': [], 'loss_train': [], 'accuracy_train': [], 
                'loss_val': [], 'accuracy_val': []}

    for epoch in range(n_epochs):

        # permute data
        permuted_idx = np.random.permutation(n)
        X = X_orig[:, permuted_idx]
        Y = Y_orig[:, permuted_idx]
        y = [y_orig[i] for i in permuted_idx]

        # iterate batches
        for j in range(n//n_batch):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            batch_idx = range(j_start, j_end)
            X_batch = X[:, batch_idx]
            Y_batch = Y[:, batch_idx]
            y_batch = [y[i] for i in batch_idx]
            
            p = EvaluateClassifier(X_batch, W, b)
            grad_W, grad_b = computeGradients(X_batch, Y_batch, p, W, lamda)
            W += -eta*grad_W
            b += -eta*grad_b

        # compute metrics
        metrics['epochs'].append(epoch)
        metrics['loss_train'].append(computeCost(X, Y, W, b, lamda))
        metrics['accuracy_train'].append(computeAccuracy(X, y, W, b))
        metrics['loss_val'].append(computeCost(X_val, Y_val, W, b, lamda))
        metrics['accuracy_val'].append(computeAccuracy(X_val, y_val, W, b))

    return W, b, metrics

def plot_curves(metrics, title=''):
    # plot the accuracy on train and validation sets in the left subplot and loss in the right subplot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[1].plot(metrics['epochs'], metrics['accuracy_train'], label='train')
    ax[1].plot(metrics['epochs'], metrics['accuracy_val'], label='validation')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Acc')
    ax[1].legend()
    ax[0].plot(metrics['epochs'], metrics['loss_train'], label='train')
    ax[0].plot(metrics['epochs'], metrics['loss_val'], label='validation')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss')
    ax[0].legend()

def plot_weights_by_output_node(W, label_names, title=''):
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(15,2.5), constrained_layout=True)
    plt.suptitle('Visualization of learnt weights'+title)
    
    # Iterate the 10 created subplots (1 for each output node)
    for c, ax in enumerate(axes.flatten()):
        
        # Subset the corresponding output node weights
        image = W[c,:]
        
        # Show the weights in image format
        min_image = min(image)
        max_image = max(image)
        image = (image-min_image)/(max_image-min_image)
        ax.imshow(image.reshape(3,32,32).transpose([1,2,0]))
        ax.axis('off')
        ax.set_title(label_names[c])
    

if __name__ == '__main__':
    # load data
    X_train, Y_train, y_train = LoadBatch('data_batch_1')
    X_val, Y_val, y_val = LoadBatch('data_batch_2')
    X_test, Y_test, y_test = LoadBatch('test_batch')
    # preprocess data
    X_train = preprocess_data(X_train)
    X_val = preprocess_data(X_val)
    X_test = preprocess_data(X_test)
    # initialize weights
    W, b = initialize_weights(X_train.shape[0], Y_train.shape[0])
    # train model

    #1)
    lmb = 0
    GDparams = {'eta': 0.1, 'n_batch': 100, 'n_epochs': 40}
    W, b, metrics = MiniBatchGD(X_train, Y_train, y_train, GDparams, W, b, X_val, Y_val, y_val, lmb)
    # plot curves
    plot_curves(metrics)
    plt.show()
    # compute accuracy on test set
    print('Accuracy on test set (1): {}'.format(computeAccuracy(X_test, y_test, W, b)))

    #2)
    lmb = 0
    GDparams = {'eta': 0.001, 'n_batch': 100, 'n_epochs': 40}
    W, b, metrics = MiniBatchGD(X_train, Y_train, y_train, GDparams, W, b, X_val, Y_val, y_val, lmb)
    # plot curves
    plot_curves(metrics)
    plt.show()
    # compute accuracy on test set
    print('Accuracy on test set (2): {}'.format(computeAccuracy(X_test, y_test, W, b)))

    #3)
    lmb = 0.1
    GDparams = {'eta': 0.001, 'n_batch': 100, 'n_epochs': 40}
    W, b, metrics = MiniBatchGD(X_train, Y_train, y_train, GDparams, W, b, X_val, Y_val, y_val, lmb)
    # plot curves
    plot_curves(metrics)
    plt.show()
    # compute accuracy on test set
    print('Accuracy on test set (3): {}'.format(computeAccuracy(X_test, y_test, W, b)))

    #2)
    lmb = 1
    GDparams = {'eta': 0.001, 'n_batch': 100, 'n_epochs': 40}
    W, b, metrics = MiniBatchGD(X_train, Y_train, y_train, GDparams, W, b, X_val, Y_val, y_val, lmb)
    # plot curves
    plot_curves(metrics)
    plt.show()
    # compute accuracy on test set
    print('Accuracy on test set (4): {}'.format(computeAccuracy(X_test, y_test, W, b)))