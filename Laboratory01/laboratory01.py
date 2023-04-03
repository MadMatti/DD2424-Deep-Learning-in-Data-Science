import itertools
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

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

def MiniBatchGD(X, Y, y, GDparams, W, b=None, lambda_=0, X_val=None, Y_val=None, y_val=None, 
                decay_eta=None, cost="Cross_Entropy"):
    n = X.shape[1]
    eta = GDparams['eta']
    n_batch = GDparams['n_batch']
    n_epochs = GDparams['n_epochs']
    
    # Create a copy of weights and bias to update
    W = W.copy()
    b = b.copy()
    
    # Create a dictionary to store the performance metrics
    metrics = {'epochs':[], 'loss_train':[], 'acc_train':[]}
    if X_val is not None:
        metrics['loss_val'] = []
        metrics['acc_val'] = []
    
    # Iterate epochs
    for epoch in range(n_epochs):        
        
        # Iterate data batches or splits
        for j in range(n//n_batch):
            j_start = j*n_batch
            j_end = (j+1)*n_batch
            inds = range(j_start, j_end)
            X_batch = X[:,inds]
            Y_batch = Y[:,inds]
            y_batch = [y[index] for index in inds]
            
            # Compute gradients and update weights and bias for this batch
            if cost=="Cross_Entropy":
                P_batch = EvaluateClassifier(X_batch, W, b)
                grad_W, grad_b = computeGradients(X_batch, Y_batch, P_batch, W, lambda_)
                W += -eta*grad_W
                b += -eta*grad_b
            elif cost=="BCE":
                P_batch = EvaluateClassifierBCE(X_batch, W, b)
                grad_W, grad_b = computeGradients(X_batch, Y_batch, P_batch, W, lambda_)
                W += -eta*grad_W
                b += -eta*grad_b
        
        # Save the performance metrics of the epoch
        metrics['epochs'].append(epoch+1)
        if cost=="Cross_Entropy":
            metrics['acc_train'].append(computeAccuracy(X, y, W, b))
            metrics['loss_train'].append(computeCost(X, Y, W, b, lambda_))
        elif cost=="BCE":
            metrics['acc_train'].append(ComputeAccuracyBCE(X, y, W, b))
            metrics['loss_train'].append(ComputeCostBCE(X, Y, W, b, lambda_))
        if X_val is not None:
            if cost=="Cross_Entropy":
                metrics['acc_val'].append(computeAccuracy(X_val, y_val, W, b))
                metrics['loss_val'].append(computeCost(X_val, Y_val, W, b, lambda_))
            elif cost=="BCE":
                metrics['acc_val'].append(ComputeAccuracyBCE(X_val, y_val, W, b))
                metrics['loss_val'].append(ComputeCostBCE(X_val, Y_val, W, b, lambda_))  
        
        # Show monitoring message of training
        sys.stdout.write("In epoch "+str(epoch+1)+": loss="+str(metrics['loss_train'][-1])+
                         " and accuracy="+str(metrics['acc_train'][-1])+"\r")
        
        # Decay the learning rate if required
        if decay_eta is not None and (epoch+1)%5==0:
            eta /= decay_eta
    
    if b is not None:
        return W, b, metrics
    else:
        return W, metrics

def plot_curves(metrics, title=''):
    # plot the accuracy on train and validation sets in the left subplot and loss in the right subplot
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(metrics['epochs'], metrics['loss_train'], label='train', color='seagreen')
    ax[0].plot(metrics['epochs'], metrics['loss_val'], label='validation', color='indianred')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Loss')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(metrics['epochs'], metrics['acc_train'], label='train', color='seagreen')
    ax[1].plot(metrics['epochs'], metrics['acc_val'], label='validation', color='indianred')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Accuracy')
    ax[1].legend()
    ax[1].grid(True)

    plt.suptitle("Learning curves "+title)

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

'''Bonus part'''

def new_train_val():
    # Load all data
    images1, labels_oh_1, labels1 = LoadBatch('data_batch_1')
    images2, labels_oh_2, labels2 = LoadBatch('data_batch_2')
    images3, labels_oh_3, labels3 = LoadBatch('data_batch_3')
    images4, labels_oh_4, labels4 = LoadBatch('data_batch_4')
    images5, labels_oh_5, labels5 = LoadBatch('data_batch_5')

    # Stack the data
    images = np.hstack((images1, images2, images3, images4, images5))
    labels_one_hot = np.hstack((labels_oh_1, labels_oh_2, labels_oh_3, labels_oh_4, labels_oh_5))
    labels = labels1 + labels2 + labels3 + labels4 + labels5
    
    # 1000 random samples for validation set 
    indexes_validation = np.random.choice(range(images.shape[1]), 1000, replace=False)
    # divide trianing and validation sets
    images_val = images[:,indexes_validation]
    images_train = np.delete(images, indexes_validation, 1)
    labels_one_hot_val = labels_one_hot[:,indexes_validation]
    labels_one_hot_train = np.delete(labels_one_hot, indexes_validation, 1)
    labels_val = [labels[i] for i in indexes_validation]
    labels_train = [labels[i] for i in range(images.shape[1]) if i not in indexes_validation]

    return images_train, labels_one_hot_train, labels_train, images_val, labels_one_hot_val, labels_val


def grid_search(images_1, labels_one_hot_1, labels_1, images_2, labels_one_hot_2, labels_2, W, b):
    # Define the list of parameters to search
    lambdas = [0, 0.0001, 0.01]
    etas = [0.0001, 0.01, 0.1]
    n_batchs = [10, 100, 200]

    # Iterate all parameters combinations
    grid_search = pd.DataFrame({'lambda':[], 'eta':[], 'n_batch':[], 'loss_train':[], 'loss_val':[]})
    for i, (lambda_, eta, n_batch) in enumerate(list(itertools.product(*[lambdas, etas, n_batchs]))):
        
        # Define the network parameters
        GDparams = {'n_batch':n_batch, 'eta':eta, 'n_epochs':40}
        
        # Train the network
        Wstar, bstar, metrics = MiniBatchGD(images_1, labels_one_hot_1, labels_1, GDparams, W, b, lambda_, 
                                            X_val=images_2, Y_val=labels_one_hot_2, y_val=labels_2)
        
        # Append performance metrics
        grid_search = grid_search.append({'eta':eta, 'lambda':lambda_, 'n_batch':n_batch,
                                        'loss_train':metrics['loss_train'][-1],
                                        'loss_val':metrics['loss_val'][-1]}, ignore_index=True)
    

def sigmoid(x):
    return 1/(1+np.exp(-x))

def EvaluateClassifierBCE(X, W, b):
    S = W@X + b
    P = sigmoid(S)
    
    return P

def ComputeAccuracyBCE(X, y, W, b):
    # Compute the predictions
    S = EvaluateClassifierBCE(X, W, b)
    
    # Compute the accuracy
    acc = np.mean(y==np.argmax(S,0))
    
    return acc

'''Same gradient can be use with BCE loss function'''

def plot_histogram(X, W, b, y):

    # select wrt to the loss function 
    # p = EvaluateClassifier(X, W, b)
    # acc = computeAccuracy(X, y, W, b)
    p = EvaluateClassifierBCE(X, W, b)
    acc = ComputeAccuracyBCE(X, y, W, b)

    prob_correct = [prob for prob in p if prob > 0.5]
    prob_incorrect = [prob for prob in p if prob < 0.5]

    plt.figure(figsize=(10,5))
    
    plt.hist(prob_correct, bins=50, label='correct')
    plt.hist(prob_incorrect, bins=50, label='incorrect')
    plt.legend()
    
    plt.show()





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


    '''Bonus part 1'''

    # a)
    # Stack the training and validation sets and use all data for training

    # call the function to create the new training and validation sets
    images_train, labels_one_hot_train, labels_train, images_val, labels_one_hot_val, labels_val = new_train_val()
    
    # Train the network with the new sets
    W, b = initialize_weights(images_train.shape[0], labels_one_hot_train.shape[0])
    # use parameters second configuration
    lambda_all = 0
    GDparams_all = {'eta': 0.001, 'n_batch': 100, 'n_epochs': 40}
    W_2_all, b_2_all, metrics_2_all = MiniBatchGD(images_train, labels_one_hot_train, labels_train, 
                                                      GDparams_all, W, b, lambda_all, X_val=images_val, 
                                                      Y_val=labels_one_hot_val, y_val=labels_val)

    plot_curves(metrics_2_all)

    # c) 
    # Grid search
    grid_search_res = grid_search(images_train, labels_one_hot_train, labels_train, images_val, labels_one_hot_val, labels_val, W, b)
    grid_search_res.sort_values(['loss_val'], inplace=True)
    print(grid_search_res)

    # d)
    # Learning rate decay
    GD_params_decay = {'n_batch':100, 'eta':0.1, 'n_epochs':40}
    W_2_decay, b_2_decay, metrics_2_decay = MiniBatchGD(images_train, labels_one_hot_train, labels_train,
                                                        GD_params_decay, W, b, lambda_all, X_val=images_val,
                                                        Y_val=labels_one_hot_val, y_val=labels_val, decay=10)

    # Plot the learning curves
    plot_curves(metrics_2_decay)

    '''Bonus part 2'''

    Wstar_BCE, bstar_BCE, metrics_BCE = MiniBatchGD(X_train, Y_train, y_train, GDparams, W, b, lambda_all,
                                                X_val=X_val, Y_val=Y_val, y_val=y_val, cost='BCE')

    plot_curves(metrics_BCE, title='BCE, C2')
    print("With BCE the test accuracy is:")
    print(str(ComputeAccuracyBCE(X_train, y_test, Wstar_BCE, bstar_BCE)*100)+'%\n')
    plot_histogram(X_train, y_train, Wstar_BCE, bstar_BCE)