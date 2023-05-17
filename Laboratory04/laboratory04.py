import numpy as np
import matplotlib.pyplot as plt
from math import ceil

import warnings
warnings.filterwarnings("ignore")

FILE_PATH = "dataset/goblet_book.txt"

def read_file():
    book_data = ''
    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        book_data += line

    book_chars = list(set(book_data))

    return book_data, book_chars, len(book_chars)

def char_to_idx(char, book_chars):
    index = np.zeros((len(book_chars),1), dtype=int)
    index[book_chars.index(char)] = 1
    return index.T

def idx_to_char(index, book_chars):
    return book_chars[np.argmax(index)]


class RNN:
    def __init__(self, d, K, char_list, num_epochs, batch_size, adam):
        self.m = 100
        self.eta = 0.1
        self.seq_length = 25
        self.d = d
        self.K = K
        self.char_list = char_list
        self.eps = 1e-8
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.adam = adam

        if self.adam:
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.m_b = np.zeros((self.m, 1))
            self.m_c = np.zeros((self.K, 1))
            self.m_U = np.zeros((self.m, self.K))
            self.m_W = np.zeros((self.m, self.m))
            self.m_V = np.zeros((self.K, self.m))

        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))
        self.U = np.zeros((self.m, self.K))
        self.W = np.zeros((self.m, self.m))
        self.V = np.zeros((self.K, self.m))

        self.grad_b = np.zeros((self.m, 1))
        self.grad_c = np.zeros((self.K, 1))
        self.grad_U = np.zeros((self.m, self.K))
        self.grad_W = np.zeros((self.m, self.m))
        self.grad_V = np.zeros((self.K, self.m))

        self.m_b = np.zeros((self.m, 1))
        self.m_c = np.zeros((self.K, 1))
        self.m_U = np.zeros((self.m, self.K))
        self.m_W = np.zeros((self.m, self.m))
        self.m_V = np.zeros((self.K, self.m))

        self.h0 = np.zeros((self.m, 1))            

        self.initilization()

    def initilization(self):
        mu = 0
        sigma = 0.01

        self.b = np.zeros(self.b.shape)
        self.c = np.zeros(self.c.shape)
        self.U = np.random.normal(mu, sigma, self.U.shape)
        self.W = np.random.normal(mu, sigma, self.W.shape)
        self.V = np.random.normal(mu, sigma, self.V.shape)

        if self.adam:
            if self.adam:
                self.m_b = np.zeros((self.m, 1))
                self.m_c = np.zeros((self.K, 1))
                self.m_U = np.zeros((self.m, self.K))
                self.m_W = np.zeros((self.m, self.m))
                self.m_V = np.zeros((self.K, self.m))
                self.v_b = np.zeros((self.m, 1))
                self.v_c = np.zeros((self.K, 1))
                self.v_U = np.zeros((self.m, self.K))
                self.v_W = np.zeros((self.m, self.m))
                self.v_V = np.zeros((self.K, self.m))

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def synthesize_text(self, x0, h0, n, b, c, W, U, V):
        x = x0
        h = h0
        y = np.zeros((self.K, n))

        for t in range(n):
            p, h, _ = self.forward(x, h, b, c, U, W, V)
            label = np.random.choice(self.K, p=p[:, 0])
            y[label][t] = 1
            x = np.zeros(x.shape)
            x[label] = 1

        return y

    def forward(self, x, h0, b, c, U, W, V):
        h_t = h0
        h = np.zeros((h0.shape[0], x.shape[1]))
        p = np.zeros((V.shape[0], x.shape[1]))
        a = np.zeros((W.shape[0], x.shape[1]))

        for t in range(x.shape[1]):
            a_t = np.dot(W, h_t) + np.dot(U, x[:, [t]]) + b
            h_t = np.tanh(a_t)
            o_t = np.dot(V, h_t) + c
            p_t = self.softmax(o_t)
            p[:, [t]] = p_t
            h[:, [t]] = h_t
            a[:, [t]] = a_t

        return p, h, a

    def cross_entropy(self, p, y):
        return -np.sum(y * np.log(p + self.eps))

    def computeCost(self, p, y):
        tot_loss = 0
        for i in range(p.shape[1]):
            tot_loss += self.cross_entropy(p[:, [i]], y[:, [i]])
        return tot_loss

    def computeGrads(self, P, X, Y, H, H0, A, V, W):
        G = -(Y.T - P.T).T
        self.grad_V = np.dot(G, H.T)
        self.grad_c = np.sum(G, axis=1, keepdims=True)

        dldh = np.zeros((X.shape[1], self.m))
        dlda = np.zeros((self.m, X.shape[1]))
        dldh[-1] = np.dot(G.T[-1], V)
        dlda[:, -1] = np.dot(dldh[-1].T, np.diag(1 - np.tanh(A[:, -1]) ** 2))

        for t in range(X.shape[1] - 2, -1, -1):
            dldh[t] = np.dot(G.T[t], V) + np.dot(dlda[:, t + 1], W)
            dlda[:, t] = np.dot(dldh[t].T, np.diag(1 - np.tanh(A[:, t]) ** 2))

        self.grad_W = np.dot(dlda, H0.T)
        self.grad_U = np.dot(dlda, X.T)
        self.grad_b = np.sum(dlda, axis=1, keepdims=True)

        # clip gradients
        for grad in [self.grad_W, self.grad_U, self.grad_V, self.grad_b, self.grad_c]:
            np.clip(grad, -5, 5, out=grad)
    
    def computeGradsNum(self, X, Y, b, c, W, U, V, h=1e-4):
        grad_b = np.zeros((self.m, 1))
        grad_c = np.zeros((self.K, 1))
        grad_U = np.zeros((self.m, self.K))
        grad_W = np.zeros((self.m, self.m))
        grad_V = np.zeros((self.K, self.m))

        # grad b
        for i in range(b.shape[0]):
            b_try = np.copy(b)
            b_try[i] -= h
            p, _, _ = self.forward(X, self.h0, b_try, c, U, W, V)
            c1 = self.computeCost(p, Y)
            b_try = np.copy(b)
            b_try[i] += h
            p, _, _ = self.forward(X, self.h0, b_try, c, U, W, V)
            c2 = self.computeCost(p, Y)
            grad_b[i] = (c2 - c1) / (2 * h)

        # grad c
        for i in range(c.shape[0]):
            c_try = np.copy(c)
            c_try[i] -= h
            p, _, _ = self.forward(X, self.h0, b, c_try, U, W, V)
            c1 = self.computeCost(p, Y)
            c_try = np.copy(c)
            c_try[i] += h
            p, _, _ = self.forward(X, self.h0, b, c_try, U, W, V)
            c2 = self.computeCost(p, Y)
            grad_c[i] = (c2 - c1) / (2 * h)

        # grad U
        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                U_try = np.copy(U)
                U_try[i, j] -= h
                p, _, _ = self.forward(X, self.h0, b, c, U_try, W, V)
                c1 = self.computeCost(p, Y)
                U_try = np.copy(U)
                U_try[i, j] += h
                p, _, _ = self.forward(X, self.h0, b, c, U_try, W, V)
                c2 = self.computeCost(p, Y)
                grad_U[i, j] = (c2 - c1) / (2 * h)

        # grad W
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                p, _, _ = self.forward(X, self.h0, b, c, U, W_try, V)
                c1 = self.computeCost(p, Y)
                W_try = np.copy(W)
                W_try[i, j] += h
                p, _, _ = self.forward(X, self.h0, b, c, U, W_try, V)
                c2 = self.computeCost(p, Y)
                grad_W[i, j] = (c2 - c1) / (2 * h)

        # grad V
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                V_try = np.copy(V)
                V_try[i, j] -= h
                p, _, _ = self.forward(X, self.h0, b, c, U, W, V_try)
                c1 = self.computeCost(p, Y)
                V_try = np.copy(V)
                V_try[i, j] += h
                p, _, _ = self.forward(X, self.h0, b, c, U, W, V_try)
                c2 = self.computeCost(p, Y)
                grad_V[i, j] = (c2 - c1) / (2 * h)

        return grad_b, grad_c, grad_V, grad_U, grad_W

    def checkGradients(self, X_chars, Y_chars):
        X = np.zeros((self.d, self.seq_length), dtype=int)
        Y = np.zeros((self.K, self.seq_length), dtype=int)

        for i in range(2):
            self.h0 = np.random.normal(0, 0.01, self.h0.shape)

            for j in range(self.seq_length):
                X[:, j] = char_to_idx(X_chars[j], self.char_list)
                Y[:, j] = char_to_idx(Y_chars[j], self.char_list)

            P, H1, A = self.forward(X, self.h0, self.b, self.c, self.U, self.W, self.V)
            H0 = np.zeros((self.m, self.seq_length))
            H0[:, [0]] = self.h0
            H0[:, 1:] = H1[:, :-1]
            
            self.computeGrads(P, X, Y, H1, H0, A, self.V, self.W)
            grad_b, grad_c, grad_V, grad_U, grad_W = self.computeGradsNum(X, Y, self.b, self.c, self.W, self.U, self.V)

            analytic_grads = {'b': self.grad_b, 'c': self.grad_c, 'U': self.grad_U, 'W': self.grad_W, 'V': self.grad_V}
            numerical_grads = {'b': grad_b, 'c': grad_c, 'U': grad_U, 'W': grad_W, 'V': grad_V}


            # print absolute error between analytical and numerical gradients
            print("For weights, the % of absolute errors below 1e-4 by layers is:")
            print(np.mean(np.abs(analytic_grads["W"] - numerical_grads["W"]) < 1e-4) * 100)
            print(np.mean(np.abs(analytic_grads["U"] - numerical_grads["U"]) < 1e-4) * 100)
            print(np.mean(np.abs(analytic_grads["V"] - numerical_grads["V"]) < 1e-4) * 100)
            print("and the maximum absolute errors are:")
            print("W: ", np.max(np.abs(analytic_grads["W"] - numerical_grads["W"])))
            print("U: ", np.max(np.abs(analytic_grads["U"] - numerical_grads["U"])))
            print("V: ", np.max(np.abs(analytic_grads["V"] - numerical_grads["V"])))
            print("For biases, the % of absolute errors below 1e-4 by layers is:")
            print(np.mean(np.abs(analytic_grads["b"] - numerical_grads["b"]) < 1e-4) * 100)
            print(np.mean(np.abs(analytic_grads["c"] - numerical_grads["c"]) < 1e-4) * 100)
            print("and the maximum absolute errors are:")
            print("b: ", np.max(np.abs(analytic_grads["b"] - numerical_grads["b"])))
            print("c: ", np.max(np.abs(analytic_grads["c"] - numerical_grads["c"])))

            self.m_b += self.grad_b ** 2
            self.m_c += self.grad_c ** 2
            self.m_U += self.grad_U ** 2
            self.m_W += self.grad_W ** 2
            self.m_V += self.grad_V ** 2

            self.b -= self.eta / np.sqrt(self.m_b + self.eps) * self.grad_b
            self.c -= self.eta / np.sqrt(self.m_c + self.eps) * self.grad_c
            self.U -= self.eta / np.sqrt(self.m_U + self.eps) * self.grad_U
            self.W -= self.eta / np.sqrt(self.m_W + self.eps) * self.grad_W
            self.V -= self.eta / np.sqrt(self.m_V + self.eps) * self.grad_V

            self.h0 = H1[:, [-1]]

    def fit(self, book_data):
        N = len(book_data)
        num_seq = N // self.seq_length
        smooth_loss = 0
        iter = 0
        loss_list = []

        for epoch in range(self.num_epochs):
            hprev = np.random.normal(0, 0.01, self.h0.shape)
            index = 0

            # # decay learning rate if not first epoch
            # if epoch > 0: self.eta *= 0.9

            for j in range(num_seq):

                if j==(num_seq-1):
                    X_chars = book_data[index:N-2]
                    Y_chars = book_data[index+1:N-1]
                    index = N
                else:
                    X_chars = book_data[index:index+self.seq_length]
                    Y_chars = book_data[index+1:index+self.seq_length+1]
                    index += self.seq_length

                X = np.zeros((self.d, len(X_chars)), dtype=int)
                Y = np.zeros((self.K, len(Y_chars)), dtype=int)

                for k in range(len(X_chars)):
                    X[:, k] = char_to_idx(X_chars[k], self.char_list)
                    Y[:, k] = char_to_idx(Y_chars[k], self.char_list)

                P, H1, A = self.forward(X, hprev, self.b, self.c, self.U, self.W, self.V)
                H0 = np.zeros((self.m, len(X_chars)))
                H0[:, [0]] = hprev
                H0[:, 1:] = H1[:, :-1]

                self.computeGrads(P, X, Y, H1, H0, A, self.V, self.W)

                loss = self.computeCost(P, Y)
                smooth_loss = 0.999 * smooth_loss + 0.001 * loss if smooth_loss != 0 else loss
                # if iter % 100 == 0:
                loss_list.append(smooth_loss)

                if self.adam:
                    self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * self.grad_b
                    self.m_c = self.beta1 * self.m_c + (1 - self.beta1) * self.grad_c
                    self.m_U = self.beta1 * self.m_U + (1 - self.beta1) * self.grad_U
                    self.m_W = self.beta1 * self.m_W + (1 - self.beta1) * self.grad_W
                    self.m_V = self.beta1 * self.m_V + (1 - self.beta1) * self.grad_V

                    self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * self.grad_b ** 2
                    self.v_c = self.beta2 * self.v_c + (1 - self.beta2) * self.grad_c ** 2
                    self.v_U = self.beta2 * self.v_U + (1 - self.beta2) * self.grad_U ** 2
                    self.v_W = self.beta2 * self.v_W + (1 - self.beta2) * self.grad_W ** 2
                    self.v_V = self.beta2 * self.v_V + (1 - self.beta2) * self.grad_V ** 2

                    m_b_hat = self.m_b / (1 - self.beta1 ** (iter + 1))
                    m_c_hat = self.m_c / (1 - self.beta1 ** (iter + 1))
                    m_U_hat = self.m_U / (1 - self.beta1 ** (iter + 1))
                    m_W_hat = self.m_W / (1 - self.beta1 ** (iter + 1))
                    m_V_hat = self.m_V / (1 - self.beta1 ** (iter + 1))

                    v_b_hat = self.v_b / (1 - self.beta2 ** (iter + 1))
                    v_c_hat = self.v_c / (1 - self.beta2 ** (iter + 1))
                    v_U_hat = self.v_U / (1 - self.beta2 ** (iter + 1))
                    v_W_hat = self.v_W / (1 - self.beta2 ** (iter + 1))
                    v_V_hat = self.v_V / (1 - self.beta2 ** (iter + 1))

                    self.b -= self.eta * m_b_hat / (np.sqrt(v_b_hat) + self.eps)
                    self.c -= self.eta * m_c_hat / (np.sqrt(v_c_hat) + self.eps)
                    self.U -= self.eta * m_U_hat / (np.sqrt(v_U_hat) + self.eps)
                    self.W -= self.eta * m_W_hat / (np.sqrt(v_W_hat) + self.eps)
                    self.V -= self.eta * m_V_hat / (np.sqrt(v_V_hat) + self.eps)

                else:
                    self.m_b += self.grad_b ** 2
                    self.m_c += self.grad_c ** 2
                    self.m_U += self.grad_U ** 2
                    self.m_W += self.grad_W ** 2
                    self.m_V += self.grad_V ** 2

                    # update weights
                    self.b -= self.eta / np.sqrt(self.m_b + self.eps) * self.grad_b
                    self.c -= self.eta / np.sqrt(self.m_c + self.eps) * self.grad_c
                    self.U -= self.eta / np.sqrt(self.m_U + self.eps) * self.grad_U
                    self.W -= self.eta / np.sqrt(self.m_W + self.eps) * self.grad_W
                    self.V -= self.eta / np.sqrt(self.m_V + self.eps) * self.grad_V

                hprev = H1[:, [-1]]

                if iter % 10000 == 0 or iter == 0:
                    print("Iter: ", iter, "Smooth Loss: ", smooth_loss)
                    Y_temp = self.synthesize_text(X[:, [0]], hprev, 200, self.b, self.c, self.W, self.U, self.V)
                    temp_text = ''
                    for t in range(Y_temp.shape[1]):
                        temp_text += idx_to_char(Y_temp[:,[t]], self.char_list)
                    print(temp_text)

                iter += 1

        # Final synthesized text 1000 characters
        Y_temp_final = self.synthesize_text(char_to_idx("H", self.char_list).T, self.h0, 1000, self.b, self.c, self.W, self.U, self.V)
        temp_text_final = ''
        for t in range(Y_temp_final.shape[1]):
            temp_text_final += idx_to_char(Y_temp_final[:,[t]], self.char_list)
        print("Final synthesized text: ")
        print(temp_text_final)

        return loss_list


def plot_learning_curve(smooth_loss, title='', length_text=None, seq_length=None):
    # Plot the learning curve
    _, ax = plt.subplots(1, 1, figsize=(15,5))
    plt.title('Learning curve '+title)
    ax.plot(range(1, len(smooth_loss)+1), smooth_loss)
        
    # Find the optimal metric value and the corresponding update
    optimal_update = np.argmin(smooth_loss)
    optimal_loss = np.round(smooth_loss[optimal_update], 4)
    label = 'Optimal training loss: '+str(optimal_loss)+' at update '+str(optimal_update+1)
    ax.axvline(optimal_update, c='green', linestyle='--', linewidth=1, label=label)
        
    # Plot vertical red lines each epoch (if required)
    if length_text is not None and seq_length is not None:
        updates_per_epoch = len([e for e in range(0, length_text-1, seq_length) \
                                 if e<=length_text-seq_length-1])
        for e in range(updates_per_epoch, len(smooth_loss)+1, updates_per_epoch):
            label = 'Epoch' if e==updates_per_epoch else ''
            ax.axvline(e, c='red', linestyle='--', linewidth=1, label=label)
        
    # Add axis, legend and grid
    ax.set_xlabel('Update step')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)
    plt.show()
    

if __name__ == "__main__":
    book_data, book_chars, K = read_file()
    model = RNN(K, K, book_chars, 2, 10, adam=False)
    # model.checkGradients(book_data[:model.seq_length], book_data[1:model.seq_length + 1])
    loss_list = model.fit(book_data)
    title = 'for 2 epochs, eta = 0.1 and seq_length = 25'
    plot_learning_curve(loss_list, title=title, length_text=len(book_data), seq_length=25)
