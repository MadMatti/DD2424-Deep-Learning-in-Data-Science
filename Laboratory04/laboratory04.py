import numpy as np
import matplotlib.pyplot as plt
import pickle

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
    def __init__(self, d, K, char_list):
        self.m = 100
        self.eta = 0.1
        self.seq_length = 25
        self.d = d
        self.K = K
        self.char_list = char_list
        self.eps = 1e-8

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

    def softmax(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    def synthesize_text(self, x0, h0, n, b, c, W, U, V):
        x = x0
        h = h0
        y = np.zeros((self.K, n))

        for t in range(n):
            p, h, _ = self.forward(x, h, b, c, U, W, V)
            label = np.random.choice(self.K, p=p[:, 0])
            y[label, t] = 1
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
    
    def computeGradsNum(self, X, Y, b, c, W, U, V, h=1e-6):
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
            print("For weights, the % of absolute errors below 1e-6 by layers is:")
            print(np.mean(np.abs(analytic_grads["W"] - numerical_grads["W"]) < 1e-6) * 100)
            print(np.mean(np.abs(analytic_grads["U"] - numerical_grads["U"]) < 1e-6) * 100)
            print(np.mean(np.abs(analytic_grads["V"] - numerical_grads["V"]) < 1e-6) * 100)
            print("and the maximum absolute errors are:")
            print("W: ", np.max(np.abs(analytic_grads["W"] - numerical_grads["W"])))
            print("U: ", np.max(np.abs(analytic_grads["U"] - numerical_grads["U"])))
            print("V: ", np.max(np.abs(analytic_grads["V"] - numerical_grads["V"])))
            print("For biases, the % of absolute errors below 1e-6 by layers is:")
            print(np.mean(np.abs(analytic_grads["b"] - numerical_grads["b"]) < 1e-6) * 100)
            print(np.mean(np.abs(analytic_grads["c"] - numerical_grads["c"]) < 1e-6) * 100)
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



if __name__ == "__main__":
    book_data, book_chars, K = read_file()
    model = RNN(K, K, book_chars)
    model.checkGradients(book_data[:model.seq_length], book_data[1:model.seq_length + 1])
