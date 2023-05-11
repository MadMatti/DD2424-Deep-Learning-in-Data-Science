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
    print(len(book_chars))

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
        self.seq_legth = 25
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

    def forward(self, x, h0, b, c, U, W, V):
        h_t = h0
        h = np.zeros((h0.shape[0], x.shape[1]))
        p = np.zeros((V.shape[0], x.shape[1]))
        a = np.zeros((W.shape[0], x.shape[1]))

        for t in range(x.shape[1]):
            a_t = np.dot(W, h_t) + np.dot(U, x[:, t]) + b
            h_t = np.tanh(a_t)
            o_t = np.dot(V, h_t) + c
            p_t = self.softmax(o_t)
            p[:, t] = p_t.reshape(-1)
            h[:, t] = h_t.reshape(-1)
            a[:, t] = a_t.reshape(-1)

        return p, h, a

    





if __name__ == "__main__":
    read_file()
