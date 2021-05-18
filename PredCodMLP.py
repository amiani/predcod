import numpy as np
from math import sqrt
from numpy.random import default_rng
from typing import Tuple

class PredCodMLP:
    def __init__(self, layer_dims: list[int]):
        self.params = []
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        rng = default_rng()
        for i, d in enumerate(layer_dims[:-1]):
            next_d = layer_dims[i+1]
            w = rng.standard_normal((d, next_d)) * sqrt(2.0/d)
            W = np.zeros((d+1, next_d))
            W[:-1,:] = w
            self.params.append(W)

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.__predict(self.params, input)[0]

    def __predict(self, params: list[np.ndarray], input: np.ndarray) -> Tuple[np.ndarray, list[np.ndarray]]:
        layers = []
        X = self.add_bias_col(input)
        layers.append(X)
        layers.append(X.dot(params[0]))
        for W in params[1:]:
            activated = np.maximum(0, layers[-1])
            h = self.add_bias_col(activated)
            layers.append(h.dot(W))
        return layers[-1], layers

    def train_step(self, input: np.ndarray, output: np.ndarray, lr=0.01):
        self.__train_step(self.params, input, output, lr)

    def __train_step(self, params: list[np.ndarray], input: np.ndarray, output: np.ndarray, lr=0.01) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        preds, layers = self.__predict(params, input)
        #print("train_step prediction", preds[1])
        layers[-1] = output
        X = self.add_bias_col(input)
        for t in range(self.num_layers):
            curr_mu = X.dot(params[0])
            curr_err = layers[1] - curr_mu
            if t == self.num_layers - 1:
                params[0] += lr * X.T.dot(curr_err)
            for i in range(1, self.num_layers - 1):
                activated = np.maximum(0, layers[i])
                h = self.add_bias_col(activated)
                next_mu = h.dot(params[i])
                next_err = layers[i+1] - next_mu
                relu_mask = layers[i] > 0
                layers[i] += -curr_err + relu_mask * next_err.dot(params[i].T[:,:-1])
                #can we avoid recalculating curr_mu during learning?
                activated = np.maximum(0, layers[i])
                h = self.add_bias_col(activated)
                curr_mu = h.dot(params[i])
                curr_err = layers[i+1] - curr_mu
                if self.num_layers - 1 - i == t:
                    print('h', h)
                    print('curr_err', curr_err)
                    print('h.T.curr_err', h.T.dot(curr_err))
                    params[i] += lr * h.T.dot(curr_err)
        return params, layers
    
    def __update_layer(self, X: np.ndarray, err: np.ndarray, W: np.ndarray, next_X: np.ndarray, update_W: bool, lr=.01) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        activated = np.maximum(0, X)
        h = self.add_bias_col(activated)
        next_mu = h.dot(W)
        next_err = next_X - next_mu
        relu_mask = X > 0
        X += -err + relu_mask * next_err.dot(W.T[:,:-1])
        #can we avoid recalculating curr_mu during learning?
        activated = np.maximum(0, X)
        h = self.add_bias_col(activated)
        curr_mu = h.dot(W)
        curr_err = next_X - curr_mu
        if update_W:
            print('h', h)
            print('curr_err', curr_err)
            print('h.T.curr_err', h.T.dot(curr_err))
            W += lr * h.T.dot(curr_err)
        return X, W, next_err
    
    def add_bias_col(self, X: np.ndarray) -> np.ndarray:
        N, D = X.shape
        biased = np.ones((N, D+1))
        biased[:,:-1] = X
        return biased