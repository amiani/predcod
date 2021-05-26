import numpy as np
from math import sqrt
from numpy.random import default_rng
from typing import Tuple, List

from adam import adam

class PredCodMLP:
    def __init__(self, layer_dims: List[int]):
        self.params = []
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims)
        self.optim_configs = []
        rng = default_rng()
        for i, d in enumerate(layer_dims[:-1]):
            next_d = layer_dims[i+1]
            w = rng.standard_normal((d, next_d)) * sqrt(2.0/d)
            W = np.zeros((d+1, next_d))
            W[:-1,:] = w
            self.params.append(W)
            self.optim_configs.append({})

    def predict(self, input: np.ndarray) -> np.ndarray:
        return self.__predict(self.params, input)[0]

    def __predict(self, params: List[np.ndarray], input: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        layers = []
        X = add_bias_col(input)
        layers.append(X)
        layers.append(X.dot(params[0]))
        for W in params[1:]:
            activated = np.maximum(0, layers[-1])
            h = add_bias_col(activated)
            layers.append(h.dot(W))
        return layers[-1], layers
    
    def train_step(self, input: np.ndarray, output: np.ndarray, lr=0.01):
        self.__train_step(self.params, input, output, self.optim_configs, lr)

    def __train_step(self, params: List[np.ndarray], input: np.ndarray, output: np.ndarray, optim_configs: List[dict], lr=0.01) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        scores, layers = self.__predict(params, input)
        #layers[-1] = output
        X = add_bias_col(input)
        for t in range(self.num_layers-1):
            curr_mu = X.dot(params[0])
            curr_err = layers[1] - curr_mu
            if t == self.num_layers - 2:
                N = X.shape[0]
                dW = X.T.dot(curr_err) / N
                params[0], optim_configs[0] = adam(params[0], dW, optim_configs[0])
            """
            for i in range(1, self.num_layers - 2):
                update_W = t == self.num_layers - 2 - i
                #print(t, i)
                config = optim_configs[i]
                layers[i], params[i], curr_err, optim_configs[i] = self.__update_layer(layers[i], curr_err, params[i], layers[i+1], update_W, config, lr)
            """
            update_W = t == 0
            loss, dscores = cross_entropy_loss(scores, output)
            layers[1], params[1], optim_configs[1] = self.__update_final_layer(layers[1], curr_err, params[1], dscores, update_W, optim_configs[1], lr)
        return params, layers
    
    def __update_layer(self, X: np.ndarray, err: np.ndarray, W: np.ndarray, next_X: np.ndarray, update_W: bool, config=None, lr=.01) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        h = add_bias_col(np.maximum(0, X))
        next_mu = h.dot(W)
        next_err = next_X - next_mu
        #print(np.sum(next_err), '\n')
        relu_mask = X > 0
        #print(-err, next_err.dot(W.T[:,:-1]))
        X += -err + relu_mask * next_err.dot(W.T[:,:-1])
        if update_W:
            N = X.shape[0]
            dW = h.T.dot(next_err) / N
            W, config = adam(W, dW, config)
        h = add_bias_col(np.maximum(0, X))
        next_mu = h.dot(W)
        next_err = next_X - next_mu
        return X, W, next_err, config
    
    def __update_final_layer(self, X:np.ndarray, err: np.ndarray, W: np.ndarray, next_err: np.ndarray, update_W: bool, config=None, lr=0.1) \
        -> Tuple[np.ndarray, np.ndarray, dict]:
        h = add_bias_col(np.maximum(0, X))
        relu_mask = X > 0
        X += -err + relu_mask * next_err.dot(W.T[:,:-1])
        if update_W:
            N = X.shape[0]
            dW = h.T.dot(next_err) / N
            W, config = adam(W, dW, config)
        return X, W, config

    
def add_bias_col(X: np.ndarray) -> np.ndarray:
    N, D = X.shape
    biased = np.ones((N, D+1))
    biased[:,:-1] = X
    return biased

def softmax(X: np.ndarray) -> np.ndarray:
    shift_X = X - np.max(X, 1, keepdims=True)
    exp_X = np.exp(shift_X)
    return exp_X / np.sum(exp_X, 1, keepdims=True)

def cross_entropy_loss(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = scores.shape[0]
    probs = softmax(scores)
    loss = np.mean(-np.log(probs[range(N),y]))
    dscores = probs.copy()
    dscores[range(N),y] -= 1
    return loss, dscores