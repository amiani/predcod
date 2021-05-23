import numpy as np
#import jax.numpy as jnp
from typing import Tuple, Union, Dict

from fast_layers.fast_layers import conv_forward_fast, conv_backward_fast, max_pool_forward_fast, max_pool_backward_fast

class CNN:
    def __init__(   self,
                    batch_size: int, C:int, H: int, W: int,
                    num_filters: int, num_classes: int):
        self.W1 = np.random.normal(scale=2/np.sqrt(25), size=(num_filters, C, 5, 5))
        self.b1 = np.zeros(num_filters)
        num_conv_weights = int(num_filters * (H/2)**2)
        self.W2 = np.random.normal(scale=2/np.sqrt(num_conv_weights), size=(num_conv_weights, num_classes))
        self.b2 = np.zeros(num_classes)
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        h1, cache1 = conv_relu_forward(X, self.W1, self.b1, self.W1.shape[0])
        pool_param = { 'pool_height': 2, 'pool_width': 2, 'stride': 2}
        h2, cache2 = max_pool_forward_fast(h1, pool_param)
        N, D, H, W = h2.shape
        scores, cache3 = affine_forward(h2.reshape(N, -1), self.W2, self.b2)

        cache = (cache1, cache2, cache3)
        return scores, cache

    def loss(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        scores, cache = self.forward(X)
        loss, dscores = softmax_loss(scores, y)
        return loss, dscores, cache


    def backward(self, dscores: np.ndarray, cache: Tuple) -> Dict[str, np.ndarray]:
        return {}

def conv_relu_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray, num_filters: int) -> Tuple[np.ndarray, Tuple]:
    N, C, H, width = X.shape
    F, D, HH, WW = W.shape
    pad = int((HH-1)/2)
    conv_param = { 'stride': 1, 'pad': pad }
    h, conv_cache = conv_forward_fast(X, W, b, conv_param)
    a, relu_cache = relu_forward(h)
    return a, (conv_cache, relu_cache)

def relu_forward(X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    A = np.maximum(0, X)
    return A, (X > 0)

def affine_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    h = X.dot(W) + b
    return h, (X, W, b)

def softmax_loss(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N, K = scores.shape
    scores -= np.max(scores, 1)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, 1)[:,np.newaxis]
    loss = np.mean(-np.log(probs[y]), 0)

    dscores = probs.copy()
    dscores[range(N),y] -= 1
    return loss, dscores