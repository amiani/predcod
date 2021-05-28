import numpy as np
from jax import random
from typing import Tuple, Union, Dict

from fast_layers.fast_layers import conv_forward_fast, conv_backward_fast, max_pool_forward_fast, max_pool_backward_fast

class CNN:
    def __init__(   self,
                    batch_size: int, C:int, H: int, W: int,
                    num_filters: int, num_classes: int, keys):
        self.params = {}
        self.params['W1'] = np.sqrt(2/25) * random.normal(keys[0], (num_filters, C, 5, 5))
        self.params['b1'] = np.zeros(num_filters)
        num_conv_weights = int(num_filters * (H/2)**2)
        self.params['W2'] = np.sqrt(2/num_conv_weights) * random.normal(keys[1], (num_conv_weights, num_classes))
        self.params['b2'] = np.zeros(num_classes)
    
def forward(params: dict[str, np.ndarray], X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    h1, cache1 = conv_relu_forward(X, params['W1'], params['b1'], params['W1'].shape[0])
    pool_param = { 'pool_height': 2, 'pool_width': 2, 'stride': 2}
    h2, cache2 = max_pool_forward_fast(h1, pool_param)
    N, D, H, W = h2.shape
    scores, cache3 = affine_forward(h2.reshape(N, -1), params['W2'], params['b2'])

    cache = (cache1, cache2, cache3)
    return scores, cache

def loss(params: dict[str,np.ndarray], X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple]:
    scores, cache = forward(params, X)
    loss, dscores = softmax_loss(scores, y)
    return loss, dscores, cache


def backward( dscores: np.ndarray, cache: Tuple) -> Dict[str, np.ndarray]:
    grads = {}
    conv_relu_cache, pool_cache, affine_cache = cache
    dh2, dW2, db2 = affine_backward(dscores, affine_cache)
    grads['W2'] = dW2
    grads['b2'] = db2
    #TODO: determine reshape dims more elegantly (cache them?)
    N, C, H, width = conv_relu_cache[0][0].shape
    dh1 = max_pool_backward_fast(dh2.reshape(N,25,int(H/2),int(H/2)), pool_cache)
    dX, dW1, db1 = conv_relu_backward(dh1, conv_relu_cache)
    grads['W1'] = dW1
    grads['b1'] = db1
    #print(grads['W1'])
    return grads

def conv_relu_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray, num_filters: int) -> Tuple[np.ndarray, Tuple]:
    N, C, H, width = X.shape
    F, D, HH, WW = W.shape
    pad = int((HH-1)/2)
    conv_param = { 'stride': 1, 'pad': pad }
    h, conv_cache = conv_forward_fast(X, W, b, conv_param)
    a, relu_cache = relu_forward(h)
    return a, (conv_cache, relu_cache)

def conv_relu_backward(dup: np.ndarray, cache: Tuple[Tuple, Tuple[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    conv_cache, relu_cache = cache
    dH = relu_backward(dup, relu_cache)
    dX, dW, db = conv_backward_fast(dH, conv_cache)
    return dX, dW, db

def relu_forward(X: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    A = np.maximum(0, X)
    return A, (X > 0)

def relu_backward(dup: np.ndarray, cache: Tuple[np.ndarray]) -> np.ndarray:
    mask = cache[0]
    return dup * mask

def affine_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    h = X.dot(W) + b
    return h, (X, W, b)

def affine_backward(dup: np.ndarray, cache: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, W, b = cache
    dx = dup.dot(W.T)
    dW = X.T.dot(dup)
    db = np.sum(dup, 0)
    return dx, dW, db

def softmax_loss(scores: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
    N, K = scores.shape
    scores -= np.max(scores, 1)[:, np.newaxis]
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, 1, keepdims=True)
    #print(probs)
    loss = np.mean(-np.log(probs[range(N),y]))

    dscores = probs.copy()
    dscores[range(N),y] -= 1
    dscores /= N
    return loss, dscores