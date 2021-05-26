import numpy as np
from typing import Tuple

def adam(w: np.ndarray, dw: np.ndarray, config=None) -> Tuple[np.ndarray, dict]:
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    learning_rate = config['learning_rate']
    beta1 = config['beta1']
    beta2 = config['beta2']
    eps = config['epsilon']
    m = config['m']
    v = config['v']
    t = config['t'] + 1

    # t is your iteration counter going from 1 to infinity
    m = beta1*m + (1-beta1)*dw
    mt = m / (1-beta1**t)
    v = beta2*v + (1-beta2)*(dw**2)
    vt = v / (1-beta2**t)
    next_w = w - learning_rate * mt / (np.sqrt(vt) + eps)

    config['m'] = m
    config['v'] = v
    config['t'] = t

    return next_w, config