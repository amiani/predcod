import numpy as np
from typing import Tuple

class CNN:
    def __init__(   self,
                    N: int, H: int, W: int,
                    num_filters: int, num_classes: int):
        self.W1 = np.random.normal(scale=2/np.sqrt(25), size=(num_filters,5,5))
        self.b1 = np.zeros(num_filters)
        num_conv_weights = num_filters * 25
        self.W2 = np.random.normal(scale=2/np.sqrt(num_conv_weights), size=(num_conv_weights, num_classes))
        self.b2 = np.zeros(num_classes)
    
    def forward(self, X: np.ndarray, y=None) -> Tuple[np.ndarray, ]:
        h1, cache1 = conv_relu_forward(X, self.W1, self.b1)

def conv_relu_forward(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    pass