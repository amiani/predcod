import unittest
import numpy as np

from PredCodMLP import PredCodMLP

class PredCodMLPTest(unittest.TestCase):
    def test_predict_all_zeros(self):
        net = PredCodMLP([3,2,1])
        input = np.array([  [0,0,0],
                            [0,0,0]])
        preds, layers = net._PredCodMLP__predict(net.params, input)
        np.testing.assert_array_equal(preds, np.zeros((2,1)))
    
    def test_predict_ints(self):
        net = PredCodMLP([3,2,1])
        input = np.array([  [1,2,3],
                            [4,5,6]])
        params = [np.array([[1,2],
                            [3,4],
                            [5,6],
                            [0,0]]),
                np.array([[6],[7],[0]])]
        preds, layers = net._PredCodMLP__predict(params, input)
        np.testing.assert_array_equal(preds, np.array([[328],[742]]))

    def test_predict_negative_out(self):
        net = PredCodMLP([3,2,1])
        input = np.array([  [1,2,3],
                            [4,5,6]])
        params = [np.array([[1,2],
                            [3,4],
                            [5,6],
                            [0,0]]),
                np.array([[-6],[-7],[0]])]
        preds, layers = net._PredCodMLP__predict(params, input)
        np.testing.assert_array_equal(preds, np.array([[-328],[-742]]))

    def test_train_step_same_update_as_backprop(self):
        net = PredCodMLP([3,2,1])
        input = np.array