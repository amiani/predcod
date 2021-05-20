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

    def test_update_layer_zero_err_no_param_update(self):
        net = PredCodMLP([3,2,1])
        X = np.array([[2.,3]])
        err = np.array([0., 0])
        W = np.array([[4.],[5],[6]])
        next_X = np.array([[10.]])
        X, W, next_err = net._PredCodMLP__update_layer(X, err, W, next_X, False, 1)
        np.testing.assert_array_equal(X, np.array([[-74, -92]]))

    def test_update_layer_no_param_update(self):
        net = PredCodMLP([3,2,1])
        X = np.array([[2.,3]])
        err = np.array([1., 4])
        W = np.array([[4.],[5],[6]])
        next_X = np.array([[10.]])
        X, W, next_err = net._PredCodMLP__update_layer(X, err, W, next_X, False, 1)
        np.testing.assert_array_equal(X, np.array([[-75, -96]]))

    def test_update_layer_param_update(self):
        net = PredCodMLP([3,2,1])
        X = np.array([[2.,3]])
        err = np.array([-10., 4])
        W = np.array([[-2.],[4],[-1]])
        next_X = np.array([[10.]])
        X, W, next_err = net._PredCodMLP__update_layer(X, err, W, next_X, True, 1)
        np.testing.assert_array_equal(W, np.array([[-128], [-227], [-22]]))
    
    def test_train_step_with_backprop(self):
        net = PredCodMLP([3,2,1])
        input = np.array([[1, 0, 1]])
        X = net.add_bias_col(input)
        params = [p.copy() for p in net.params]
        output = np.array([[1]])

        h = np.maximum(0, X.dot(params[0]))
        h = net.add_bias_col(h)
        pred = h.dot(params[1])
        #print('pred: ', pred)
        #print('net pred: ', net.predict(input))
        dpred = pred - output
        dw2 = h.T.dot(dpred)
        #print('dw2: ', dw2)
        dh = dpred.dot(params[1][:-1,:].T)
        dh[h[:,:-1] <= 0] = 0
        dw1 = X.T.dot(dh)
        params[0] += -dw1
        params[1] += -dw2

        pc_params, layers = net._PredCodMLP__train_step(net.params, input, output, 1)
        np.testing.assert_array_almost_equal(pc_params[0], params[0])
        np.testing.assert_array_equal(pc_params[1], params[1])

    def test_train_step_with_backprop2(self):
        N = 100 # number of points per class
        D = 2 # dimensionality
        K = 3 # number of classes
        input = np.zeros((N*K,D)) # data matrix (each row = single example)
        y = np.zeros(N*K, dtype='uint8') # class labels
        for j in range(K):
            ix = range(N*j,N*(j+1))
            r = np.linspace(0.0,1,N) # radius
            t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
            input[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
            y[ix] = j

        net = PredCodMLP([D,100,K])
        X = net.add_bias_col(input)
        params = [p.copy() for p in net.params]
        output = np.array([[1,0,0]])

        h = np.maximum(0, X.dot(params[0]))
        h = net.add_bias_col(h)
        pred = h.dot(params[1])
        dpred = (pred - output) / (N*K)
        dw2 = h.T.dot(dpred)
        dh = dpred.dot(params[1][:-1,:].T)
        dh[h[:,:-1] <= 0] = 0
        dw1 = X.T.dot(dh)
        lr = 0.1
        params[0] += -lr * dw1
        params[1] += -lr * dw2

        pc_params, layers = net._PredCodMLP__train_step(net.params, input, output, lr)
        np.testing.assert_array_almost_equal(pc_params[0], params[0])
        np.testing.assert_array_almost_equal(pc_params[1], params[1])

        h = np.maximum(0, X.dot(params[0]))
        h = net.add_bias_col(h)
        pred = h.dot(params[1])
        #print(pred[1])