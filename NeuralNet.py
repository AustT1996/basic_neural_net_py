"""
A neural net class in python
"""

__author__ = "Austin Tripp"

import numpy as np

class NeuralNet:
    """
    Class to hold a neural net
    """
    
    def __init__(self, input_size, layer_sizes, act_funcs, w_init=None):
        self.input_size = input_size
        self.layers = []
        self.act_funcs = []
        self.biases = []
        
        last_size = input_size
        for ls, af in zip(layer_sizes, act_funcs):
        
            layer_matrix_shape = (ls, last_size)
            if w_init is None:
                self.layers.append(np.random.normal(0., 1., layer_matrix_shape))  # TODO STD NORMALIZATION?
            else:
                self.layers.append(w_init(layer_matrix_shape))
            
            self.act_funcs.append(NeuralNet.ACT_FUNC_DICT[af])
            self.biases.append(np.ones((ls, 1)))
                
            last_size = ls
            
    def eval(self, inputs):
        last_M = inputs
        for l, b, a in zip(self.layers, self.biases, self.act_funcs):
            x = l @ last_M + b
            last_M = a(x)
        
        return last_M
                
    def eval_sup(self, inputs, correct_outputs):
        """
        Inputs are along rows
        """
        
        assert inputs.shape[0] == self.input_size
        # Keep track of all outputs anyways
        outputs = []
        
        last_M = inputs
        for l, b, a in zip(self.layers, self.biases, self.act_funcs):
            x = l @ last_M + b
            last_M = a(x)
            outputs.append(last_M)
        
        # Error
        error = 1./2. *(last_M - correct_outputs)**2
        error = np.sum(error, axis=0)
        error = np.average(error, axis=-1)
        
        return error, inputs, correct_outputs, outputs
        
    def gradient_backprop(self, inputs, correct_outputs, outputs):
    
        if inputs.shape[1] > 1:
            raise ValueError("Not equipped for more than 1 example. TODO!")
        
        # This is what gets returned
        grads_w = []
        grads_b = []
        
        # Keep track of the last one
        last_l = None
        last_b = None
        last_a = None
        last_o = None
        do_by_dx = None
        for l, b, a, o in reversed(list(zip(self.layers, self.biases, self.act_funcs, outputs))):
        
            # Complete the last derivative
            if do_by_dx is not None:
                grads_w.append(dE_by_dout * do_by_dx * o.T)
                grads_b.append(dE_by_dout * do_by_dx)
            
            
            if last_l is None:
                # This is the first pass
                dE_by_dout = o - correct_outputs
            else:
            
                # Long procedure to calculate from previous
                prev_deriv = NeuralNet.ACT_FUNC_Y_DERIVS[last_a]
                dE_by_dout = dE_by_dout * prev_deriv(last_o)
                dE_by_dout = dE_by_dout.reshape(dE_by_dout.T.shape + (1,))
                last_l_repeat = np.tile(last_l.reshape(1, *last_l.shape), (last_o.shape[1], 1, 1))
                dE_by_dout = dE_by_dout * last_l_repeat
                dE_by_dout = np.sum(dE_by_dout, axis=1).T
                
            do_by_dx = NeuralNet.ACT_FUNC_Y_DERIVS[a](o)
            
            # Store the last results
            last_l = l
            last_b = b
            last_a = a
            last_o = o       
        
        # Final gradient
        grads_w.append(dE_by_dout * do_by_dx * inputs.T)
        grads_b.append(dE_by_dout * do_by_dx)
        
        # Reverse the list
        grads_w = list(reversed(grads_w))
        grads_b = list(reversed(grads_b))
        
        return grads_w, grads_b
                
    def train_step(self, inputs, correct_outputs, learning_rate):
        grads_list = []
        N = correct_outputs.shape[1]
        total_error = 0.
        for i in range(N):
            inputs_i = inputs[:, i:i+1]
            correct_outputs_i = correct_outputs[:, i:i+1]
            error, _, _, outputs = self.eval_sup(inputs_i, correct_outputs_i)
            grads_list.append(self.gradient_backprop(inputs_i, correct_outputs_i, outputs))
            total_error += error
        total_error /= N  # Average error
            
        # Apply the gradient descent
        total_grads_w, total_grads_b = grads_list[0]
        for i in range(N):
            total_grads_w = [old+new for old, new in zip(total_grads_w, grads_list[i][0])]
            total_grads_b = [old+new for old, new in zip(total_grads_b, grads_list[i][1])]
            
        for l, b, dw, db in zip(self.layers, self.biases, total_grads_w, total_grads_b):
            l -= learning_rate * dw / N
            b -= learning_rate * db / N
            
        return total_error
    
    def linear(x):
        return x
        
    def linear_deriv(y):
        return 1.
        
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))
    
    def sigmoid_deriv(y):
        return y * (1-y) 
    
    ACT_FUNC_DICT = {"sigmoid": sigmoid,
                     "linear": linear}
    ACT_FUNC_Y_DERIVS = {sigmoid: sigmoid_deriv,
                         linear: linear_deriv}