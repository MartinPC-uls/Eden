#from eden import _clib, _Matrix
from ctypes import *
import eden

# C Version, could be optimal but I couldn't fix the memory issues.
#class Parameters:
#    def __init__(self, parameters: list):
#        size = len(parameters)
#        c_array = (POINTER(_Matrix) * size)(*parameters)
#        print(c_array)
#        self._p = _clib.init_parameters(c_array, size)
#        #self.size = _clib.get_nparameters(self.p)
#        
#    def update(self, lr=0.01):
#        _clib.update_parameters(self._p, lr)

class Parameters:
    def __init__(self, *args):
        self.parameters = []
        self.total = 0
        self.append(*args)
        
    def append(self, *args):
        for matrix in args:
            self.parameters.append(matrix)
            self.total += matrix.c * matrix.r
        
    def update(self, lr=0.01):
        for parameter in self.parameters:
            parameter.update(lr)
            
    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()
            
class Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.dataset = zip(inputs, targets)

class MLP:
    def __init__(self, parameters, *layers):
        self.parameters = parameters
        self.layers = []
        for layer in layers:
            self.layers.append(layer)
        
    def forward(self, input):
        for layer in self.layers:
            input = layer(input)
        return input
    
    def train(self, epochs, lr, dataset: Dataset):
        for e in range(epochs):
            total_loss = eden.zeros(1,1)
            for input, target in zip(dataset.inputs, dataset.targets):
                predicted = self.forward(input)
                loss = (target - predicted)**2
                total_loss += loss
                #print(loss)
            
            total_loss = eden.Matrix(1.0) / eden.Matrix(len(dataset.inputs)) * total_loss
            self.parameters.zero_grad()
            total_loss.backward()
            self.parameters.update(lr)
            
            if e % 100 == 0:
                print(f"Epoch {(e+1)}/{epochs} completed - loss: {total_loss}")