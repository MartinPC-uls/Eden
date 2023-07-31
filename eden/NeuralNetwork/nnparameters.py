from eden import _clib, _Matrix
from ctypes import *

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