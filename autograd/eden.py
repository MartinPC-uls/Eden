import ctypes
from typing import Any

class _Tensor(ctypes.Structure):
    _fields_ = [
        ('requires_grad', ctypes.c_bool),
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('grad', ctypes.POINTER(ctypes.c_float)),
        ('data_size', ctypes.c_int),
        ('nthreads', ctypes.c_int),
        ('test_value', ctypes.c_int)
    ]
    
class Point(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int),
                ("y", ctypes.c_int)]

_clib = ctypes.CDLL('./tensor.so', winmode=0)

_clib.get_threads.argtypes = None
_clib.get_threads.restype = ctypes.c_int

_clib.tensor_float32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.tensor_float32.restype = ctypes.POINTER(_Tensor)

_clib.tensor_float32_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.tensor_float32_init.restype = ctypes.POINTER(_Tensor)

_clib.tensor_float32_object.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_clib.tensor_float32_object.restype = ctypes.c_void_p

_clib.get_tensor_data.argtypes = [ctypes.c_void_p]
_clib.get_tensor_data.restype = ctypes.POINTER(ctypes.c_float)

_clib.get_tensor_grad.argtypes = [ctypes.c_void_p]
_clib.get_tensor_grad.restype = ctypes.POINTER(ctypes.c_float)

_clib.get_tensor_requieres_grad = [ctypes.c_void_p]
_clib.get_tensor_requieres_grad = ctypes.c_bool

_clib.delete_tensor.argtypes = [ctypes.c_void_p]
_clib.delete_tensor.restype = None

_clib.get_tensor_size.argtypes = [ctypes.c_void_p]
_clib.get_tensor_size.restype = ctypes.c_int

_clib.add.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(_Tensor)]
_clib.add.restype = ctypes.POINTER(_Tensor)

_clib.sub.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(_Tensor)]
_clib.sub.restype = ctypes.POINTER(_Tensor)

_clib.mul.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(_Tensor)]
_clib.mul.restype = ctypes.POINTER(_Tensor)

_clib.dot.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(_Tensor)]
_clib.dot.restype = ctypes.POINTER(_Tensor)

_clib.divv.argtypes = [ctypes.POINTER(_Tensor), ctypes.POINTER(_Tensor)]
_clib.divv.restype = ctypes.POINTER(_Tensor)

_clib.poww.argtypes = [ctypes.POINTER(_Tensor), ctypes.c_float]
_clib.poww.restype = ctypes.POINTER(_Tensor)

_clib.relu.argtypes = [ctypes.POINTER(_Tensor)]
_clib.relu.restype = ctypes.POINTER(_Tensor)

_clib.sigmoid.argtypes = [ctypes.POINTER(_Tensor)]
_clib.sigmoid.restype = ctypes.POINTER(_Tensor)

_clib.tanhh.argtypes = [ctypes.POINTER(_Tensor)]
_clib.tanhh.restype = ctypes.POINTER(_Tensor)

_clib.expp.argtypes = [ctypes.POINTER(_Tensor)]
_clib.expp.restype = ctypes.POINTER(_Tensor)

_clib.backward.argtypes = [ctypes.POINTER(_Tensor)]
_clib.backward.restype = None

_clib.full.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.full.restype = ctypes.POINTER(_Tensor)

_clib.ones.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.ones.restype = ctypes.POINTER(_Tensor)

_clib.zeros.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.zeros.restype = ctypes.POINTER(_Tensor)

_clib._randn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool]
_clib._randn.restype = ctypes.POINTER(_Tensor)

_clib._random.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_clib._random.restype = ctypes.POINTER(_Tensor)

_clib.shuffle.argtypes = [ctypes.POINTER(_Tensor), ctypes.c_int]
_clib.shuffle.restype = ctypes.POINTER(_Tensor)

_clib.print_tensor.argtypes = [ctypes.POINTER(_Tensor), ctypes.c_bool, ctypes.c_bool]
_clib.print_tensor.restype = ctypes.c_char_p

_clib.release_int_memory.argtypes = [ctypes.c_void_p]
_clib.release_int_memory.restype = None

_clib.release_float32_memory.argtypes = [ctypes.c_void_p]
_clib.release_float32_memory.restype = None

_clib.release_float64_memory.argtypes = [ctypes.c_void_p]
_clib.release_float64_memory.restype = None

_clib.release_char_memory.argtypes = [ctypes.c_void_p]
_clib.release_char_memory.restype = None

_clib.get_data_size.argtypes = [ctypes.POINTER(_Tensor)]
_clib.get_data_size.restype = ctypes.c_int

_clib.get_requires_grad.argtypes = [ctypes.POINTER(_Tensor)]
_clib.get_requires_grad.restype = ctypes.c_bool

_clib.get_nthreads.argtypes = [ctypes.POINTER(_Tensor)]
_clib.get_nthreads.restype = ctypes.c_int

_GLOBAL_SEED = 0

def get_threads():
    return _clib.get_threads()

_GLOBAL_THREADS = get_threads()-1

def set_threads(threads):
    global _GLOBAL_THREADS
    available_threads = get_threads()
    if threads > available_threads:
        print(f"WARNING: You're exceeding the maximum number of threads on your CPU!.\nCPU Threads: {available_threads}")
    _GLOBAL_THREADS = threads

def delete(tensor):
    if tensor.alive:
        _clib.delete_tensor(tensor._t)
        tensor._alive = False
    else:
        print(f'Tensor is not alive. (operation: delete, data: {tensor.data})')
        
def manual_seed(seed):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

def get_seed():
    global _GLOBAL_SEED
    return _GLOBAL_SEED


def randn(size, mean = 0, SD = 1, spare = False):
    # 'randn' uses "standard normal distribution" to generate numbers.
    # If 'spare' is set to False, it will use cos function internally to generate a number,
    # otherwise it will use sin function.
    seed = get_seed()
    
    return Tensor(_clib._randn(size, seed, mean, SD, spare))

def random(size, lower_bound = 0, upper_bound = 0x7FFF):
    seed = get_seed()
    return Tensor(_clib._random(size, seed, lower_bound, upper_bound))

def shuffle(tensor):
    seed = get_seed()
    return Tensor(_clib.shuffle(tensor._t, seed))

def full(data, size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Tensor(_clib.full(data, size, threads, requires_grad))

def tensor(data, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Tensor(_clib.tensor_float32_init((ctypes.c_float * len(data))(*data), len(data), _GLOBAL_THREADS, requires_grad))

def ones(size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Tensor(_clib.ones(size, threads, requires_grad))

def zeros(size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Tensor(_clib.zeros(size, threads, requires_grad))
    
class Tensor:
    def __init__(self, data, requires_grad=False, dtype='float32'):
        global _GLOBAL_THREADS
        if isinstance(data, ctypes.POINTER(_Tensor)):
            self._alive = True
            #self.data_size = data.contents.data_size
            #self.requires_grad = data.contents.requires_grad
            self.data_size = _clib.get_data_size(data)
            self.requires_grad = _clib.get_requires_grad(data)
            self._t = data
        else:
            if dtype == 'float32':
                self._alive = True
                self.data_size = len(data)
                self.requires_grad = requires_grad
                #print("[TENSOR]", self.requires_grad)
                self._t = _clib.tensor_float32_init((ctypes.c_float * self.data_size)(*data), self.data_size,
                                               _GLOBAL_THREADS, requires_grad) # _Tensor
                #print("[TENSORAfter]", self._t.contents.requires_grad)
            
    def __repr__(self):
        return _clib.print_tensor(self._t, False, False).decode()
    
    def __add__(self, other):
        if other.data_size > self.data_size:
            return Tensor(_clib.add(other._t, self._t))
        return Tensor(_clib.add(self._t, other._t))
    
    def __sub__(self, other):
        if other.data_size > self.data_size:
            return Tensor(_clib.sub(other._t, self._t))
        return Tensor(_clib.sub(self._t, other._t))
    
    def __mul__(self, other):
        if other.data_size > self.data_size:
            return Tensor(_clib.mul(other._t, self._t))
        return Tensor(_clib.mul(self._t, other._t))
    
    def __matmul__(self, other):
        return Tensor(_clib.dot(self._t, other._t))
    
    def __truediv__(self, other):
        if other.data_size > self.data_size:
            return Tensor(_clib.divv(other._t, self._t))
        return Tensor(_clib.divv(self._t, other._t))
    
    def __pow__(self, other):
        return Tensor(_clib.poww(self._t, other))
    
    def relu(self):
        return Tensor(_clib.relu(self._t))
    
    def sigmoid(self):
        return Tensor(_clib.sigmoid(self._t))
    
    def tanh(self):
        return Tensor(_clib.tanhh(self._t))
    
    def exp(self):
        return Tensor(_clib.expp(self._t))
    
    def backward(self):
        if (self.requires_grad):
            _clib.backward(self._t)
        else:
            raise RuntimeError("Can't invoke backward() method because this tensor has no gradient.")
        
    @property
    def grad(self):
        if self.requires_grad:
            return _clib.print_tensor(self._t, True, False).decode()
        else:
            return []

class Stack:
    def __init__(self, tensors, requires_grad = True, dtype='float32'):
        if isinstance(tensors[0], list):
            self.tensors = []
            for ltensor in tensors:
                self.tensors.append(Tensor(ltensor, requires_grad=requires_grad, dtype=dtype))
        else:
            self.tensors = tensors
        self.requires_grad = requires_grad
        self.size = len(tensors)
        
    def __add__(self, other):
        return Stack([a+b for a, b in zip(self.tensors, other.tensors)])
    
    def __sub__(self, other):
        return Stack([a-b for a, b in zip(self.tensors, other.tensors)])
    
    def __mul__(self, other):
        return Stack([a*b for a, b in zip(self.tensors, other.tensors)])
    
    def __matmul__(self, other):
        return Stack([a@b for a, b in zip(self.tensors, other.tensors)])
    
    def __truediv__(self, other):
        return Stack([a/b for a, b in zip(self.tensors, other.tensors)])
    
    def __pow__(self, other):
        return Stack([a**other for a in self.tensors])
    
    def relu(self):
        return Stack([a.relu() for a in self.tensors])
    
    def sigmoid(self):
        return Stack([a.sigmoid() for a in self.tensors])
    
    def tanh(self):
        return Stack([a.tanh() for a in self.tensors])
    
    def exp(self):
        return Stack([a.exp() for a in self.tensors])
    
    def backward(self):
        return Stack([a.backward() for a in self.tensors])
    
    def __repr__(self):
        tcontainer = ""
        for i in range(self.size):
            tcontainer += _clib.print_tensor(self.tensors[i]._t, self.requires_grad, True).decode()
            if i != self.size-1:
                tcontainer += ",\n      "
                
        return f"Stack({tcontainer})"
        
    