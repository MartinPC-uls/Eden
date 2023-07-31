import ctypes
from typing import Any

class _Vector(ctypes.Structure):
    _fields_ = [
        ('requires_grad', ctypes.c_bool),
        ('data', ctypes.POINTER(ctypes.c_float)),
        ('grad', ctypes.POINTER(ctypes.c_float)),
        ('data_size', ctypes.c_int),
        ('nthreads', ctypes.c_int),
        ('test_value', ctypes.c_int)
    ]

_clib = ctypes.CDLL('./vector.so', winmode=0)

_clib.get_threads.argtypes = None
_clib.get_threads.restype = ctypes.c_int

_clib.vector_float32.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.vector_float32.restype = ctypes.POINTER(_Vector)

_clib.vector_float32_init.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.vector_float32_init.restype = ctypes.POINTER(_Vector)

_clib.vector_float32_object.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
_clib.vector_float32_object.restype = ctypes.c_void_p

_clib.get_vector_data.argtypes = [ctypes.c_void_p]
_clib.get_vector_data.restype = ctypes.POINTER(ctypes.c_float)

_clib.get_vector_grad.argtypes = [ctypes.c_void_p]
_clib.get_vector_grad.restype = ctypes.POINTER(ctypes.c_float)

_clib.get_vector_requieres_grad = [ctypes.c_void_p]
_clib.get_vector_requieres_grad = ctypes.c_bool

_clib.delete_vector.argtypes = [ctypes.c_void_p]
_clib.delete_vector.restype = None

_clib.get_vector_size.argtypes = [ctypes.c_void_p]
_clib.get_vector_size.restype = ctypes.c_int

_clib.get_one_item.argtypes = [ctypes.c_void_p]
_clib.get_one_item.restype = ctypes.c_float

_clib.add.argtypes = [ctypes.POINTER(_Vector), ctypes.POINTER(_Vector)]
_clib.add.restype = ctypes.POINTER(_Vector)

_clib.sub.argtypes = [ctypes.POINTER(_Vector), ctypes.POINTER(_Vector)]
_clib.sub.restype = ctypes.POINTER(_Vector)

_clib.mul.argtypes = [ctypes.POINTER(_Vector), ctypes.POINTER(_Vector)]
_clib.mul.restype = ctypes.POINTER(_Vector)

_clib.dot.argtypes = [ctypes.POINTER(_Vector), ctypes.POINTER(_Vector)]
_clib.dot.restype = ctypes.POINTER(_Vector)

_clib.divv.argtypes = [ctypes.POINTER(_Vector), ctypes.POINTER(_Vector)]
_clib.divv.restype = ctypes.POINTER(_Vector)

_clib.poww.argtypes = [ctypes.POINTER(_Vector), ctypes.c_float]
_clib.poww.restype = ctypes.POINTER(_Vector)

_clib.relu.argtypes = [ctypes.POINTER(_Vector)]
_clib.relu.restype = ctypes.POINTER(_Vector)

_clib.sigmoid.argtypes = [ctypes.POINTER(_Vector)]
_clib.sigmoid.restype = ctypes.POINTER(_Vector)

_clib.tanhh.argtypes = [ctypes.POINTER(_Vector)]
_clib.tanhh.restype = ctypes.POINTER(_Vector)

_clib.expp.argtypes = [ctypes.POINTER(_Vector)]
_clib.expp.restype = ctypes.POINTER(_Vector)

_clib.backward.argtypes = [ctypes.POINTER(_Vector)]
_clib.backward.restype = None

_clib.full.argtypes = [ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.full.restype = ctypes.POINTER(_Vector)

_clib.ones.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.ones.restype = ctypes.POINTER(_Vector)

_clib.zeros.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool]
_clib.zeros.restype = ctypes.POINTER(_Vector)

_clib._randn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_bool, ctypes.c_int, ctypes.c_bool]
_clib._randn.restype = ctypes.POINTER(_Vector)

_clib._random.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
_clib._random.restype = ctypes.POINTER(_Vector)

_clib.shuffle.argtypes = [ctypes.POINTER(_Vector), ctypes.c_int]
_clib.shuffle.restype = ctypes.POINTER(_Vector)

_clib.print_vector.argtypes = [ctypes.POINTER(_Vector), ctypes.c_bool, ctypes.c_bool]
_clib.print_vector.restype = ctypes.c_char_p

_clib.release_int_memory.argtypes = [ctypes.c_void_p]
_clib.release_int_memory.restype = None

_clib.release_float32_memory.argtypes = [ctypes.c_void_p]
_clib.release_float32_memory.restype = None

_clib.release_float64_memory.argtypes = [ctypes.c_void_p]
_clib.release_float64_memory.restype = None

_clib.release_char_memory.argtypes = [ctypes.c_void_p]
_clib.release_char_memory.restype = None

_clib.get_data_size.argtypes = [ctypes.POINTER(_Vector)]
_clib.get_data_size.restype = ctypes.c_int

_clib.get_requires_grad.argtypes = [ctypes.POINTER(_Vector)]
_clib.get_requires_grad.restype = ctypes.c_bool

_clib.get_nthreads.argtypes = [ctypes.POINTER(_Vector)]
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

def delete(vector):
    if vector.alive:
        _clib.delete_vector(vector._v)
        vector._alive = False
    else:
        print(f'vector is not alive. (operation: delete, data: {vector.data})')
        
def manual_seed(seed):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

def get_seed():
    global _GLOBAL_SEED
    return _GLOBAL_SEED


def randn(size, mean = 0, SD = 1, spare = False, nthreads = _GLOBAL_THREADS, requires_grad = False):
    # 'randn' uses "standard normal distribution" to generate numbers.
    # If 'spare' is set to False, it will use cos function internally to generate a number,
    # otherwise it will use sin function.
    seed = get_seed()
    
    return Vector(_clib._randn(size, seed, mean, SD, spare, nthreads, requires_grad))

def randn_stack(vector_size, n_vectors, mean = 0, SD = 1, spare = False, nthreads = _GLOBAL_THREADS, requires_grad = False):
    return Stack([randn(vector_size, mean, SD, spare, nthreads, requires_grad=requires_grad) for _ in range(n_vectors)])

def random(size, lower_bound = 0, upper_bound = 0x7FFF):
    seed = get_seed()
    return Vector(_clib._random(size, seed, lower_bound, upper_bound))

def shuffle(vector):
    seed = get_seed()
    return Vector(_clib.shuffle(vector._v, seed))

def full(data, size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Vector(_clib.full(data, size, threads, requires_grad))

def vector(data, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Vector(_clib.vector_float32_init((ctypes.c_float * len(data))(*data), len(data), _GLOBAL_THREADS, requires_grad))

def ones(size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Vector(_clib.ones(size, threads, requires_grad))

def zeros(size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Vector(_clib.zeros(size, threads, requires_grad))

def zeros_stack(size, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Stack([[0] for _ in range(size)], requires_grad)

def ReLU(x):
    if isinstance(x, Vector):
        return Vector(_clib.relu(x._v))
    elif isinstance(x, Stack):
        return Stack([Vector(_clib.relu(v._v)) for v in x.vectors])

def Tanh(x):
    return Vector(_clib.tanhh(x._v))

def Exp(x):
    return Vector(_clib.expp(x._v))

def Sigmoid(x):
    return Vector(_clib.sigmoid(x._v))

def TLU(x):
    return Vector(x._v)
    
class Vector:
    def __init__(self, data, requires_grad=False, dtype='float32'):
        global _GLOBAL_THREADS
        if isinstance(data, ctypes.POINTER(_Vector)):
            self._alive = True
            #self.data_size = data.contents.data_size
            #self.requires_grad = data.contents.requires_grad
            self.data_size = _clib.get_data_size(data)
            self.requires_grad = _clib.get_requires_grad(data)
            self._v = data
        else:
            if dtype == 'float32':
                self._alive = True
                self.data_size = len(data)
                self.requires_grad = requires_grad
                self._v = _clib.vector_float32_init((ctypes.c_float * self.data_size)(*data), self.data_size,
                                               _GLOBAL_THREADS, requires_grad) # _vector
            
    def __repr__(self):
        return _clib.print_vector(self._v, False, False).decode()
    
    def __add__(self, other):
        if other.data_size > self.data_size:
            return Vector(_clib.add(other._v, self._v))
        return Vector(_clib.add(self._v, other._v))
    
    def __sub__(self, other):
        if other.data_size > self.data_size:
            return Vector(_clib.sub(other._v, self._v))
        return Vector(_clib.sub(self._v, other._v))
    
    def __mul__(self, other):
        if other.data_size > self.data_size:
            return Vector(_clib.mul(other._v, self._v))
        return Vector(_clib.mul(self._v, other._v))
    
    def __matmul__(self, other):
        return Vector(_clib.dot(self._v, other._v))
    
    def __truediv__(self, other):
        if other.data_size > self.data_size:
            return Vector(_clib.divv(other._v, self._v))
        return Vector(_clib.divv(self._v, other._v))
    
    def __pow__(self, other):
        return Vector(_clib.poww(self._v, other))
    
    def relu(self):
        return Vector(_clib.relu(self._v))
    
    def sigmoid(self):
        return Vector(_clib.sigmoid(self._v))
    
    def tanh(self):
        return Vector(_clib.tanhh(self._v))
    
    def exp(self):
        return Vector(_clib.expp(self._v))
    
    def backward(self):
        if (self.requires_grad):
            _clib.backward(self._v)
        else:
            raise RuntimeError("Can't invoke backward() method because this vector has no gradient.")
        
    @property
    def grad(self):
        if self.requires_grad:
            return _clib.print_vector(self._v, True, False).decode()
        else:
            return []

class Stack:
    def __init__(self, vectors, requires_grad = False, dtype='float32'):
        if isinstance(vectors[0], list):
            self.vectors = []
            for lvector in vectors:
                self.vectors.append(vector(lvector, requires_grad=requires_grad, dtype=dtype))
        else:
            self.vectors = vectors
            
        self.requires_grad = requires_grad
        self.size = len(vectors)
        
    def __add__(self, other):
        if isinstance(other, Stack):
            return Stack([a+b for a, b in zip(self.vectors, other.vectors)])
        elif isinstance(other, Vector):
            return Stack([a+other for a in self.vectors])
    
    def __sub__(self, other):
        if isinstance(other, Stack):
            return Stack([a-b for a, b in zip(self.vectors, other.vectors)])
        elif isinstance(other, Vector):
            return Stack([a-other for a in self.vectors])
    
    def __mul__(self, other):
        if isinstance(other, Stack):
            return Stack([a*b for a, b in zip(self.vectors, other.vectors)])
        elif isinstance(other, Vector):
            return Stack([a*other for a in self.vectors])
    
    def __matmul__(self, other):
        if isinstance(other, Stack):
            return Stack([a@b for a, b in zip(self.vectors, other.vectors)])
        elif isinstance(other, Vector):
            if other.data_size == 1:
                raise NotImplementedError("This part is not yet implemented.")
            return Stack([a@other for a in self.vectors])
    
    def __truediv__(self, other):
        if isinstance(other, Stack):
            return Stack([a/b for a, b in zip(self.vectors, other.vectors)])
        elif isinstance(other, Vector):
            return Stack([a/other for a in self.vectors])
    
    def __pow__(self, other):
        return Stack([a**other for a in self.vectors])
    
    def relu(self):
        return Stack([a.relu() for a in self.vectors])
    
    def sigmoid(self):
        return Stack([a.sigmoid() for a in self.vectors])
    
    def tanh(self):
        return Stack([a.tanh() for a in self.vectors])
    
    def exp(self):
        return Stack([a.exp() for a in self.vectors])
    
    def backward(self):
        return Stack([a.backward() for a in self.vectors])
    
    def to_vector(self):
        return Vector([_clib.get_one_item(v._v) for v in self.vectors])
    
    def __repr__(self):
        tcontainer = ""
        for i in range(self.size):
            tcontainer += _clib.print_vector(self.vectors[i]._v, False, True).decode()
            if i != self.size-1:
                tcontainer += ",\n      "
                
        return f"Stack({tcontainer})"
    
    @property
    def grad(self):
        if self.requires_grad:
            _str = "";
            for v in self.vectors:
                _str += _clib.print_vector(v._v, True, False).decode() + "\n"
            return _str
        else:
            return []

    
    
        
    