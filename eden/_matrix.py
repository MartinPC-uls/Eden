from eden import _clib, _Matrix
import ctypes as _C

_GLOBAL_SEED = 0

def get_threads_available():
    return _clib.get_threads()

_GLOBAL_THREADS = get_threads_available()-1

def get_designed_threads():
    global _GLOBAL_THREADS
    return _GLOBAL_THREADS

def set_threads(threads):
    global _GLOBAL_THREADS
    available_threads = get_threads_available()
    if threads > available_threads:
        print(f"WARNING: You're exceeding the maximum number of threads on your CPU!.\nCPU Threads: {available_threads}")
    _GLOBAL_THREADS = threads

def delete(matrix):
    if matrix.alive:
        _clib.delete_vector(matrix._m)
        matrix._alive = False
    else:
        print(f'vector is not alive. (operation: delete, data: {matrix.data})')
        
def manual_seed(seed):
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    
def get_seed():
    global _GLOBAL_SEED
    return _GLOBAL_SEED

class Matrix:
    def __init__(self, data, requires_grad=False, dtype='float32'):
        global _GLOBAL_THREADS
        self._alive = True
        if isinstance(data, _C.POINTER(_Matrix)):
            self.r = _clib.get_rows(data)
            self.c = _clib.get_cols(data)
            self.requires_grad = _clib.get_requires_grad(data)
            self._m = data
        else:
            if dtype == 'float32':
                if isinstance(data, float) or isinstance(data, int): data = [data]
                
                if isinstance(data[0], float) or isinstance(data[0], int): # Vector
                    self.r = 1
                    self.c = len(data)
                    c_matrix = (_C.POINTER(_C.c_float) * self.r)()
                    c_matrix[0] = (_C.c_float * self.c)(*data)
                else: # Matrix
                    self.r = len(data)
                    self.c = len(data[0])
                    c_matrix = (_C.POINTER(_C.c_float) * self.r)()
                    for i in range(self.r): c_matrix[i] = (_C.c_float * self.c)(*data[i])
                
                self.requires_grad = requires_grad
                self._m = _clib.matrix_float32_init(c_matrix, self.r, self.c, _GLOBAL_THREADS, requires_grad)
                
        self.shape = (self.r, self.c)
                
    def __repr__(self):
        _clib.print_matrix(self._m)
        return ""
        
    def __add__(self, other):
        other = Matrix(other) if not isinstance(other, Matrix) else other
        return Matrix(_clib.add(self._m, other._m))
    
    def __sub__(self, other):
        other = Matrix(other) if not isinstance(other, Matrix) else other
        return Matrix(_clib.sub(self._m, other._m))
    
    def __mul__(self, other):
        other = Matrix(other) if not isinstance(other, Matrix) else other
        return Matrix(_clib.mul(self._m, other._m))
    
    def __matmul__(self, other):
        other = Matrix(other) if not isinstance(other, Matrix) else other
        return Matrix(_clib.matmul(self._m, other._m))
    
    def __truediv__(self, other):
        other = Matrix(other) if not isinstance(other, Matrix) else other
        return Matrix(_clib.divv(self._m, other._m))
    
    def __pow__(self, other):
        # Only supporting floating and integer numbers for now
        return Matrix(_clib.poww(self._m, other))
    
    def backward(self, sgrad=1.0):
        if self.shape != (1, 1):
            raise RuntimeError("backward() function can only be called on scalar outputs (1x1 matrix).")
        elif self.requires_grad:
            _clib.backward(self._m, sgrad)
        else:
            raise RuntimeError("Can't invoke backward() because this matrix has no gradient.")
        
    def update(self, multiplier=0.01):
        _clib.update(self._m, multiplier)
        
    def update_graph(self, multiplier=0.01):
        raise NotImplementedError("update_graph function is not yet implemented.")
        _clib.update_graph(self._m, multiplier)
        
    def zero_grad(self):
        _clib.zero_grad(self._m)
        
    def transpose(self):
        return Matrix(_clib.transpose(self._m))
    t = transpose
    
    @property
    def grad(self):
        if self.requires_grad:
            _clib.print_grad(self._m)
        else:    
            return None
        
        return ""
    
    @property
    def T(self):
        return Matrix(_clib.transpose(self._m))
    
    