from typing import Any
from .c_import import _C, setup_matrix, setup_parameters
from ._stack import Stack

class _Matrix(_C.Structure):
    _fields_ = [
        ('requires_grad', _C.c_bool),
        ('data', _C.POINTER(_C.POINTER(_C.c_float))),
        ('grad', _C.POINTER(_C.POINTER(_C.c_float))),
        ('r', _C.c_int),
        ('c', _C.c_int),
        ('nthreads', _C.c_int),
    ]
    
class _Parameters(_C.Structure):
    _fields_ = [
        ('parameters', _C.POINTER(_C.POINTER(_Matrix))),
        ('size', _C.c_int),
    ]

_clib = _C.CDLL("eden\\lib\\binaries\\matrix.so", winmode=0)
setup_matrix(_clib, _Matrix)
setup_parameters(_clib, _Parameters, _Matrix)

from ._matrix import *

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

def random(r=1, c=1, seed=0, lower_bound=0, upper_bound=0x7FFF, nthreads=None, requires_grad=False):
    global _GLOBAL_SEED
    nthreads = _GLOBAL_THREADS if nthreads == None else nthreads
    seed = _GLOBAL_SEED if _GLOBAL_SEED != 0 and seed == 0 else seed
    _GLOBAL_SEED += 1
    return Matrix(_clib._random(r, c, seed, lower_bound, upper_bound, nthreads, requires_grad))

def randn(r=1, c=1, seed=0, M=0, SD=1, spare=False, nthreads=None, requires_grad=False):
    global _GLOBAL_SEED
    nthreads = _GLOBAL_THREADS if nthreads == None else nthreads
    seed = _GLOBAL_SEED if _GLOBAL_SEED != 0 and seed == 0 else seed
    _GLOBAL_SEED += 1
    return Matrix(_clib._randn(r, c, seed, M, SD, spare, nthreads, requires_grad))

# shuffle

def full(data, r, c, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Matrix(_clib.full(data, r, c, threads, requires_grad))

def ones(r, c, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Matrix(_clib.ones(r, c, threads, requires_grad))

def zeros(r, c, threads=None, requires_grad=False):
    threads = threads if threads != None else _GLOBAL_THREADS
    return Matrix(_clib.ones(r, c, threads, requires_grad))

def add(a, b):
    return Matrix(_clib.add(a._m, b._m))

def sub(a, b):
    return Matrix(_clib.sub(a._m, b._m))

def mul(a, b):
    return Matrix(_clib.mul(a._m, b._m))

def matmul(a, b):
    return Matrix(_clib.matmul(a._m, b._m))

def div(a, b):
    return Matrix(_clib.divv(a._m, b._m))

def pow(a, b):
    return Matrix(_clib.poww(a._m, b))

def ReLU(x):
    return Matrix(_clib.relu(x._m))
relu = ReLU

def Tanh(x):
    return Matrix(_clib.tanhh(x._m))
tanh = Tanh

def Exp(x):
    return Matrix(_clib.expp(x._m))
exp = Exp

def Sigmoid(x):
    return Matrix(_clib.sigmoid(x._m))
sigmoid = Sigmoid

def TLU(x):
    return Matrix(x._m)
tlu = TLU

def Softsign(x):
    return Matrix(_clib.softsign(x._m))
softsign = Softsign

def sum(x):
    return Matrix(_clib.sum(x._m))

def cos(x):
    return Matrix(_clib.coss(x._m))

def sin(x):
    return Matrix(_clib.sinn(x._m))

def tan(x):
    return Matrix(_clib.tann(x._m))

def asin(x):
    return Matrix(_clib.asinn(x._m))
arcsin = asin

def acos(x):
    return Matrix(_clib.acoss(x._m))
arccos = acos

def cosh(x):
    return Matrix(_clib.coshh(x._m))

def sinh(x):
    return Matrix(_clib.sinhh(x._m))

def atan(x):
    return Matrix(_clib.atann(x._m))
arctan = atan

def sqrt(x):
    return Matrix(_clib.sqrtt(x._m))

def transpose(x):
    return Matrix(_clib.transpose(x._m))
t = transpose

def matrix(data, requires_grad=False, dtype='float32'):
    return Matrix(data, requires_grad, dtype)
