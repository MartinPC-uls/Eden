import ctypes as _C

def setup_matrix(lib, struct):
    lib.get_threads.argtypes = None
    lib.get_threads.restype = _C.c_int

    lib.matrix_float32.argtypes = [_C.POINTER(_C.POINTER(_C.c_float)), _C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib.matrix_float32.restype = _C.POINTER(struct)

    lib.matrix_float32_init.argtypes = [_C.POINTER(_C.POINTER(_C.c_float)), _C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib.matrix_float32_init.restype = _C.POINTER(struct)

    lib.get_requires_grad.argtypes = [_C.POINTER(struct)]
    lib.get_requires_grad.restype = _C.c_bool

    lib.get_rows.argtypes = [_C.POINTER(struct)]
    lib.get_rows.restype = _C.c_int

    lib.get_cols.argtypes = [_C.POINTER(struct)]
    lib.get_cols.restype = _C.c_int

    lib.get_nthreads.argtypes = [_C.POINTER(struct)]
    lib.get_nthreads.restype = _C.c_int

    lib.delete_matrix.argtypes = [_C.POINTER(struct)]
    lib.delete_matrix.restype = None

    lib.get_matrix_data.argtypes = [_C.POINTER(struct)]
    lib.get_matrix_data.restype = _C.POINTER(_C.POINTER(_C.c_float))

    lib.get_matrix_grad.argtypes = [_C.POINTER(struct)]
    lib.get_matrix_grad.restype = _C.POINTER(_C.POINTER(_C.c_float))

    lib.print_matrix.argtypes = [_C.POINTER(struct)]
    lib.print_matrix.restype = None

    lib.print_grad.argtypes = [_C.POINTER(struct)]
    lib.print_grad.restype = None

    lib.add.argtypes = [_C.POINTER(struct), _C.POINTER(struct)]
    lib.add.restype = _C.POINTER(struct)

    lib.sub.argtypes = [_C.POINTER(struct), _C.POINTER(struct)]
    lib.sub.restype = _C.POINTER(struct)

    lib.mul.argtypes = [_C.POINTER(struct), _C.POINTER(struct)]
    lib.mul.restype = _C.POINTER(struct)

    lib.divv.argtypes = [_C.POINTER(struct), _C.POINTER(struct)]
    lib.divv.restype = _C.POINTER(struct)

    lib.matmul.argtypes = [_C.POINTER(struct), _C.POINTER(struct)]
    lib.matmul.restype = _C.POINTER(struct)

    lib.poww.argtypes = [_C.POINTER(struct), _C.c_float]
    lib.poww.restype = _C.POINTER(struct)

    lib.relu.argtypes = [_C.POINTER(struct)]
    lib.relu.restype = _C.POINTER(struct)

    lib.sigmoid.argtypes = [_C.POINTER(struct)]
    lib.sigmoid.restype = _C.POINTER(struct)

    lib.tanhh.argtypes = [_C.POINTER(struct)]
    lib.tanhh.restype = _C.POINTER(struct)

    lib.expp.argtypes = [_C.POINTER(struct)]
    lib.expp.restype = _C.POINTER(struct)

    lib.full.argtypes = [_C.c_float, _C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib.full.restype = _C.POINTER(struct)

    lib.ones.argtypes = [_C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib.ones.restype = _C.POINTER(struct)

    lib.zeros.argtypes = [_C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib.zeros.restype = _C.POINTER(struct)

    lib._randn.argtypes = [_C.c_int, _C.c_int, _C.c_int, _C.c_float, _C.c_float, _C.c_bool, _C.c_int, _C.c_bool]
    lib._randn.restype = _C.POINTER(struct)

    lib._random.argtypes = [_C.c_int, _C.c_int, _C.c_int, _C.c_int, _C.c_int, _C.c_int, _C.c_bool]
    lib._random.restype = _C.POINTER(struct)

    lib.backward.argtypes = [_C.POINTER(struct), _C.c_float]
    lib.backward.restype = None
    
    lib.update.argtypes = [_C.POINTER(struct), _C.c_float]
    lib.update.restype = None
    
    lib.update_graph.argtypes = [_C.POINTER(struct), _C.c_float]
    lib.update_graph.restype = None
    
    lib.zero_grad.argtypes = [_C.POINTER(struct)]
    lib.zero_grad.restype = None
    
    lib.sum.argtypes = [_C.POINTER(struct)]
    lib.sum.restype = _C.POINTER(struct)
    
    lib.coss.argtypes = [_C.POINTER(struct)]
    lib.coss.restype = _C.POINTER(struct)
    
    lib.sinn.argtypes = [_C.POINTER(struct)]
    lib.sinn.restype = _C.POINTER(struct)
    
    lib.tann.argtypes = [_C.POINTER(struct)]
    lib.tann.restype = _C.POINTER(struct)
    
    lib.asinn.argtypes = [_C.POINTER(struct)]
    lib.asinn.restype = _C.POINTER(struct)
    
    lib.acoss.argtypes = [_C.POINTER(struct)]
    lib.acoss.restype = _C.POINTER(struct)
    
    lib.coshh.argtypes = [_C.POINTER(struct)]
    lib.coshh.restype = _C.POINTER(struct)
    
    lib.sinhh.argtypes = [_C.POINTER(struct)]
    lib.sinhh.restype = _C.POINTER(struct)
    
    lib.atann.argtypes = [_C.POINTER(struct)]
    lib.atann.restype = _C.POINTER(struct)
    
    lib.sqrtt.argtypes = [_C.POINTER(struct)]
    lib.sqrtt.restype = _C.POINTER(struct)
    
    lib.transpose.argtypes = [_C.POINTER(struct)]
    lib.transpose.restype = _C.POINTER(struct)
    
    lib.softsign.argtypes = [_C.POINTER(struct)]
    lib.softsign.restype = _C.POINTER(struct)
    
def setup_parameters(lib, struct, matrix_struct):
    lib.init_parameters.argtypes = [_C.POINTER(_C.POINTER(matrix_struct)), _C.c_int]
    lib.init_parameters.restype = _C.POINTER(struct)
    
    lib.update_parameters.argtypes = [_C.POINTER(struct), _C.c_float]
    lib.update_parameters.restype = None
