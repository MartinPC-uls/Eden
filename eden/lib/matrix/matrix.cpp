#include <iostream>
#include <list>
#include <typeinfo>
#include <functional>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <set>
#include <chrono>
#include <stack>
#include <cstdint>
#include <sstream>
#include <string.h>
#include <random>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "../ops/binaryops.h"
#include "../ops/activationops.h"
#include "../ops/trigonometric.h"
#include "../ops/transforms.h"
#include "../cpu/mtmatrix.h"
#include "../cpu/cpumatrix.h"
#include "../parameters.h"

extern "C" {
    int get_threads() {
        return thread::hardware_concurrency();
    }

    Matrix* matrix_float32(float** data, int rows, int cols, int nthreads, bool requires_grad) {
        return new Matrix(data, cols, rows, nthreads, requires_grad);
    }

    Matrix* matrix_float32_init(float** data, int rows, int cols, int nthreads, bool requires_grad) {
        float** new_data;
        malloc_float32_matrix(new_data, rows, cols, nthreads);
        fill_float32_matrix(new_data, rows, cols, data, nthreads);
        return new Matrix(new_data, rows, cols, nthreads, requires_grad);
    }

    bool get_requires_grad(Matrix* obj) { return obj->requires_grad; }
    int get_cols(Matrix* obj) { return obj->c; }
    int get_rows(Matrix* obj) { return obj->r; }
    int get_nthreads(Matrix* obj) { return obj->nthreads; }

    void delete_matrix(Matrix* obj) {
        delete obj;
    }

    float** get_matrix_data(Matrix* obj) {
        return obj->data;
    }

    float** get_matrix_grad(Matrix* obj) {
        return obj->grad;
    }

    void print_matrix(Matrix* obj) {
        int elements = obj->r * obj->c;
        printf("Matrix(");
        if (obj->r > 1) printf("[");
        for (int i = 0; i < obj->r; i++) {
            if (i > 0) printf("\t");
            printf("[");
            for (int j = 0; j < obj->c; j++) {
                if (obj->data[i][j] == (int)obj->data[i][j]) {
                    printf("%.0f.", obj->data[i][j]);
                } else {
                    printf("%.4f", obj->data[i][j]);
                }

                if (j != obj->c-1) printf(", ");
                if (elements > 1000 && obj->c > 6 && j == 2) {
                    printf("..., ");
                    j = obj->c-4;
                }
            }
            printf("]");
            if (i != obj->r-1) printf(",\n");
            if (elements > 1000 && obj->r > 6 && i == 2) {
                printf("\t...,\n");
                i = obj->r-4;
            }
        }
        if (obj->r > 1) printf("]");
        if (obj->requires_grad) printf(", requires_grad=True");
        printf(")");
    }

    void print_grad(Matrix* obj) {
        printf("grad(");
        if (obj->r > 1) printf("[");
        for (int i = 0; i < obj->r; i++) {
            if (i > 0) printf("      ");
            printf("[");
            for (int j = 0; j < obj->c; j++) {
                if (obj->grad[i][j] == (int)obj->grad[i][j]) printf("%.0f.", obj->grad[i][j]);
                else printf("%.3f", obj->grad[i][j]);

                if (j != obj->c-1) printf(", ");
            }
            printf("]");
            if (i != obj->r-1) printf(",\n");
        }
        if (obj->r > 1) printf("]");
        printf(")");
    }


    void release_int_memory(int* ptr) { free(ptr); }
    void release_float32_memory(int* ptr) { free(ptr); }
    void release_float64_memory(int* ptr) { free(ptr); }
    void release_char_memory(int* ptr) { free(ptr); }

    Matrix* add(Matrix* a, Matrix* b) {
        float** result;
        int r = a->r;
        int c = a->c;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, r, c, 1);

        if (r == b->r && c == b->c) { // if dimensions are the same
            _mcpu_bop_ew(a->data, b->data, matrix_add_float32_simd, matrix_add_float32, nthreads, result, r, c);
        } else if (b->r == 1 && c == b->c) { // if 'b' is a vector
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] + b->data[0][j];
                }
            }
        } else if (r == b->r && b->c == 1) { // if 'b' is a matrix of a.r==b.r x 1
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] + b->data[i][0];
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] + b->data[0][0];
                }
            }
        } else {
            throw std::invalid_argument("Matrix 'a' and 'b' are not compatible shapes.");
        }

        Matrix* out = new Matrix(result, r, c, nthreads, (a->requires_grad || b->requires_grad), {a, b});
        out->_backward = matrix_addition_backward(a, b, out);

        return out;
    }
    
    Matrix* sub(Matrix* a, Matrix* b) {
        float** result;
        int r = a->r;
        int c = a->c;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, r, c, 1);

        if (r == b->r && c == b->c) { // if dimensions are the same
            _mcpu_bop_ew(a->data, b->data, matrix_sub_float32_simd, matrix_sub_float32, nthreads, result, r, c);
        } else if (b->r == 1 && c == b->c) { // if 'b' is a vector
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] - b->data[0][j];
                }
            }
        } else if (r == b->r && b->c == 1) { // if 'b' is a matrix of a.r==b.r x 1
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] - b->data[i][0];
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] - b->data[0][0];
                }
            }
        } else {
            throw std::invalid_argument("Matrix 'a' and 'b' are not compatible shapes.");
        }

        Matrix* out = new Matrix(result, r, c, nthreads, (a->requires_grad || b->requires_grad), {a, b});
        out->_backward = matrix_substraction_backward(a, b, out);

        return out;
    }

    Matrix* mul(Matrix* a, Matrix* b) {
        float** result;
        int r = a->r;
        int c = a->c;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, r, c, nthreads);

        if (r == b->r && c == b->c) { // if dimensions are the same
            _mcpu_bop_ew(a->data, b->data, matrix_mul_float32_simd, matrix_mul_float32, nthreads, result, r, c);
        } else if (b->r == 1 && c == b->c) { // if 'b' is a vector
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] * b->data[0][j];
                }
            }
        } else if (r == b->r && b->c == 1) { // if 'b' is a matrix of a.r==b.r x 1
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] * b->data[i][0];
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] * b->data[0][0];
                }
            }
        } else {
            throw std::invalid_argument("Matrix 'a' and 'b' are not compatible shapes.");
        }

        Matrix* out = new Matrix(result, r, c, nthreads, (a->requires_grad || b->requires_grad), {a, b});
        out->_backward = matrix_multiplication_backward(a, b, out);

        return out;
    }

    Matrix* divv(Matrix* a, Matrix* b) {
        float** result;
        int r = a->r;
        int c = a->c;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, r, c, 1);

        if (r == b->r && c == b->c) { // if dimensions are the same
            _mcpu_bop_ew(a->data, b->data, matrix_div_float32_simd, matrix_div_float32, nthreads, result, r, c);
        } else if (b->r == 1 && c == b->c) { // if 'b' is a vector
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] / b->data[0][j];
                }
            }
        } else if (r == b->r && b->c == 1) { // if 'b' is a matrix of a.r==b.r x 1
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] / b->data[i][0];
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    result[i][j] = a->data[i][j] / b->data[0][0];
                }
            }
        } else {
            throw std::invalid_argument("Matrix 'a' and 'b' are not compatible shapes.");
        }

        Matrix* out = new Matrix(result, r, c, nthreads, (a->requires_grad || b->requires_grad), {a, b});
        out->_backward = matrix_division_backward(a, b, out);

        return out;
    }

    Matrix* matmul(Matrix* a, Matrix* b) {
        float** result;
        int nthreads = a->nthreads;

        int out_r = a->r;
        int out_c = b->c;

        // Check if 'b' is a vector (1xN) and if cols in 'a' and 'b' are the same,
        // if so, we do matrix-vector product
        if (a->r == 1 && b->r == 1) { // If 'a' and 'b' are vectors, returns a 1x1 matrix (scalar)
            out_c = 1;
            malloc_float32_matrix(result, 1, 1, 1);
            result[0][0] = 0;
            _mcpu_bop_matmul__v_v(a->data, b->data, nthreads, result, a->c);
        } else if (b->r == 1 && a->c == b->c) { // If 'b' is a vector
            out_r = 1;
            out_c = a->r;
            malloc_float32_matrix(result, out_r, out_c, 1);
            _mcpu_bop_matmul__m_v(a->data, b->data, nthreads, result, out_c, a->c);
        } else if (a->c == b->r) { // If shapes are compatible
            malloc_float32_matrix(result, out_r, out_c, 1);
            _mcpu_bop_matmul__m_m(a->data, b->data, nthreads, result, a->r, a->c, b->r, b->c, out_r, out_c);
        } else {
            throw std::invalid_argument("Both shapes are not compatible for matrix multiplication.");
        }

        Matrix* out = new Matrix(result, out_r, out_c, nthreads, (a->requires_grad || b->requires_grad), {a, b});
        out->_backward = matrix_matmul_backward(a, b, out);

        return out;
    }

    Matrix* sum(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, 1, 1, nthreads);

        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[0][0] += a->data[i][j];
            }
        }

        Matrix* out = new Matrix(result, 1, 1, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_sum_backward(a, out);

        return out;
    }

    Matrix* poww(Matrix* a, float b) {
        if (b == 1) return a;
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);

        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = pow(a->data[i][j], b);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_pow_backward(a, b, out);

        return out;
    }

    Matrix* relu(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = a->data[i][j] > 0 ? a->data[i][j] : 0;
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_relu_backward(a, out);

        return out;
    }

    Matrix* sigmoid(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = 1 / (1 + exp(-a->data[i][j]));
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_sigmoid_backward(a, out);

        return out;
    }

    Matrix* tanhh(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                float _exp = std::exp(2*a->data[i][j]);
                result[i][j] = (_exp - 1)/(_exp + 1);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_tanh_backward(a, out);

        return out;
    }

    Matrix* coss(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = cos(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_cos_backward(a, out);

        return out;
    }

    Matrix* sinn(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = sin(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_sin_backward(a, out);

        return out;
    }

    Matrix* tann(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = tan(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_tan_backward(a, out);

        return out;
    }

    Matrix* asinn(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = asin(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_asin_backward(a, out);

        return out;
    }

    Matrix* acoss(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = acos(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_acos_backward(a, out);

        return out;
    }

    Matrix* coshh(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = cosh(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_cosh_backward(a, out);

        return out;
    }

    Matrix* sinhh(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = sinh(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_sinh_backward(a, out);

        return out;
    }

    Matrix* atann(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = atan(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_atan_backward(a, out);

        return out;
    }

    Matrix* softsign(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = a->data[i][j]/(1+abs(a->data[i][j]));
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_softsign_backward(a, out);

        return out;
    }

    Matrix* sqrtt(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = sqrt(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_sqrt_backward(a, out);

        return out;
    }

    Matrix* expp(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->r, a->c, nthreads);
        for (int i = 0; i < a->r; i++) {
            for (int j = 0; j < a->c; j++) {
                result[i][j] = exp(a->data[i][j]);
            }
        }

        Matrix* out = new Matrix(result, a->r, a->c, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_exp_backward(a, out);

        return out;
    }

    Matrix* transpose(Matrix* a) {
        float** result;
        int nthreads = a->nthreads;
        malloc_float32_matrix(result, a->c, a->r, nthreads);
        for (int i = 0; i < a->c; i++) {
            for (int j = 0; j < a->r; j++) {
                result[i][j] = a->data[j][i];
            }
        }

        Matrix* out = new Matrix(result, a->c, a->r, nthreads, (a->requires_grad), {a});
        out->_backward = matrix_transpose_backward(a, out);

        return out;
    }

    Matrix* full(float data, int r, int c, int nthreads, bool requires_grad) {
        float **result;
        malloc_float32_matrix(result, r, c, nthreads);
        full_float32_matrix(result, r, c, data, nthreads);

        return new Matrix(result, r, c, nthreads, requires_grad);
    }

    Matrix* ones(int r, int c, int nthreads, bool requires_grad) {
        return full(1, r, c, nthreads, requires_grad);
    }

    Matrix* zeros(int r, int c, int nthreads, bool requires_grad) {
        return full(0, r, c, nthreads, requires_grad);
    }

    Matrix* _random(int r, int c, int seed, int lower_bound, int upper_bound, int nthreads, bool requires_grad) {
        random_device rd;
        mt19937_64 mtrand;
        if (seed == 0) {
            mtrand.seed(rd());
        } else {
            mtrand.seed(seed);
        }
        
        float** data;
        malloc_float32_matrix(data, r, c, nthreads);
        for (int i = 0; i < r; i++) {
            for (int j = 0; j < c; j++) {
                data[i][j] = static_cast<float>((mtrand() % (upper_bound - lower_bound + 1)) + lower_bound);
            }
        }

        return new Matrix(data, r, c, nthreads, requires_grad);
    }

    Matrix* _randn(int r, int c, int seed, float M, float SD, bool spare, int nthreads, bool requires_grad) {
        // M stands for mean
        // SD stands for Standard Deviation
        random_device rd;
        mt19937_64 mtrand;
        if (seed == 0) {
            mtrand.seed(rd());
        } else {
            mtrand.seed(seed);
        }

        float** data;
        malloc_float32_matrix(data, r, c, 1);

        _mcpu_randn(data, r, c, nthreads, M, SD, spare, mtrand);

        return new Matrix(data, r, c, nthreads, requires_grad);
    }

    void backward(Matrix* a, float sgrad) { a->backward(sgrad); }
    void update(Matrix* a, float learning_rate = 0.01f) { a->update(learning_rate); }
    void update_graph(Matrix* a, float learning_rate = 0.01f) {
        a->update_graph(learning_rate);
    }

    // Parameters functions
    Parameters* init_parameters(Matrix** parameters, int size) {
        return new Parameters(parameters, size);
    }

    void update_parameters(Parameters* parameters_obj, float learning_rate) {
        parameters_obj->update(learning_rate);
    }

    void zero_grad(Matrix* a) {
        a->zero_grad();
    }

}

/* 
    Command to export to .SO:

        clang++ -Os -fPIC -mfma -shared -o "../binaries/matrix.so" matrix.cpp -mavx

*/