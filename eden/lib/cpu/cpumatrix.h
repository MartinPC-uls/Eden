#ifndef __CPUMATRIX_H__
#define __CPUMATRIX_H__

#include <thread>
#include <random>
#include <immintrin.h>
#include "multithreading.h"

using namespace std;

typedef float (*BinaryMatrixOperator)(float, float);

float matrix_add_float32(float a, float b) { return a + b; }
float matrix_sub_float32(float a, float b) { return a - b; }
float matrix_mul_float32(float a, float b) { return a * b; }
float matrix_dot_float32(float a, float b) { return a * b; } // equal to mul_float32, made like this for simplicity
float matrix_div_float32(float a, float b) { return b != 0 ? a / b : 0; }
float matrix_pow_float32(float a, float b) { return pow(a, b); }
float matrix_relu_float32(float a, float b = NAN) { if (a < 0) return 0; return a; }
float matrix_sigmoid_float32(float a, float b = NAN) { return 1 / (1 + exp(a)); }
float matrix_tanh_float32(float a, float b = NAN) {
    float _exp = std::exp(2*a);
    return (_exp - 1)/(_exp + 1);
}
float matrix_exp_float32(float a, float b = NAN) { return exp(a); }

typedef __m256 (*BinaryMatrixOperatorSIMD)(__m256, __m256);

__m256 matrix_add_float32_simd(__m256 a, __m256 b) { return _mm256_add_ps(a, b); }
__m256 matrix_sub_float32_simd(__m256 a, __m256 b) { return _mm256_sub_ps(a, b); }
__m256 matrix_mul_float32_simd(__m256 a, __m256 b) { return _mm256_mul_ps(a, b); }
__m256 matrix_div_float32_simd(__m256 a, __m256 b) { return _mm256_div_ps(a, b); }

// Element-wise
void _mcpu_bop_ew(float** a, float** b,
                  BinaryMatrixOperatorSIMD op,
                  BinaryMatrixOperator _op,
                  int nthreads,
                  float** result,
                  int rows, int cols) {

    const int threshold = 30;
    if (nthreads <= 1 || (rows <= threshold && cols <= threshold)) {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result[r][c] = _op(a[r][c], b[r][c]);
            }
        }
        return;
    }

    vector<thread> thread_pool;
    thread_pool.reserve(nthreads);

    int chunk_size = rows / nthreads;
    int remainder = rows % nthreads;
    int start = 0;
    int finish = 0;

    for (int i = 0; i < nthreads; i++) {
        finish = start + chunk_size + (i < remainder);
        thread_pool.emplace_back([start, finish, a, b, result, cols, op, _op]() {
            for (int r = start; r < finish; r++) {
                int c = 0;
                for (; c < cols - 7; c += 8) {
                    __m256 simd_a = _mm256_loadu_ps(&a[r][c]);
                    __m256 simd_b = _mm256_loadu_ps(&b[r][c]);
                    __m256 simd_result = op(simd_a, simd_b);
                    _mm256_storeu_ps(&result[r][c], simd_result);
                }

                for (; c < cols; c++) {
                    result[r][c] = _op(a[r][c], b[r][c]);
                }
            }
        });
        start = finish;
    }

    for (auto& thread : thread_pool) {
        thread.join();
    }
}

// Vector x Vector
void _mcpu_bop_matmul__v_v(float** a, float** b,
                           int nthreads,
                           float** result,
                           int cols) {

    const int threshold = 30;
    if (nthreads <= 1 || (cols <= threshold && cols <= threshold)) {
        for (int i = 0; i < cols; i++) {
            result[0][0] += a[0][i] * b[0][i];
        }
        return;
    }

    vector<thread> thread_pool;
    thread_pool.reserve(nthreads);

    int chunk_size = cols / nthreads;
    int remainder = cols % nthreads;
    int start = 0;
    int finish = 0;

    for (int i = 0; i < nthreads; i++) {
        finish = start + chunk_size + (i < remainder);
        thread_pool.emplace_back([start, finish, a, b, result, cols]() {
            for (int i = start; i < finish; i++) {
                result[0][0] += a[0][i] * b[0][i];
            }
        });
        start = finish;
    }

    for (auto& thread : thread_pool) {
        thread.join();
    }
}

// Matrix x Vector
void _mcpu_bop_matmul__m_v(float** a, float** b,
                           int nthreads,
                           float** result,
                           int rows, int cols) {

    const int threshold = 30;
    if (nthreads <= 1 || (rows <= threshold && cols <= threshold)) {
        for (int i = 0; i < rows; i++) {
            result[0][i] = 0;
            for (int k = 0; k < cols; k++) {
                result[0][i] += a[i][k] * b[0][k];
            }
        }
        return;
    }

    vector<thread> thread_pool;
    thread_pool.reserve(nthreads);

    int chunk_size = rows / nthreads;
    int remainder = rows % nthreads;
    int start = 0;
    int finish = 0;

    for (int i = 0; i < nthreads; i++) {
        finish = start + chunk_size + (i < remainder);
        thread_pool.emplace_back([start, finish, a, b, result, cols]() {
            for (int i = start; i < finish; i++) {
                result[0][i] = 0;
                for (int k = 0; k < cols; k++) {
                    result[0][i] += a[i][k] * b[0][k];
                }
            }
        });
        start = finish;
    }

    for (auto& thread : thread_pool) {
        thread.join();
    }
}

// Matrix x Matrix
void _mcpu_bop_matmul__m_m(float** a, float** b,
                           int nthreads,
                           float** result,
                           int a_r, int a_c,
                           int b_r, int b_c,
                           int out_r, int out_c) {

    const int threshold = 8;
    if (nthreads <= 1 || (a_r <= threshold && b_r <= threshold) || (a_c <= threshold && b_c <= threshold)) {
        for (int i = 0; i < out_r; i++) {
            for (int j = 0; j < out_c; j++) {
                result[i][j] = 0;
                for (int k = 0; k < a_c; k++) {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return;
    }

    vector<thread> thread_pool;
    thread_pool.reserve(nthreads);

    int chunk_size = out_r / nthreads;
    int remainder = out_r % nthreads;
    int start = 0;
    int finish = 0;

    for (int i = 0; i < nthreads; i++) {
        finish = start + chunk_size + (i < remainder);
        thread_pool.emplace_back([start, finish, a, b, result, out_r, out_c, a_c]() {
            for (int i = start; i < finish; i++) {
                for (int j = 0; j < out_c; j++) {
                    result[i][j] = 0;
                    for (int k = 0; k < a_c; k++) {
                        result[i][j] += a[i][k] * b[k][j];
                    }
                }
            }
        });
        start = finish;
    }

    for (auto& thread : thread_pool) {
        thread.join();
    }
}

void _mcpu_randn(float** a,
                 int rows, int cols,
                 int nthreads,
                 float M, float SD, bool spare,
                 mt19937_64 mtrand) {
    if (nthreads > 1 && (rows > 1 || cols > 1)) {
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        int val1, val2;
        if (rows > cols) {
            val1 = rows;
            val2 = cols;
        } else {
            val1 = cols;
            val2 = rows;
        }
        if (nthreads > val1) nthreads = val1;

        int chunk_size = rows / nthreads;
        int remainder = rows % nthreads;

        //float const1 = 1.0f/(sqrt(2.0f*M_PI*SD));
        //float const2 = 2.0f*SD;

        
        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            t[i] = thread([start, finish, a, val2,
                           mtrand, spare, M, SD, cols]() mutable {
                for (int _i = start; _i < finish; _i++) {
                    for (int _j = 0; _j < cols; _j++) {
                        float u1, u2, z;
                        float val;
                        do {
                            u1 = static_cast<float>(mtrand()) / static_cast<float>(mtrand.max());
                            u2 = static_cast<float>(mtrand()) / static_cast<float>(mtrand.max());
                            if (spare) {
                                z = sqrt(-2.0f * log(u1)) * sin(2.0f * M_PI * u2);
                            } else {
                                z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
                            }

                            val = M + SD * z;
                        } while (std::isinf(val) || std::isnan(val));
                        a[_i][_j] = val;
                    }
                }
            });
            start = finish;
        }

        _init_threads(t, nthreads);
    } else {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float u1, u2, z;
                float val;
                do {
                    u1 = static_cast<float>(mtrand()) / static_cast<float>(mtrand.max());
                    u2 = static_cast<float>(mtrand()) / static_cast<float>(mtrand.max());
                    if (spare) {
                        z = sqrt(-2.0f * log(u1)) * sin(2.0f * M_PI * u2);
                    } else {
                        z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
                    }

                    val = M + SD * z;
                } while (std::isinf(val) || std::isnan(val));
                a[r][c] = val;
            }
        }
    }
}


#endif