#ifndef _MULTITHREADING_H_
#define _MULTITHREADING_H_

#include <functional>
#include <thread>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include "../matrix/matrix.h"

using namespace std;

void init_threads(thread* t, int nthreads) {
    for (int i = 0; i < nthreads; i++) {
        t[i].join();
    }
}

void fill_float32_array(float* array, float* data, int size, int nthreads) {
    if (nthreads > 1 && size > 1) {
        if (nthreads > size) nthreads = size;
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        int chunk_size = size / nthreads;
        int remainder = size % nthreads;
        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            t[i] = thread([start, finish, array, data, size]() {
                for (int j = start; j < finish; j++) {
                    array[j] = data[j];
                }
            });
            start = finish;
        }

        init_threads(t, nthreads);
    } else {
        for (int i = 0; i < size; i++) {
            array[i] = data[i];
        }
    }
}

void full_float32_array(float* array, float data, unsigned int size, int nthreads) {
    if (nthreads > 1 && size > 1) {
        if (nthreads > size) nthreads = size;
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        int chunk_size = size / nthreads;
        int remainder = size % nthreads;
        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            t[i] = thread([start, finish, array, data]() {
                for (int i = start; i < finish; i++) {
                    array[i] = data;
                }
            });
            start = finish;
        }

        init_threads(t, nthreads);
    } else if (size > 1) {
        for (int i = 0; i < size; i++) {
            array[i] = data;
        }
    } else {
        array[0] = data;
    }
}

typedef float (*BinaryOperator)(float, float);

float add_float32(float a, float b) { return a + b; }
float sub_float32(float a, float b) { return a - b; }
float mul_float32(float a, float b) { return a * b; }
float dot_float32(float a, float b) { return a * b; } // equal to mul_float32, made like this for simplicity
float div_float32(float a, float b) { return b != 0 ? a / b : 0; }
float pow_float32(float a, float b) { return pow(a, b); }
float relu_float32(float a, float b = NAN) { if (a < 0) return 0; return a; }
float sigmoid_float32(float a, float b = NAN) { return 1 / (1 + exp(a)); }
float tanh_float32(float a, float b = NAN) {
    float _exp = std::exp(2*a);
    return (_exp - 1)/(_exp + 1);
}
float exp_float32(float a, float b = NAN) { return exp(a); }

void cpu_paralell(BinaryOperator op, int nthreads, int data_size, float* out, float* a, float* b, float c = NAN) {
    if (nthreads > 1 && data_size > 1) {
        if (nthreads > data_size) nthreads = data_size;
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        int chunk_size = data_size / nthreads;
        int remainder = data_size % nthreads;

        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            if (op != pow_float32 && op != dot_float32 && b != NULL) {
                t[i] = thread([start, finish, out, a, b, op]() {
                    for (int j = start; j < finish; j++) {
                        out[j] = op(a[j], b[j]);
                    }
                });
            } else if (op == dot_float32) {
                t[i] = thread([start, finish, out, a, b, op]() {
                    for (int j = start; j < finish; j++) {
                        out[0] += op(a[j], b[j]);
                    }
                });
            } else {
                t[i] = thread([start, finish, out, a, c, op]() {
                    for (int j = start; j < finish; j++) {
                        out[j] = op(a[j], c);
                    }
                });
            }
            start = finish;
        }
        init_threads(t, nthreads);
    } else {
        if (op != pow_float32 && op != dot_float32 && b != NULL) {
            for (int i = 0; i < data_size; i++) {
                out[i] = op(a[i], b[i]);
            }
        } else if (op == dot_float32) {
            for (int i = 0; i < data_size; i++) {
                out[0] += op(a[i], b[i]);
            }
        } else {
            for (int i = 0; i < data_size; i++) {
                out[i] = op(a[i], c);
            }
        }
    }
}

void cpu_parallel_otm(BinaryOperator op, int nthreads, int data_size, float* out, float* a, float* b) {
    /*
        This function is very similar to 'cpu_parallel', the only difference is that this supports
        One To Many operations, in this case, an scalar and a vector.
        The reason I repeated this code is because we don't want to be checking for the index of 'a' or 'b'
        all the time, that would add one more conditional to the thread operation, and that could
        slow down everything. If I wouldn't care about speed, I would simply do:

                        out[j] = op(a[is_otm ? 0 : j], b[j]);
                                               ^
                                               for OTM ops, we must fix the index to 0.

        As I said, this adds an extra conditional, and for billions of operations, you would notice it.
    */
    if (nthreads > 1 && data_size > 1) {
        if (nthreads > data_size) nthreads = data_size;
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        int chunk_size = data_size / nthreads;
        int remainder = data_size % nthreads;

        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            if (op != pow_float32 && op != dot_float32 && b != NULL) {
                t[i] = thread([start, finish, out, a, b, op]() {
                    for (int j = start; j < finish; j++) {
                        out[j] = op(a[0], b[j]);
                    }
                });
            } else if (op == dot_float32) {
                t[i] = thread([start, finish, out, a, b, op]() {
                    for (int j = start; j < finish; j++) {
                        out[0] += op(a[0], b[j]);
                    }
                });
            }
            start = finish;
        }
        init_threads(t, nthreads);
    } else {
        if (op != pow_float32 && op != dot_float32 && b != NULL) {
            for (int i = 0; i < data_size; i++) {
                out[i] = op(a[0], b[i]);
            }
        } else if (op == dot_float32) {
            for (int i = 0; i < data_size; i++) {
                out[0] += op(a[0], b[i]);
            }
        }
    }
}

#endif