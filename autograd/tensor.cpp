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
#include "ops/binaryops.h"
#include "ops/activationops.h"
#include "cpu/multithreading.h"
using namespace std;

/*
    Any random operation is gonna be handled by Mersenne Twister's Algorithm,
    which is better for parallel computing and faster generation of random numbers.
*/

extern "C" {
    int get_threads() {
        return thread::hardware_concurrency();
    }

    // -- dtype = float32 --
    Tensor* tensor_float32(float* data, int data_size, int nthreads, bool requires_grad) {
        return new Tensor(data, data_size, nthreads, requires_grad);
    }

    Tensor* tensor_float32_init(float* data, int data_size, int nthreads, bool requires_grad) {
        float* new_data = (float*)malloc(data_size * sizeof(float));

        fill_float32_array(new_data, data, data_size, nthreads);
        Tensor* out = new Tensor(new_data, data_size, nthreads, requires_grad);

        return out;
    }

    bool get_requires_grad(Tensor* obj) { return obj->requires_grad; }
    int get_data_size(Tensor* obj) { return obj->data_size; }
    int get_nthreads(Tensor* obj) { return obj->nthreads; }

    Tensor* tensor_float32_object(float* data, int data_size) {
        // Repeated for Python
        return new Tensor(data, data_size);
    }

    void delete_tensor(Tensor* obj) {
        delete obj;
    }

    const float* get_tensor_data(Tensor* obj) {
        return obj->data;
    }

    const char* print_tensor(Tensor* obj, bool grad, bool is_stack) {
        int max_char_number_representation = 10;
        int MAXIMUM_SIZE_REPRESENTATION = 1000;
        bool max_reached = false;
        int data_size = obj->data_size * (max_char_number_representation + 2);

        if (obj->data_size > MAXIMUM_SIZE_REPRESENTATION) {
            max_reached = true;
            data_size = 9 * (max_char_number_representation + 2);
        }
        int tensor_str_size = 30 + data_size;

        char* tensor_str = (char*)malloc(tensor_str_size);
        if (!tensor_str) {
            return nullptr;
        }

        strcpy(tensor_str, "");

        char data_number[max_char_number_representation];

        if (!is_stack && !grad) {
            strcat(tensor_str, "Tensor([");
        } else {
            strcat(tensor_str, "[");
        }

        for (int i = 0; i < obj->data_size; i++) {
            if (!grad) {
                snprintf(data_number, sizeof(data_number), "%f", obj->data[i]);
            } else {
                snprintf(data_number, sizeof(data_number), "%f", obj->grad[i]);
            }
            strcat(tensor_str, data_number);

            if (i != obj->data_size - 1) {
                strcat(tensor_str, ", ");
            }

            if (max_reached && i == 2) {
                strcat(tensor_str, "..., ");
                i = obj->data_size-4;
            }
        }

        strcat(tensor_str, "]");
        if (!is_stack && !grad) {
            strcat(tensor_str, ")");
        }

        return tensor_str;
    }
    
    void release_int_memory(int* ptr) { free(ptr); }
    void release_float32_memory(float* ptr) { free(ptr); }
    void release_float64_memory(double* ptr) { free(ptr); }
    void release_char_memory(char* ptr) { free(ptr); }

    float* get_tensor_grad(Tensor* obj) { return obj->grad; }
    int get_tensor_size(Tensor* obj) { return obj->data_size; }
    bool get_tensor_requieres_grad(Tensor* obj) { return obj->requires_grad; }

    Tensor* add(Tensor* a, Tensor* b) {
        int out_size = a->data_size >= b->data_size ? a->data_size : b->data_size;
        float* result = (float*)malloc(out_size * sizeof(float));

        if (a->data_size == 1 && b->data_size != 1) {
            cpu_parallel_otm(add_float32, a->nthreads, out_size, result, a->data, b->data);
        } else if ((b->data_size == 1 && a->data_size != 1)) {
            cpu_parallel_otm(add_float32, a->nthreads, out_size, result, b->data, a->data);
        } else {
            cpu_paralell(add_float32, a->nthreads, out_size, result, a->data, b->data);
        }

        Tensor* out = new Tensor(result, out_size, a->nthreads, (a->requires_grad || b->requires_grad), {a, b});
        if (b->data_size == 1 && a->data_size != 1) {
            (*out)._backward = addition_backward(b, a, out);
        } else {
            (*out)._backward = addition_backward(a, b, out);
        }
        return out;

    }

    Tensor* sub(Tensor* a, Tensor* b) {
        int out_size = a->data_size >= b->data_size ? a->data_size : b->data_size;
        float* result = (float*)malloc(out_size * sizeof(float));
        
        if (a->data_size == 1 && b->data_size != 1) {
            cpu_parallel_otm(sub_float32, a->nthreads, out_size, result, a->data, b->data);
        } else if ((b->data_size == 1 && a->data_size != 1)) {
            cpu_parallel_otm(sub_float32, a->nthreads, out_size, result, b->data, a->data);
        } else {
            cpu_paralell(sub_float32, a->nthreads, out_size, result, a->data, b->data);
        }

        Tensor* out = new Tensor(result, out_size, a->nthreads, (a->requires_grad || b->requires_grad), {a, b});
        if (b->data_size == 1 && a->data_size != 1) {
            (*out)._backward = substraction_backward(b, a, out);
        } else {
            (*out)._backward = substraction_backward(a, b, out);
        }
        return out;
    }

    Tensor* mul(Tensor* a, Tensor* b) {
        int out_size = a->data_size >= b->data_size ? a->data_size : b->data_size;
        float* result = (float*)malloc(out_size * sizeof(float));

        if (a->data_size == 1 && b->data_size != 1) {
            cpu_parallel_otm(mul_float32, a->nthreads, out_size, result, a->data, b->data);
        } else if ((b->data_size == 1 && a->data_size != 1)) {
            // Python will always go here
            cpu_parallel_otm(mul_float32, a->nthreads, out_size, result, b->data, a->data);
        } else {
            cpu_paralell(mul_float32, a->nthreads, out_size, result, a->data, b->data);
        }

        Tensor* out = new Tensor(result, out_size, a->nthreads, (a->requires_grad || b->requires_grad), {a, b});
        if (b->data_size == 1 && a->data_size != 1) {
            (*out)._backward = multiplication_backward(b, a, out);
        } else {
            (*out)._backward = multiplication_backward(a, b, out);
        }
        return out;
    }

    Tensor* dot(Tensor* a, Tensor* b) {
        float* result = (float*)malloc(sizeof(float));

        cpu_paralell(dot_float32, a->nthreads, a->data_size, result, a->data, b->data);

        Tensor* out = new Tensor(result, 1, a->nthreads, (a->requires_grad || b->requires_grad), {a, b});
        (*out)._backward = dot_backward(a, b, out);
        return out;
    }

    Tensor* divv(Tensor* a, Tensor* b) {
        int out_size = a->data_size >= b->data_size ? a->data_size : b->data_size;
        float* result = (float*)malloc(out_size * sizeof(float));
        
        if (a->data_size == 1 && b->data_size != 1) {
            cpu_parallel_otm(div_float32, a->nthreads, out_size, result, a->data, b->data);
        } else if ((b->data_size == 1 && a->data_size != 1)) {
            cpu_parallel_otm(div_float32, a->nthreads, out_size, result, b->data, a->data);
        } else {
            cpu_paralell(div_float32, a->nthreads, out_size, result, a->data, b->data);
        }

        Tensor* out = new Tensor(result, out_size, a->nthreads, (a->requires_grad || b->requires_grad), {a, b});
        if (b->data_size == 1 && a->data_size != 1) {
            (*out)._backward = division_backward(b, a, out);
        } else {
            (*out)._backward = division_backward(a, b, out);
        }
        return out;
    }

    Tensor* poww(Tensor* a, float b) {
        float* result = (float*)malloc(a->data_size * sizeof(float));
        
        cpu_paralell(pow_float32, a->nthreads, a->data_size, result, a->data, NULL, b);

        Tensor* out = new Tensor(result, a->data_size, a->nthreads, (a->requires_grad), {a});
        (*out)._backward = pow_backward(a, b, out);
        return out;
    }

    Tensor* relu(Tensor* a) {
        float* result = (float*)malloc(a->data_size * sizeof(float));
        
        cpu_paralell(relu_float32, a->nthreads, a->data_size, result, a->data, NULL);

        Tensor* out = new Tensor(result, a->data_size, a->nthreads, (a->requires_grad), {a});
        (*out)._backward = relu_backward(a, out);
        return out;
    }

    Tensor* sigmoid(Tensor* a) {
        float* result = (float*)malloc(a->data_size * sizeof(float));
        
        cpu_paralell(sigmoid_float32, a->nthreads, a->data_size, result, a->data, NULL);

        Tensor* out = new Tensor(result, a->data_size, a->nthreads, (a->requires_grad), {a});
        (*out)._backward = sigmoid_backward(a, out);
        return out;
    }

    Tensor* tanhh(Tensor* a) {
        float* result = (float*)malloc(a->data_size * sizeof(float));
        
        cpu_paralell(tanh_float32, a->nthreads, a->data_size, result, a->data, NULL);

        Tensor* out = new Tensor(result, a->data_size, a->nthreads, (a->requires_grad), {a});
        (*out)._backward = tanh_backward(a, out);
        return out;
    }

    Tensor* expp(Tensor* a) {
        float* result = (float*)malloc(a->data_size * sizeof(float));

        cpu_paralell(exp_float32, a->nthreads, a->data_size, result, a->data, NULL);

        Tensor* out = new Tensor(result, a->data_size, a->nthreads, (a->requires_grad), {a});
        (*out)._backward = exp_backward(a, out);
        return out;
    }

    Tensor* full(float data, int size, int nthreads, bool requires_grad) {
        float* result = (float*)malloc(size * sizeof(float));

        full_float32_array(result, data, size, nthreads);

        return new Tensor(result, size, nthreads, requires_grad, {});
    }

    Tensor* ones(int size, int nthreads, bool requires_grad) {
        return full(1, size, nthreads, requires_grad);
    }

    Tensor* zeros(int size, int nthreads, bool requires_grad) {
        return full(0, size, nthreads, requires_grad);
    }

    Tensor* _random(int size, int seed, int lower_bound, int upper_bound) {
        random_device rd;
        mt19937_64 mtrand;
        if (seed == 0) {
            mtrand.seed(rd());
        } else {
            mtrand.seed(seed);
        }
        
        float data[size];
        for (int i = 0; i < size; i++) {
            data[i] = static_cast<float>((mtrand() % (upper_bound - lower_bound + 1)) + lower_bound);
        }

        return new Tensor(data, size);
    }

    Tensor* _randn(int size, int seed, float M, float SD, bool spare) {
        // M stands for mean
        // SD stands for Standard Deviation
        random_device rd;
        mt19937_64 mtrand;
        if (seed == 0) {
            mtrand.seed(rd());
        } else {
            mtrand.seed(seed);
        }

        //float* data = new float[size];
        //float data[size];
        float* data = (float*)malloc(size*sizeof(float));
        float const1 = 1.0f/(sqrt(2.0f*M_PI*SD));
        float const2 = 2.0f*SD;

        for (int i = 0; i < size; i++) {
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
            data[i] = val;
        }

        return new Tensor(data, size);
    }

    Tensor* shuffle(Tensor* a, int seed) {
        random_device rd;
        mt19937_64 mtrand;
        if (seed == 0) {
            mtrand.seed(rd());
        } else {
            mtrand.seed(seed);
        }

        int size = a->data_size;
        float shuffled[size];

        memcpy(shuffled, a->data, size * sizeof(float));

        std::shuffle(shuffled, shuffled + size, mtrand);

        return new Tensor(shuffled, size);
    }
    

    void backward(Tensor* a) {
        a->backward();
    }

}

/* 
    Command to export to .SO:

        clang++ -Ofast -fPIC -shared -o tensor.so tensor.cpp

*/