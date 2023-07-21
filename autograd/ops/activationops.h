#ifndef _ACTIVATIONOPS_H_
#define _ACTIVATIONOPS_H_

#include <iostream>
#include <functional>
#include "../tensorobj.h"

using namespace std;

function<void()> relu_backward(Tensor* a, Tensor* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += ((*out).data[i] > 0) * (*out).grad[i];
        }
    };
}

function<void()> sigmoid_backward(Tensor* a, Tensor* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (*out).data[i] * (1 - (*out).data[i]) * (*out).grad[i];
        }
    };
}

function<void()> tanh_backward(Tensor* a, Tensor* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (1 - (*out).data[i]*(*out).data[i]) * (*out).grad[i];
        }
    };
}

function<void()> exp_backward(Tensor* a, Tensor* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (*out).data[i] * (*out).grad[i];
        }
    };
}

#endif