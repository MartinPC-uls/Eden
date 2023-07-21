#ifndef _BINARYOPS_H_
#define _BINARYOPS_H_

#include <iostream>
#include <functional>
#include "../tensorobj.h"

using namespace std;

function<void()> addition_backward(Tensor* a, Tensor* b, Tensor* out) {
   return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += out->grad[i];
            }
            if (b->requires_grad) {
                b->grad[i] += out->grad[i];
            }
        }
    };
}

function<void()> substraction_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += out->grad[i];
            }
            if (b->requires_grad) {
                b->grad[i] -= out->grad[i];
            }
        }
    };
}

function<void()> multiplication_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += b->data[i] * out->grad[i];
            }
            if (b->requires_grad) {
                b->grad[i] += a->data[i] * out->grad[i];
            }
            // previously b was declared as (*out).prev.back(), which is the same pointer as b.
        }
    };
}

function<void()> dot_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += b->data[i] * out->grad[0];
            }
            if (b->requires_grad) {
                b->grad[i] += a->data[i] * out->grad[0];
            }
        }
    };
}

function<void()> division_backward(Tensor* a, Tensor* b, Tensor* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += 1 / b->data[i] * (*out).grad[i];
            }
            if (b->requires_grad) {
                b->grad[i] += -a->data[i] / (b->data[i]*b->data[i]) * out->grad[i];
            }
        }
    };
}

function<void()> pow_backward(Tensor* a, float b, Tensor* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += (b * pow(a->data[i], b-1)) * (*out).grad[i];
            }
        }
    };
}

#endif