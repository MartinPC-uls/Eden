#ifndef _ACTIVATIONOPS_H_
#define _ACTIVATIONOPS_H_

#include <iostream>
#include <functional>
#include "../vector/vectorobj.h"
#include "../matrix/matrix.h"

using namespace std;

function<void()> relu_backward(Vector* a, Vector* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += ((*out).data[i] > 0) * (*out).grad[i];
        }
    };
}

function<void()> matrix_relu_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (out->data[i][j] > 0) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> sigmoid_backward(Vector* a, Vector* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (*out).data[i] * (1 - (*out).data[i]) * (*out).grad[i];
        }
    };
}

function<void()> matrix_sigmoid_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += out->data[i][j] * (1 - out->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> tanh_backward(Vector* a, Vector* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (1 - (*out).data[i]*(*out).data[i]) * (*out).grad[i];
        }
    };
}

function<void()> matrix_tanh_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (1 - out->data[i][j]*out->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_softsign_backward(Matrix* a, Matrix* out) { 
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    float v = abs(out->data[i][j])+1;
                    a->grad[i][j] += 1/(v*v) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> exp_backward(Vector* a, Vector* out) {
    return [a, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            a->grad[i] += (*out).data[i] * (*out).grad[i];
        }
    };
}

function<void()> matrix_exp_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] = out->data[i][j] * out->grad[i][j];
                }
            }
        }
    };
}

#endif