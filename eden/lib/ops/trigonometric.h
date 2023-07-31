#ifndef __TRIGONOMETRIC_H__
#define __TRIGONOMETRIC_H__

#include <iostream>
#include <functional>
#include <math.h>
#include "../matrix/matrix.h"

function<void()> matrix_cos_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += -sin(a->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_sin_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += cos(a->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_tan_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    float sec = 1.0f/cos(a->data[i][j]);
                    a->grad[i][j] += sec*sec * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_asin_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (1/sqrt(1-(a->data[i][j]*a->data[i][j]))) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_acos_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (-1/sqrt(1-(a->data[i][j]*a->data[i][j]))) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_cosh_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += sinh(a->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_sinh_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += cosh(a->data[i][j]) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_atan_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (1/(1+(a->data[i][j]*a->data[i][j]))) * out->grad[i][j];
                }
            }
        }
    };
}

#endif