#ifndef __TRANSFORMS_H__
#define __TRANSFORMS_H__

#include <iostream>
#include <functional>
#include "../matrix/matrix.h"

function<void()> matrix_transpose_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += out->grad[j][i];
                }
            }
        }
    };
}


#endif