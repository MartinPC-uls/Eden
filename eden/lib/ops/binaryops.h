#ifndef _BINARYOPS_H_
#define _BINARYOPS_H_

#include <iostream>
#include <functional>
#include "../matrix/matrix.h"
#include "../vector/vectorobj.h"

using namespace std;

function<void()> addition_backward(Vector* a, Vector* b, Vector* out) {
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

function<void()> matrix_addition_backward(Matrix* a, Matrix* b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->r == b->r && a->c == b->c) { // if dimensions are the same
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][j] += out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && a->c == b->c) { // if 'b' is a vector
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][j] += out->grad[i][j];
                    }
                }
            }
        } else if (a->r == b->r && b->c == 1) { // if 'b' is a matrix of a.c==b.c x 1
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][0] += out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][0] += out->grad[i][j];
                    }
                }
            }
        }
    };
}

function<void()> matrix_sum_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += out->grad[0][0];
                }
            }
        }
    };
}

function<void()> matrix_substraction_backward(Matrix* a, Matrix* b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->r == b->r && a->c == b->c) { // if dimensions are the same
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][j] -= out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && a->c == b->c) { // if 'b' is a vector
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][j] -= out->grad[i][j];
                    }
                }
            }
        } else if (a->r == b->r && b->c == 1) { // if 'b' is a matrix of a.c==b.c x 1
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][0] -= out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += out->grad[0][0];
                    }
                    if (b->requires_grad) {
                        b->grad[0][0] -= out->grad[0][0];
                    }
                }
            }
        }
    };
}

function<void()> substraction_backward(Vector* a, Vector* b, Vector* out) {
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

function<void()> multiplication_backward(Vector* a, Vector* b, Vector* out) {
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

function<void()> matrix_multiplication_backward(Matrix* a, Matrix* b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->r == b->r && a->c == b->c) { // if dimensions are the same
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += b->data[i][j] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][j] += a->data[i][j] * out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && a->c == b->c) { // if 'b' is a vector
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += b->data[0][j] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][j] += a->data[i][j] * out->grad[i][j];
                    }
                }
            }
        } else if (a->r == b->r && b->c == 1) { // if 'b' is a matrix of a.c==b.c x 1
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += b->data[i][0] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][0] += a->data[i][j] * out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += b->data[0][0] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][0] += a->data[i][j] * out->grad[i][j];
                    }
                }
            }
        }
    };
}

function<void()> dot_backward(Vector* a, Vector* b, Vector* out) {
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

function<void()> division_backward(Vector* a, Vector* b, Vector* out) {
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

function<void()> matrix_division_backward(Matrix* a, Matrix* b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->r == b->r && a->c == b->c) { // if dimensions are the same
            for (int i = 0; i < b->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += 1 / b->data[i][j] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][j] += -a->data[i][j] / (b->data[i][j]*b->data[i][j]) * out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && a->c == b->c) { // if 'b' is a vector
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += 1 / b->data[0][j] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][j] += -a->data[i][j] / (b->data[0][j]*b->data[0][j]) * out->grad[i][j];
                    }
                }
            }
        } else if (a->r == b->r && b->c == 1) { // if 'b' is a matrix of a.r==b.r x 1
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += 1 / b->data[i][0] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[i][0] += -a->data[i][j] / (b->data[i][0]*b->data[i][0]) * out->grad[i][j];
                    }
                }
            }
        } else if (b->r == 1 && b->c == 1) { // if 'b' is a scalar (1 x 1)
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    if (a->requires_grad) {
                        a->grad[i][j] += 1 / b->data[0][0] * out->grad[i][j];
                    }
                    if (b->requires_grad) {
                        b->grad[0][0] += -a->data[i][j] / (b->data[0][0]*b->data[0][0]) * out->grad[i][j];
                    }
                }
            }
        }
    };
}

function<void()> matrix_matmul_backward(Matrix* a, Matrix* b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->r == 1 && b->r == 1) { // If 'a' and 'b' are vectors, returns a 1x1 matrix (scalar)
            for (int i = 0; i < a->c; i++) {
                if (a->requires_grad) {
                    a->grad[0][i] += b->data[0][i] * out->grad[0][0];
                }
                if (b->requires_grad) {
                    b->grad[0][i] += a->data[0][i] * out->grad[0][0];
                }
            }
        } else if (b->r == 1 && a->c == b->c) { // If 'b' is a vector
            for (int i = 0; i < out->c; i++) {
                for (int k = 0; k < a->c; k++) {
                    if (a->requires_grad) {
                        a->grad[i][k] += b->data[0][k] * out->grad[0][i];
                    }
                    if (b->requires_grad) {
                        b->grad[0][k] += a->data[i][k] * out->grad[0][i];
                    }
                }
            }
        } else if (a->c == b->r) {
            for (int i = 0; i < out->r; i++) {
                for (int j = 0; j < out->c; j++) {
                    for (int k = 0; k < a->c; k++) {
                        if (a->requires_grad) {
                            a->grad[i][k] += b->data[k][j] * out->grad[i][j];
                        }
                        if (b->requires_grad) {
                            b->grad[k][j] += a->data[i][k] * out->grad[i][j];
                        }
                    }
                }
            }
        }
    };
}

function<void()> pow_backward(Vector* a, float b, Vector* out) {
    return [a, b, out]() mutable {
        for (int i = 0; i < a->data_size; i++) {
            if (a->requires_grad) {
                a->grad[i] += (b * pow(a->data[i], b-1)) * (*out).grad[i];
            }
        }
    };
}

function<void()> matrix_pow_backward(Matrix* a, float b, Matrix* out) {
    return [a, b, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (b * pow(a->data[i][j], b-1)) * out->grad[i][j];
                }
            }
        }
    };
}

function<void()> matrix_sqrt_backward(Matrix* a, Matrix* out) {
    return [a, out]() mutable {
        if (a->requires_grad) {
            for (int i = 0; i < a->r; i++) {
                for (int j = 0; j < a->c; j++) {
                    a->grad[i][j] += (1/(2*sqrt(a->data[i][j]))) * out->grad[i][j];
                }
            }
        }
    };
}

#endif