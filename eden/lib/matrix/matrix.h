#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>
#include <list>
#include <typeinfo>
#include <functional>
#include <math.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <functional>
#include <chrono>
#include <stack>
#include <cstdint>
#include <sstream>
#include <string.h>
#include <thread>
#include "../cpu/mtmatrix.h"

using namespace std;

struct Matrix {
    private:
        thread thread1;
        int64_t memory_to_int64(const void* memory) {
            return *reinterpret_cast<const int64_t*>(memory);
        }
        string _hash;
        void setHash() {
            std::stringstream ss;
            ss << hex << &(*this);
            string hexstr = ss.str();

            std::hash<string> hashFunction;
            size_t hash = hashFunction(hexstr);
            _hash = std::to_string(hash);
        }
    public:
        function<void()> _backward;
        bool requires_grad;
        float** data;
        list<Matrix*> prev;
        float** grad;
        int r, c;
        int nthreads;

        bool get_requires_grad() { return requires_grad; }
        int get_data_size() { return r*c; }
        list<int> get_shape() { return {r, c}; }
        int get_nthreads() { return nthreads; }

        Matrix(float** _data, int r, int c, int nthreads = 0, bool requires_grad = false, list<Matrix*> children = {}) {
            if (nthreads == 0) this->nthreads = thread::hardware_concurrency()-1;
            else this->nthreads = nthreads;

            this->data = _data;
            this->c = c;
            this->r = r;
            this->requires_grad = requires_grad;
            if (requires_grad) {
                malloc_float32_matrix(this->grad, r, c, nthreads);
                full_float32_matrix(this->grad, r, c, 0, nthreads);
            }
            this->prev = children;
            setHash();
        }

        void print_matrix() {
            printf("Matrix(");
            if (this->r > 1) printf("[");
            for (int i = 0; i < this->r; i++) {
                if (i > 0) printf("\t");
                printf("[");
                for (int j = 0; j < this->c; j++) {
                    if (this->data[i][j] == (int)this->data[i][j]) {
                        printf("%.0f.", this->data[i][j]);
                    } else {
                        printf("%.3f", this->data[i][j]);
                    }

                    if (j != this->c-1) printf(", ");
                }
                printf("]");
                if (i != this->r-1) printf(",\n");
            }
            if (this->r > 1) printf("]");
            if (this->requires_grad) printf(", requires_grad=True");
            printf(")\n");
        }

        bool operator==(const Matrix& other) {
            if (_hash == other._hash) return true;

            return false;
        }

        bool operator!=(const Matrix& other) {
            if (_hash != other._hash) return false;

            return true;
        }

        void backward(float sgrad = 1.0f) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    this->grad[i][j] = sgrad;
                }
            }
            list<Matrix*> topo = topological_sort();

            for (Matrix* v : topo) {
                if (!v->prev.empty()) {
                    v->_backward();
                }
            }
        }

        void update_graph(float multiplier) {
            list<Matrix*> topo = topological_sort();

            for (Matrix* v : topo) {
                if (!v->prev.empty() && v->requires_grad) {
                    v->update(multiplier);
                }
            }
        }

        void update(float multiplier) {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    this->data[i][j] -= multiplier * grad[i][j];
                }
            }
        }

        void zero_grad() {
            for (int i = 0; i < r; i++) {
                for (int j = 0; j < c; j++) {
                    this->grad[i][j] = 0;
                }
            }
        }

        void topological_sort_util(stack<Matrix*>& sorted_stack, list<Matrix*>& visited) {
            visited.push_back(this);

            for (Matrix* child : prev) {
                if (find(visited.begin(), visited.end(), child) == visited.end() && child->requires_grad) {
                    child->topological_sort_util(sorted_stack, visited);
                }
            }

            sorted_stack.push(this);
        }

        list<Matrix*> topological_sort() {
            stack<Matrix*> sorted_stack;
            list<Matrix*> visited;

            topological_sort_util(sorted_stack, visited);

            list<Matrix*> order;
            while (!sorted_stack.empty()) {
                order.push_back(sorted_stack.top());
                sorted_stack.pop();
            }

            return order;
        }

        ~Matrix() {
            //delete[] data;
            //delete[] grad;
        }
};

#endif