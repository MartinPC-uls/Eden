#ifndef _TENSOROBJ_H_
#define _TENSOROBJ_H_

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
#include "cpu/multithreading.h"

using namespace std;

struct Tensor {
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
        float* data;
        list<Tensor*> prev;
        float* grad;
        int data_size;
        int nthreads;

        bool get_requires_grad() { return requires_grad; }
        int get_data_size() { return data_size; }
        int get_nthreads() { return nthreads; }

        Tensor(float* _data, int size, int nthreads = 0, bool requires_grad = false, list<Tensor*> children = {}) {
            if (nthreads == 0) {
                this->nthreads = thread::hardware_concurrency()-1;
            } else {
                this->nthreads = nthreads;
            }
            this->data = _data;
            this->data_size = size;
            this->requires_grad = requires_grad;
            if (requires_grad) {
                this->grad = (float*)malloc(size * sizeof(float));
                full_float32_array(this->grad, 0, size, nthreads);
            }
            this->prev = children;
            setHash();
        }

        bool operator==(const Tensor& other) {
            if (_hash == other._hash) {
                return true;
            }
            return false;
        }

        bool operator!=(const Tensor& other) {
            if (_hash != other._hash) {
                return false;
            }
            return true;
        }

        void backward() {
            for (int i = 0; i < data_size; i++) {
                this->grad[i] = 1;
            }
            list<Tensor*> topo = topological_sort();

            for (Tensor* v : topo) {
                if (!(*v).prev.empty()) {
                    (*v)._backward();
                }
            }
        }

        void topological_sort_util(stack<Tensor*>& sorted_stack, list<Tensor*>& visited) {
            visited.push_back(this);

            for (Tensor* child : prev) {
                if (find(visited.begin(), visited.end(), child) == visited.end() && child->requires_grad) {
                    child->topological_sort_util(sorted_stack, visited);
                }
            }

            sorted_stack.push(this);
        }

        list<Tensor*> topological_sort() {
            stack<Tensor*> sorted_stack;
            list<Tensor*> visited;

            topological_sort_util(sorted_stack, visited);

            list<Tensor*> order;
            while (!sorted_stack.empty()) {
                order.push_back(sorted_stack.top());
                sorted_stack.pop();
            }

            return order;
        }

        ~Tensor() {
            //delete[] data;
            //delete[] grad;
        }
};

#endif