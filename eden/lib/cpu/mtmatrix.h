#ifndef __MTMATRIX_H__
#define __MTMATRIX_H__

#include <thread>
#include <list>
#include <stdlib.h>
#include <vector>
#include <stdio.h>
#include <iostream>

using namespace std;

void _init_threads(thread* t, int nthreads) { for (int i = 0; i < nthreads; i++) t[i].join(); }

void malloc_float32_matrix(float**& matrix, int rows, int cols, int nthreads) {
    matrix = new float*[rows];

    if (nthreads > 1 && cols > 1 && rows > 1) {
        if (nthreads > rows) nthreads = rows;

        vector<thread> threads;

        int chunk_size = rows / nthreads;
        int remainder = rows % nthreads;
        int start = 0;
        int finish = 0;

        for (int i = 0; i < nthreads; i++) {
            finish = start + chunk_size + (i < remainder);
            threads.emplace_back([start, finish, cols, &matrix]() {
                for (int j = start; j < finish; j++) {
                    matrix[j] = new float[cols];
                }
            });
            start = finish;
        }
        
        for (thread& thread : threads) {
            thread.join();
        }
    } else {
        for (int i = 0; i < rows; i++) {
            matrix[i] = new float[cols];
        }
    }
}

void full_float32_matrix(float** matrix, int rows, int cols, float data, int nthreads) {
    if (nthreads > 1 && cols > 1 && rows > 1) {
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        if (rows > cols) {
            if (nthreads > rows) nthreads = rows;
            int chunk_size = rows / nthreads;
            int remainder = rows % nthreads;
            for (int i = 0; i < nthreads; i++) {
                finish = start + chunk_size + (i < remainder);
                t[i] = thread([start, finish, matrix, data, cols]() {
                    for (int r = start; r < finish; r++) {
                        for (int c = 0; c < cols; c++) {
                            matrix[r][c] = data;
                        }
                    }
                });
                start = finish;
            }

            _init_threads(t, nthreads);
        } else {
            if (nthreads > cols) nthreads = cols;
            int chunk_size = cols / nthreads;
            int remainder = cols % nthreads;
            for (int i = 0; i < nthreads; i++) {
                finish = start + chunk_size + (i < remainder);
                t[i] = thread([start, finish, matrix, data, rows]() {
                    for (int c = start; c < finish; c++) {
                        for (int r = 0; r < rows; r++) {
                            matrix[r][c] = data;
                        }
                    }
                });
                start = finish;
            }

            for (int i = 0; i < nthreads; i++) {
                t[i].join();
            }
        }
    } else {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                matrix[r][c] = data;
            }
        }
    }
}

void fill_float32_matrix(float** matrix, int rows, int cols, float** data, int nthreads) {
    if (nthreads > 1 && cols > 1 && rows > 1) {
        thread t[nthreads];
        int start = 0;
        int finish = 0;
        if (rows > cols) {
            if (nthreads > rows) nthreads = rows;
            int chunk_size = rows / nthreads;
            int remainder = rows % nthreads;
            for (int i = 0; i < nthreads; i++) {
                finish = start + chunk_size + (i < remainder);
                t[i] = thread([start, finish, matrix, data, cols]() {
                    for (int r = start; r < finish; r++) {
                        for (int c = 0; c < cols; c++) {
                            matrix[r][c] = data[r][c];
                        }
                    }
                });
                start = finish;
            }

            _init_threads(t, nthreads);
        } else {
            if (nthreads > cols) nthreads = cols;
            int chunk_size = cols / nthreads;
            int remainder = cols % nthreads;
            for (int i = 0; i < nthreads; i++) {
                finish = start + chunk_size + (i < remainder);
                t[i] = thread([start, finish, matrix, data, rows]() {
                    for (int c = start; c < finish; c++) {
                        for (int r = 0; r < rows; r++) {
                            matrix[r][c] = data[r][c];
                        }
                    }
                });
                start = finish;
            }

            for (int i = 0; i < nthreads; i++) {
                t[i].join();
            }
        }
    } else {
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                matrix[r][c] = data[r][c];
            }
        }
    }
}


#endif