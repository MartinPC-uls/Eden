#ifndef __PARAMETERS_H__
#define __PARAMETERS_H__
#include "./matrix/matrix.h"

class Parameters {
    public:
        Matrix** parameters;
        int size;

        Parameters(Matrix** parameters, int size) {
            this->parameters = parameters;
            this->size = size;
        }

        void update(float learning_rate) {
            // Single threaded at the moment
            for (int i = 0; i < size; i++) {
                parameters[i]->update(learning_rate);
            }
        }
};

#endif