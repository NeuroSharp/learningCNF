#ifndef Glucose_ModelHelper_h
#define Glucose_ModelHelper_h

#include <math.h>

namespace Glucose {

#define LAYER(INPUT, INPUT_SIZE, WEIGHTS, WEIGHT_SIZE, BIAS, ACTIVATION, OUTPUT)\
for (int i=0; i < WEIGHT_SIZE; i++) {       \
    double sum = 0;                         \
    for (int j=0; j < INPUT_SIZE; j++)      \
        sum += WEIGHTS[j][i] * INPUT[j];    \
    sum += BIAS[i];                         \
    OUTPUT[i] = ACTIVATION(sum);            \
}                                           

#define LAYER_1(INPUT, INPUT_SIZE, WEIGHTS, BIAS, ACTIVATION, OUTPUT)\
double sum = 0;                             \
for (int j=0; j < INPUT_SIZE; j++)          \
    sum += WEIGHTS[j] * INPUT[j];           \
OUTPUT[0] = ACTIVATION(sum +  BIAS[0]);     

class ModelHelper {
public:
    static inline double relu(double x) {
        return (x>0) ? x : 0;
    }
    static inline double no_op(double x) {
        return x;
    }
    static inline double sigmoid(double x) {
        double exp_value;
        double return_value;

        /*** Exponential calculation ***/
        exp_value = exp(-x);

        /*** Final sigmoid value ***/
        return_value = 1 / (1 + exp_value);

        return return_value;
    }
};

}
#endif