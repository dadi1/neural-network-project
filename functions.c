
#include <stdio.h>
#include <math.h>

double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double deriv_sigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}
