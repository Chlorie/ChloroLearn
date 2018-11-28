#pragma once

#include <cmath>

#include "../operand.h"

namespace chloro::operators
{
    Operand identity(Operand operand);

    Operand add(Operand left, Operand right);
    Operand operator+(Operand left, Operand right);
    Operand subtract(Operand left, Operand right);
    Operand operator-(Operand left, Operand right);
    Operand multiply(Operand left, Operand right);
    Operand operator*(Operand left, Operand right);
    Operand divide(Operand left, Operand right);
    Operand operator/(Operand left, Operand right);
    Operand matrix_multiply(Operand left, Operand right);
    Operand dot(Operand left, Operand right);

    Operand repeat(Operand scalar, const ArrayShape& shape);
    Operand reshape(Operand input, const ArrayShape& shape);
    Operand sum(Operand operand);

    Operand power(Operand base, double exponent);
    Operand exp(Operand exponent, double base = std::exp(1.0));
}
