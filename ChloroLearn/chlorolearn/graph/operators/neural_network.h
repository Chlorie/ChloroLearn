#pragma once

#include "../operand.h"

// ReSharper disable CppInconsistentNaming

namespace chloro::operators
{
    /** \brief Reshape an operand into a column vector (shape N x 1). */
    Operand flatten(Operand input);

    // Convolution and pooling
    Operand convolution_2d_with_padding(Operand input, Operand filters, size_t stride = 1);
    Operand max_pool_2d(Operand input, size_t stride);
}
