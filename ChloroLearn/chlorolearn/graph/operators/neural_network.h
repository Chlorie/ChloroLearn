#pragma once

#include "../operand.h"

// ReSharper disable CppInconsistentNaming

namespace chloro::operators
{
    /** \brief Reshape an operand into a column vector (shape N x 1). */
    Operand flatten(Operand input);

    /**
     * \brief Perform a 2D convolution operation with padding on the bottom-right side.
     * \param input The input operand. Should be a 3D array of shape (rows x columns x feature map amount).
     * \param filters Filters array. Should be a 4D array of shape (filter amount x filter rows x filter
     * columns x feature map amount).
     * \param stride Stride of convolution, should be a 2D shape.
     * \return The result operand.
     */
    Operand convolution_2d_with_padding(Operand input, Operand filters, const ArrayShape& stride = { 1, 1 });
    /**
     * \brief Perform a max pooling operation.
     * \param input The input operand. Should be a 3D array of shape (rows x columns x feature map amount).
     * \param pool_size Define pool size and pool stride to shape (stride x stride).
     * \return The result operand.
     */
    Operand max_pool_2d(Operand input, size_t pool_size);
    /**
     * \brief Perform a max pooling operation.
     * \param input The input operand. Should be a 3D array of shape (rows x columns x feature map amount).
     * \param pool_size Pool size, should be a 2D shape.
     * \param pool_stride Stride of pooling, should be a 2D shape.
     * \return The result operand.
     */
    Operand max_pool_2d(Operand input, const ArrayShape& pool_size, const ArrayShape& pool_stride);
}
