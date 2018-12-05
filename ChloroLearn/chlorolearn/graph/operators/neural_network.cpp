#include <numeric>

#include "basic_operators.h"
#include "neural_network.h"
#include "../../utility/utility.h"

// ReSharper disable CppInconsistentNaming

namespace chloro::operators
{
    Operand flatten(Operand input)
    {
        const ArrayShape& shape = input.shape();
        const size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
        return reshape(std::move(input), { int(size),1 });
    }

    Operand convolution_2d_with_padding(Operand input, Operand filters, const ArrayShape& stride)
    {
        const ArrayShape& input_shape = input.shape();
        const ArrayShape& filter_shape = filters.shape();
        if (stride.size() != 2) throw IllegalArgumentException("Stride should be a 2D array shape");
        if (stride[0] == 0 || stride[1] == 0) throw IllegalArgumentException("Stride should be positive");
        if (input_shape.size() != 3)
            throw IllegalArgumentException("Input should be 3D (2D feature maps)");
        if (filter_shape.size() != 4)
            throw IllegalArgumentException("Filter array should be 4D");
        // filter_shape = { filter_amount, filter_row, filter_column, input_features };
        const size_t input_row = input_shape[0];
        const size_t input_column = input_shape[1];
        const size_t input_features = input_shape[2];
        const size_t filter_amount = filter_shape[0];
        const size_t filter_row = filter_shape[1];
        const size_t filter_column = filter_shape[2];
        const size_t stride_row = stride[0];
        const size_t stride_column = stride[1];
        if (input_features != filter_shape[3])
            throw MismatchedSizesException("Length of 4th dimension of filters should be the same as the amount of "
                "feature maps of the input");
        const size_t output_row = (input_row + stride_row - 1 - filter_row) / stride_row + 1;
        const size_t output_column = (input_column + stride_column - 1 - filter_column) / stride_column + 1;
        const ArrayShape output_shape{ output_row, output_column, filter_amount };
        const auto input_index = [=](const size_t i, const size_t j, const size_t k)
        { return (i * input_column + j) * input_features + k; };
        const auto output_index = [=](const size_t i, const size_t j, const size_t k)
        { return (i * output_column + j) * filter_amount + k; };
        const auto filter_index = [=](const size_t i, const size_t j, const size_t k, const size_t l)
        { return ((i * filter_row + j) * filter_column + k) * input_features + l; };
        Operator op(
            [=](InParams params)
            {
                const Array<double>& input_value = params[0];
                const Array<double>& filter_value = params[1];
                Array result = Array<double>::zeros(output_shape);
                for (size_t i = 0; i < filter_amount; i++)
                    for (size_t j = 0; j < output_row; j++)
                        for (size_t k = 0; k < output_column; k++)
                        {
                            const size_t result_index = output_index(j, k, i);
                            const size_t max_row = std::min(filter_row, input_row - j * stride_row);
                            const size_t max_column = std::min(filter_column, input_column - k * stride_column);
                            for (size_t l = 0; l < max_row; l++)
                                for (size_t m = 0; m < max_column; m++)
                                    for (size_t n = 0; n < input_features; n++)
                                    {
                                        const size_t input_i = input_index(j * stride_row + l, k * stride_column + m, n);
                                        const size_t filter_i = filter_index(i, l, m, n);
                                        result[result_index] += input_value[input_i] * filter_value[filter_i];
                                    }
                        }
                return result;
            },
            [=](const BackwardParams params)
            {
                Array input_grad = Array<double>::zeros(input_shape);
                Array filter_grad = Array<double>::zeros(filter_shape);
                const Array<double>& grad = params.gradient;
                const Array<double>& input_value = params.childs[0];
                const Array<double>& filter_value = params.childs[1];
                for (size_t i = 0; i < filter_amount; i++)
                    for (size_t j = 0; j < output_row; j++)
                        for (size_t k = 0; k < output_column; k++)
                        {
                            const size_t result_index = output_index(j, k, i);
                            const size_t max_row = std::min(filter_row, input_row - j * stride_row);
                            const size_t max_column = std::min(filter_column, input_column - k * stride_column);
                            for (size_t l = 0; l < max_row; l++)
                                for (size_t m = 0; m < max_column; m++)
                                    for (size_t n = 0; n < input_features; n++)
                                    {
                                        const size_t input_i = input_index(j * stride_row + l, k * stride_column + m, n);
                                        const size_t filter_i = filter_index(i, l, m, n);
                                        input_grad[input_i] += grad[result_index] * filter_value[filter_i];
                                        filter_grad[filter_i] += grad[result_index] * input_value[input_i];
                                    }
                        }
                return OutParams{ input_grad, filter_grad };
            }, output_shape);
        return Operand::join(std::move(op), { std::move(input), std::move(filters) });
    }

    Operand max_pool_2d(Operand input, const size_t pool_size)
    {
        return max_pool_2d(std::move(input),
            { pool_size, pool_size },
            { pool_size, pool_size });
    }

    Operand max_pool_2d(Operand input, const ArrayShape& pool_size, const ArrayShape& pool_stride)
    {
        const ArrayShape& input_shape = input.shape();
        if (input_shape.size() != 3)
            throw IllegalArgumentException("Input should be 3D (2D feature maps)");
        if (pool_size.size() != 2) throw IllegalArgumentException("Pool size should be a 2D shape");
        if (pool_stride.size() != 2) throw IllegalArgumentException("Pool stride should be a 2D array shape");
        if (pool_size[0] == 0 || pool_size[1] == 0) throw IllegalArgumentException("Pool size should be positive");
        if (pool_stride[0] == 0 || pool_stride[1] == 0) throw IllegalArgumentException("Pool stride should be positive");
        const size_t input_row = input_shape[0];
        const size_t input_column = input_shape[1];
        const size_t pool_row = pool_size[0];
        const size_t pool_column = pool_size[1];
        const size_t stride_row = pool_stride[0];
        const size_t stride_column = pool_stride[1];
        const size_t output_row = (input_row + stride_row - 1 - pool_row) / stride_row + 1;
        const size_t output_column = (input_column + stride_column - 1 - pool_column) / stride_column + 1;
        const size_t features = input_shape[2];
        const ArrayShape output_shape{ output_row, output_column, features };
        const size_t output_size = output_row * output_column * features;
        const auto input_index = [=](const size_t i, const size_t j, const size_t k)
        { return (i * input_column + j) * features + k; };
        const auto output_index = [=](const size_t i, const size_t j, const size_t k)
        { return (i * output_column + j) * features + k; };
        Operator op(
            [=](InParams params)
            {
                const Array<double>& param = params[0];
                Array result = Array<double>::repeats(std::numeric_limits<double>::lowest(), output_shape);
                size_t result_index = 0;
                for (size_t i = 0; i < output_row; i++)
                    for (size_t j = 0; j < output_column; j++)
                    {
                        const size_t max_row = std::min(pool_row, input_row - i * stride_row);
                        const size_t max_column = std::min(pool_column, input_column - j * stride_column);
                        for (size_t k = 0; k < features; k++)
                        {
                            for (size_t l = 0; l < max_row; l++)
                                for (size_t m = 0; m < max_column; m++)
                                {
                                    const size_t input_i = input_index(i * stride_row + l, j * stride_column + m, k);
                                    if (param[input_i] > result[result_index]) result[result_index] = param[input_i];
                                }
                            result_index++;
                        }
                    }
                return result;
            }, [=](ForwardParams params)
            {
                const Array<double>& param = params.childs[0];
                StateParam state = params.state;
                Array result = Array<double>::repeats(std::numeric_limits<double>::lowest(), output_shape);
                size_t result_index = 0;
                for (size_t i = 0; i < output_row; i++)
                    for (size_t j = 0; j < output_column; j++)
                    {
                        const size_t max_row = std::min(pool_row, input_row - i * stride_row);
                        const size_t max_column = std::min(pool_column, input_column - j * stride_column);
                        for (size_t k = 0; k < features; k++)
                        {
                            for (size_t l = 0; l < max_row; l++)
                                for (size_t m = 0; m < max_column; m++)
                                {
                                    const size_t input_i = input_index(i * stride_row + l, j * stride_column + m, k);
                                    if (param[input_i] > result[result_index])
                                    {
                                        result[result_index] = param[input_i];
                                        state[result_index] = input_i;
                                    }
                                }
                            result_index++;
                        }
                    }
                return result;
            }, [=](BackwardParams params)
            {
                const Array<double>& gradient = params.gradient;
                StateParam state = params.state;
                Array result = Array<double>::zeros(input_shape);
                for (size_t i = 0; i < output_size; i++) result[size_t(state[i])] = gradient[i];
                return OutParams{ result };
            }, output_shape);
        return Operand::join(std::move(op), { std::move(input) });
    }
}
