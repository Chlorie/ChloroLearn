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

    Operand convolution_2d_with_padding(Operand input, Operand filters, const size_t stride)
    {
        const ArrayShape& input_shape = input.shape();
        const ArrayShape& filter_shape = filters.shape();
        if (stride == 0) throw IllegalArgumentException("Stride should be positive");
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
        if (input_features != filter_shape[3])
            throw MismatchedSizesException("Length of 4th dimension of filters should be the same as the amount of "
                "feature maps of the input");
        const size_t output_row = (input_row + stride - 1 - filter_row) / stride + 1;
        const size_t output_column = (input_column + stride - 1 - filter_column) / stride + 1;
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
                Array result = Array<double>::zeros(output_shape);
                for (size_t i = 0; i < filter_amount; i++)
                {
                    
                }
                return OutParam{};
            },
            [=](const BackwardParams params)
            {
                return OutParams{};
            }, output_shape);
        throw NotImplementedException("Sorry, not implemented yet O_O");
    }

    Operand max_pool_2d(Operand input, const size_t stride)
    {
        const ArrayShape& input_shape = input.shape();
        if (input_shape.size() != 3)
            throw IllegalArgumentException("Input should be 3D (2D feature maps)");
        if (stride <= 1) throw IllegalArgumentException("Stride should be at least 2");
        const size_t row = input_shape[0];
        const size_t column = input_shape[1];
        const size_t output_row = (row + stride - 1) / stride;
        const size_t output_column = (column + stride - 1) / stride;
        const size_t features = input_shape[2];
        const ArrayShape output_shape{ output_row, output_column, features };
        const size_t output_size = output_row * output_column * features;
        const auto output_index = [column = output_column, features](const size_t i, const size_t j, const size_t k)
        { return (i * column + j) * features + k; };
        Operator op(
            [=](InParams params)
            {
                const Array<double>& param = params[0];
                Array result = Array<double>::repeats(std::numeric_limits<double>::lowest(), output_shape);
                size_t index = 0;
                for (size_t i = 0; i < row; i++)
                    for (size_t j = 0; j < column; j++)
                        for (size_t k = 0; k < features; k++)
                        {
                            index++;
                            const size_t result_index = output_index(i / stride, j / stride, k);
                            if (param[index] > result[result_index]) result[result_index] = param[index];
                        }
                return result;
            }, [=](ForwardParams params)
            {
                const Array<double>& param = params.childs[0];
                StateParam state = params.state;
                Array result = Array<double>::repeats(std::numeric_limits<double>::lowest(), output_shape);
                size_t index = 0;
                for (size_t i = 0; i < row; i++)
                    for (size_t j = 0; j < column; j++)
                        for (size_t k = 0; k < features; k++)
                        {
                            index++;
                            const size_t result_index = output_index(i / stride, j / stride, k);
                            if (param[index] > result[result_index])
                            {
                                result[result_index] = param[index];
                                state[result_index] = index;
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
