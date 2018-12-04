#include <functional>

#include "basic_operators.h"
#include "../nodes/operator.h"
#include "../../basic/array.h"

namespace chloro
{
    Operand operator+(Operand left, Operand right) { return operators::add(std::move(left), std::move(right)); }
    Operand operator-(Operand left, Operand right) { return operators::subtract(std::move(left), std::move(right)); }
    Operand operator*(Operand left, Operand right) { return operators::multiply(std::move(left), std::move(right)); }
    Operand operator/(Operand left, Operand right) { return operators::divide(std::move(left), std::move(right)); }

    namespace operators
    {
        Operand identity(Operand operand)
        {
            Operator op([](InParams params) { return params[0]; },
                [](const BackwardParams params) { return OutParams{ params.gradient }; }, operand.shape());
            return Operand::join(std::move(op), { std::move(operand) });
        }

        Operand add(Operand left, Operand right)
        {
            Operator op([](InParams params) { return params[0] + params[1]; },
                [](const BackwardParams params) { return OutParams{ params.gradient, params.gradient }; }, left.shape());
            return Operand::join(std::move(op), { std::move(left), std::move(right) });
        }

        Operand subtract(Operand left, Operand right)
        {
            Operator op([](InParams params) { return params[0] - params[1]; },
                [](const BackwardParams params) { return OutParams{ params.gradient, -params.gradient }; }, left.shape());
            return Operand::join(std::move(op), { std::move(left), std::move(right) });
        }

        Operand multiply(Operand left, Operand right)
        {
            Operator op([](InParams params) { return params[0] * params[1]; },
                [](const BackwardParams params)
                {
                    InParam gradient = params.gradient;
                    InParams childs = params.childs;
                    return OutParams
                    {
                        childs[1] * gradient,
                        childs[0] * gradient
                    };
                }, left.shape());
            return Operand::join(std::move(op), { std::move(left), std::move(right) });
        }

        Operand divide(Operand left, Operand right)
        {
            Operator op([](InParams params) { return params[0] / params[1]; },
                [](const BackwardParams params)
                {
                    InParam left = params.childs[0];
                    InParam right = params.childs[1];
                    InParam gradient = params.gradient;
                    return OutParams
                    {
                        gradient / right,
                        -left / right / right * gradient
                    };
                }, left.shape());
            return Operand::join(std::move(op), { std::move(left), std::move(right) });
        }

        Operand matrix_multiply(Operand left, Operand right)
        {
            if (left.shape().size() != 2 || right.shape().size() != 2)
                throw MismatchedSizesException("The operands are not matrices");
            const size_t left_row = left.shape()[0];
            const size_t left_col = left.shape()[1];
            if (left_col != right.shape()[0])
                throw MismatchedSizesException("The two matrices cannot be multiplied");
            const size_t right_col = right.shape()[1];
            const ArrayShape shape{ left_row, right_col };
            Operator op(
                [=](InParams params)
                {
                    const Array<double>& first = params[0];
                    const Array<double>& second = params[1];
                    Array result = Array<double>::zeros(shape);
                    for (size_t i = 0; i < left_row; i++)
                        for (size_t j = 0; j < right_col; j++)
                            for (size_t k = 0; k < left_col; k++)
                                result[i * right_col + j] += first[i * left_col + k] * second[k * right_col + j];
                    return result;
                },
                [=](const BackwardParams params)
                {
                    InParam first = params.childs[0];
                    InParam second = params.childs[1];
                    InParam gradient = params.gradient;
                    Array left_grad = Array<double>::zeros(first.shape());
                    Array right_grad = Array<double>::zeros(second.shape());
                    for (size_t i = 0; i < left_row; i++)
                        for (size_t j = 0; j < left_col; j++)
                            for (size_t k = 0; k < right_col; k++)
                                left_grad[i * left_col + j] += gradient[i * right_col + k] * second[j * right_col + k];
                    for (size_t i = 0; i < left_col; i++)
                        for (size_t j = 0; j < right_col; j++)
                            for (size_t k = 0; k < left_row; k++)
                                right_grad[i * right_col + j] += first[k * left_col + i] * gradient[k * right_col + j];
                    return OutParams{ left_grad, right_grad };
                }, shape);
            return Operand::join(std::move(op), { std::move(left), std::move(right) });
        }

        Operand dot(Operand left, Operand right) { return sum(std::move(left) * std::move(right)); }

        Operand repeat(Operand scalar, const ArrayShape& shape)
        {
            const ArrayShape& scalar_shape = scalar.shape();
            if (scalar_shape.size() != 1 || scalar_shape[0] != 1)
                throw MismatchedSizesException("Repeated value isn't a scalar");
            Operator op([&](InParams params) { return Array<double>::repeats(params[0].get()[0], shape); },
                [](const BackwardParams params) { return OutParams{ params.gradient.accumulate(0) }; }, shape);
            return Operand::join(std::move(op), { std::move(scalar) });
        }

        Operand reshape(Operand input, const DefaultableArrayShape& shape)
        {
            Array array = Array<double>::zeros(input.shape());
            array.reshape(shape);
            const ArrayShape& new_shape = array.shape();
            Operator op(
                [=](InParams params)
                {
                    Array result = params[0].get();
                    result.force_reshape(new_shape);
                    return result;
                },
                [](const BackwardParams params)
                {
                    Array result = params.gradient;
                    result.force_reshape(params.childs[0].get().shape());
                    return OutParams{ result };
                }, new_shape);
            return Operand::join(std::move(op), { std::move(input) });
        }

        Operand sum(Operand operand)
        {
            const ArrayShape& shape = operand.shape();
            Operator op([](InParams params) { return params[0].get().accumulate(0); },
                [=](const BackwardParams params)
                { return OutParams{ Array<double>::repeats(params.gradient[0], shape) }; }, { 1 });
            return Operand::join(std::move(op), { std::move(operand) });
        }

        Operand power(Operand base, const double exponent)
        {
            Operator op([=](InParams params)
                { return params[0].get().apply([=](const double v) { return std::pow(v, exponent); }); },
                [=](const BackwardParams params)
                {
                    InParam gradient = params.gradient;
                    InParam operand = params.childs[0];
                    return OutParams
                    {
                        exponent * gradient * operand.apply(
                            [=](const double v) { return std::pow(v, exponent - 1); })
                    };
                }, base.shape());
            return Operand::join(std::move(op), { std::move(base) });
        }

        Operand exp(Operand exponent, const double base)
        {
            Operator op([=](InParams params)
                { return params[0].get().apply([=](const double v) { return std::pow(base, v); }); },
                [=](const BackwardParams params)
                {
                    InParam gradient = params.gradient;
                    InParam operand = params.childs[0];
                    return OutParams
                    {
                        std::log(base) * gradient * operand.apply(
                            [=](const double v) { return std::pow(base, v); })
                    };
                }, exponent.shape());
            return Operand::join(std::move(op), { std::move(exponent) });
        }
    }
}
