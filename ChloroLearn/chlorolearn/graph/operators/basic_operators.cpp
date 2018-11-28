#include <functional>

#include "basic_operators.h"
#include "../nodes/operator.h"
#include "../../basic/array.h"

namespace chloro::operators
{
    namespace
    {
        OutParam matmul_lr_vec_eval(InParams params) { return params[1].get()[0] * params[0]; }
        OutParams matmul_lr_vec_grad(InParam gradient, InParams params)
        {
            return
            {
                params[1].get()[0] * gradient,
                { (params[0] * gradient).accumulate(0) }
            };
        }
        Evaluation matmul_l_vec_eval(const size_t left_row, const size_t right_col)
        {
            return [=](InParams params)
            {
                InParam first = params[0];
                InParam second = params[1];
                Array<double> result = Array<double>::zeros({ left_row, right_col });
                for (size_t i = 0; i < left_row; i++)
                    for (size_t j = 0; j < right_col; j++)
                        result[i * right_col + j] = first[i] * second[j];
                return result;
            };
        }
        Partials matmul_l_vec_grad(const size_t left_row, const size_t right_col)
        {
            return [=](InParam gradient, InParams params)
            {
                InParam first = params[0];
                InParam second = params[1];
                Array<double> grad_left = Array<double>::zeros({ left_row });
                for (size_t i = 0; i < left_row; i++)
                    for (size_t j = 0; j < right_col; j++)
                        grad_left[i] += gradient[i * right_col + j] * second[j];
                Array<double> grad_right = Array<double>::zeros({ 1, right_col });
                for (size_t i = 0; i < left_row; i++)
                    for (size_t j = 0; j < right_col; j++)
                        grad_right[j] += first[i] * gradient[i * right_col + j];
                return OutParams{ grad_left,grad_right };
            };
        }
        Evaluation matmul_r_vec_eval(const size_t left_row, const size_t right_col)
        {
            
        }
    }

    Operand identity(Operand operand)
    {
        Operator op([](InParams params) { return params[0]; },
            [](InParam gradient, InParams) { return OutParams{ gradient }; }, operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand add(Operand left, Operand right)
    {
        Operator op([](InParams params) { return params[0] + params[1]; },
            [](InParam gradient, InParams) { return OutParams{ gradient,gradient }; }, left.shape());
        return Operand::join(std::move(op), { std::move(left), std::move(right) });
    }
    Operand operator+(Operand left, Operand right) { return add(std::move(left), std::move(right)); }

    Operand subtract(Operand left, Operand right)
    {
        Operator op([](InParams params) { return params[0] - params[1]; },
            [](InParam gradient, InParams) { return OutParams{ gradient,-gradient }; }, left.shape());
        return Operand::join(std::move(op), { std::move(left), std::move(right) });
    }
    Operand operator-(Operand left, Operand right) { return subtract(std::move(left), std::move(right)); }

    Operand multiply(Operand left, Operand right)
    {
        Operator op([](InParams params) { return params[0] * params[1]; },
            [](InParam gradient, InParams params)
            {
                return OutParams
                {
                    params[1] * gradient,
                    params[0] * gradient
                };
            }, left.shape());
        return Operand::join(std::move(op), { std::move(left), std::move(right) });
    }
    Operand operator*(Operand left, Operand right) { return multiply(std::move(left), std::move(right)); }

    Operand divide(Operand left, Operand right)
    {
        Operator op([](InParams params) { return params[0] / params[1]; },
            [](InParam gradient, InParams params)
            {
                return OutParams
                {
                    gradient / params[1],
                    -params[0].get() / params[1] / params[1] * gradient
                };
            }, left.shape());
        return Operand::join(std::move(op), { std::move(left), std::move(right) });
    }
    Operand operator/(Operand left, Operand right) { return divide(std::move(left), std::move(right)); }

    Operand matrix_multiply(Operand left, Operand right)
    {
        if (left.shape().size() > 2 || right.shape().size() > 2)
            throw MismatchedSizesException("The operands are not matrices");
        const bool left_vec = left.shape().size() == 1;
        const size_t left_row = left_vec ? 1 : left.shape()[0];
        const size_t left_col = left_vec ? left.shape()[0] : left.shape()[1];
        const bool right_vec = right.shape().size() == 1;
        const size_t right_row = right_vec ? 1 : right.shape()[0];
        const size_t right_col = right_vec ? right.shape()[0] : right.shape()[1];
        if (left_col != right_row) throw MismatchedSizesException("Cannot multiply these operands");
        if (left_vec)
        {
            if (right_vec)
                return Operand::join({ matmul_lr_vec_eval, matmul_lr_vec_grad, { left_row } },
                    { std::move(left),std::move(right) });
            return Operand::join(
                {
                    matmul_l_vec_eval(left_row, right_col),
                    matmul_l_vec_grad(left_row, right_col),
                    { left_row, right_col }
                }, { std::move(left),std::move(right) });
        }
        if (right_vec)
        {

        }
        else
        {

        }
    }

    Operand dot(Operand left, Operand right) { return sum(std::move(left) * std::move(right)); }

    Operand repeat(Operand scalar, const ArrayShape& shape)
    {
        const ArrayShape& scalar_shape = scalar.shape();
        if (scalar_shape.size() != 1 || scalar_shape[0] != 1) throw MismatchedSizesException("Repeated value isn't a scalar");
        Operator op([&](InParams params) { return Array<double>::repeats(params[0].get()[0], shape); },
            [](InParam gradient, InParams) { return OutParams{ gradient.accumulate(0) }; }, shape);
        return Operand::join(std::move(op), { std::move(scalar) });
    }

    Operand sum(Operand operand)
    {
        const ArrayShape& shape = operand.shape();
        Operator op([](InParams params) { return params[0].get().accumulate(0); },
            [=](InParam gradient, InParams)
            { return OutParams{ Array<double>::repeats(gradient[0], operand.shape()) }; }, shape);
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand power(Operand base, const double exponent)
    {
        Operator op([=](InParams params)
            {return params[0].get().apply([=](const double v) { return std::pow(v, exponent); }); },
            [=](InParam gradient, InParams params)
            {
                return OutParams
                {
                    exponent * gradient * params[0].get().apply(
                        [=](const double v) { return std::pow(v, exponent - 1); })
                };
            }, base.shape());
        return Operand::join(std::move(op), { std::move(base) });
    }
}
