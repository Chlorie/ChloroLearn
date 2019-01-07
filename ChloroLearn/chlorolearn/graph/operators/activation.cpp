#include <algorithm>

#include "activation.h"

namespace chloro::operators
{
    Operand relu(Operand operand)
    {
        Operator op([](InParams params) { return params[0].get().apply([](const double v) { return v > 0 ? v : 0; }); },
            [](const BackwardParams params)
            {
                InParam param = params.childs[0];
                InParam gradient = params.gradient;
                return OutParams{ gradient * param.apply([](const double v) { return v > 0 ? 1 : 0; }) };
            },
            operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand leaky_relu(Operand operand)
    {
        Operator op([](InParams params) { return params[0].get().apply(
            [](const double v) { return v > 0 ? v : 0.01 * v; }); },
            [](const BackwardParams params)
            {
                InParam param = params.childs[0];
                InParam gradient = params.gradient;
                return OutParams{ gradient * param.apply([](const double v) { return v > 0 ? 1 : 0.01; }) };
            },
            operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand sigmoid(Operand operand)
    {
        Operator op(
            [](InParams params)
            {
                return params[0].get().apply([](const double v)
                    { return 1 / (1 + std::exp(-v)); });
            },
            [](const BackwardParams params)
            {
                return OutParams
                {
                    params.gradient * params.value.apply([](const double v)
                        { return v * (1 - v); })
                };
            }, operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand softmax(Operand operand)
    {
        Operator op(
            [](InParams params)
            {
                InParam param = params[0];
                const double max = param.accumulate(param[0], [](const double x, const double y) { return x > y ? x : y; });
                Array aug_exp = (param - max).apply(std::exp<double, double>);
                const double sum = aug_exp.accumulate(0);
                return aug_exp /= sum;
            },
            [](const BackwardParams params)
            {
                InParam gradient = params.gradient;
                const Array<double>& value = params.value;
                Array result = Array<double>::zeros(value.shape());
                const size_t size = value.size();
                for (size_t i = 0; i < size; i++)
                    for (size_t j = 0; j < size; j++)
                    {
                        const double jacobian = value[i] * ((i == j) - value[j]);
                        result[i] += gradient[j] * jacobian;
                    }
                return OutParams{ result };
            }, operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }
}
