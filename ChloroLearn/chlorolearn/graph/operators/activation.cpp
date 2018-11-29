#include <algorithm>

#include "activation.h"

namespace chloro::operators
{
    Operand relu(Operand operand)
    {
        Operator op([](InParams params) { return params[0].get().apply([](const double v) { return v > 0 ? v : 0; }); },
            [](InParam gradient, InParams params)
            { return OutParams{ gradient * params[0].get().apply([](const double v) {return v > 0 ? 1 : 0; }) }; },
            operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }

    Operand softmax(Operand operand)
    {
        Operator op(
            [](InParams params)
            {
                InParam param = params[0];
                const double max = param.accumulate(param[0], [](const double x, const double y) {return x > y ? x : y; });
                Array<double> aug_exp = (param - max).apply(std::exp<double, double>);
                const double sum = aug_exp.accumulate(0);
                return aug_exp /= sum;
            },
            [](InParam gradient, InParams params)
            {
                InParam param = params[0];
                const double max = param.accumulate(param[0], [](const double x, const double y) {return x > y ? x : y; });
                Array<double> aug_exp = (param - max).apply(std::exp<double, double>);
                const double sum = aug_exp.accumulate(0);
                Array<double> exp = param.apply(std::exp<double, double>);
                const Array<double> term = 1 - std::move(aug_exp) / sum;
                return OutParams{ gradient * std::move(exp) * term };
            }, operand.shape());
        return Operand::join(std::move(op), { std::move(operand) });
    }
}
