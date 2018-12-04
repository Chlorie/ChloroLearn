#include <cmath>

#include "loss.h"

namespace chloro::operators
{
    Operand categorical_cross_entropy(Operand predicted, Operand target)
    {
        const double epsilon = 1e-8;
        if (target.shape() != scalar_shape) throw IllegalOperationException("Target should be a scalar");
        Operator op(
            [=](InParams params)
            {
                const size_t category = size_t(params[1].get()[0]);
                return Array{ -std::log(params[0].get()[category] + epsilon) };
            },
            [=](const BackwardParams params)
            {
                const double grad = params.gradient[0];
                const size_t category = size_t(params.childs[1].get()[0]);
                const Array<double>& param = params.childs[0];
                Array result = Array<double>::zeros(param.shape());
                result[category] = -grad / param[category];
                return OutParams{ result, {0} };
            }, { 1 });
        return Operand::join(std::move(op), { std::move(predicted), std::move(target) });
    }
}
