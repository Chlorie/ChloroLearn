#include <cmath>
#include <numeric>

#include "loss.h"

namespace chloro::operators
{
    Operand categorical_cross_entropy(Operand predicted, Operand target)
    {
        const double epsilon = 1e-7;
        if (target.shape() != scalar_shape) throw IllegalOperationException("Target should be a scalar");
        const ArrayShape& shape = predicted.shape();
        const size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies());
        Operator op(
            [=](InParams params)
            {
                const size_t category = size_t(std::round(params[1].get()[0]));
                Array<double> values = params[0];
                values[category] = 1 - values[category];
                values.apply_in_place([=](const double v) { return std::log(v + epsilon); });
                const double loss = -values.accumulate(0) / size;
                return Array{ loss };
            },
            [=](InParam gradient, InParams params)
            {
                const double grad = gradient[0];
                const size_t category = size_t(std::round(params[1].get()[0]));
                Array<double> result = params[0];
                result[category] += 1;
                result -= 1 - epsilon;
                result.apply_in_place([=](const double v) { return -grad / v / size; });
                return OutParams{ result, {0} };
            }, { 1 });
        return Operand::join(std::move(op), { std::move(predicted), std::move(target) });
    }
}
