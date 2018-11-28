#pragma once

#include <vector>
#include <functional>

#include "../../basic/array.h"

namespace chloro
{
    using InParam = const Array<double>&;
    using InParams = const std::vector<std::reference_wrapper<const Array<double>>>&;
    using OutParam = Array<double>;
    using OutParams = std::vector<OutParam>;
    using StateParam = Array<double>&;
    using Evaluation = std::function<OutParam(InParams)>;
    using Forward = std::function<OutParam(StateParam, InParams)>;
    using Partials = std::function<OutParams(InParam, InParams)>;

    class Operator final
    {
    private:
        Evaluation evaluation_;
        Forward forward_;
        Partials partials_;
        ArrayShape shape_;
    public:
        template <typename T, typename U, typename V>
        Operator(T&& evaluation, U&& forward, V&& partials, const ArrayShape& shape)
            :evaluation_(evaluation), forward_(forward), partials_(partials), shape_(shape) {}
        template <typename T, typename U>
        Operator(T&& evaluation, U&& partials, const ArrayShape& shape)
            : evaluation_(evaluation), forward_([&](StateParam, InParams params) { return evaluation_(params); }),
            partials_(partials), shape_(shape) {}
        const ArrayShape& shape() const { return shape_; }
        OutParam value(InParams params) const { return evaluation_(params); }
        OutParams back_propogate(InParam gradient, InParams params) const { return partials_(gradient, params); }
        const Evaluation& evaluation() const { return evaluation_; }
        const Forward& forward() const { return forward_; }
        const Partials& partials() const { return partials_; }
    };
}
