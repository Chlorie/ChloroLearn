#pragma once

#include <vector>
#include <functional>

#include "../../basic/array.h"
#include "../../basic/propagate_struct.h"

namespace chloro
{
    using Evaluation = std::function<OutParam(InParams)>;
    using Forward = std::function<OutParam(ForwardParams)>;
    using Backward = std::function<OutParams(BackwardParams)>;

    class Operator final
    {
    private:
        Array<double> state_;
        Evaluation evaluation_;
        Forward forward_;
        Backward backward_;
        ArrayShape shape_;
    public:
        Operator() = delete;
        Operator(Evaluation&& evaluation, Backward&& backward, const ArrayShape& shape) :
            evaluation_(evaluation),
            forward_([eval = std::move(evaluation)](const ForwardParams params){ return eval(params.childs); }),
            backward_(std::move(backward)),
            shape_(shape) {}
        Operator(Evaluation&& evaluation, Forward&& forward, Backward&& backward,
            const ArrayShape& shape, const ArrayShape& state_shape = {}) :
            evaluation_(std::move(evaluation)),
            forward_(std::move(forward)),
            backward_(std::move(backward)),
            shape_(shape)
        {
            if (state_shape.empty())
                state_ = Array<double>::zeros(shape);
            else
                state_ = Array<double>::zeros(state_shape);
        }
        const ArrayShape& shape() const { return shape_; }
        OutParam evaluate(InParams params) const { return evaluation_(params); }
        OutParam forward_propagate(InParams childs) { return forward_({ childs, state_ }); }
        OutParams back_propogate(InParam gradient, InParams childs, InParam value)
        {
            return backward_({ gradient, childs, value, state_ });
        }
    };
}
