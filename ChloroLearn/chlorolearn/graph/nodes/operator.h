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

    /**
     * \brief This kind of node content holds a lazy-evaluated operation.
     * \details This class is the heart of this library. The other three types of nodes only contains
     * user input or a specific value, while this type of node does not contain an array value per se,
     * but like other nodes, operators have fixed shapes as well. Other nodes can be connected to an
     * operator node, for example, connect two variable nodes to a plus operator.
     * \details An operator has three operating mode: Evaluation, forward propagation and back propagation. 
     * This three modes corresponds to three private fields of the object. Evaluation takes some array
     * values and returns a single array value, like plus operator taking two arrays and returning the sum.
     * Forward propagation evaluates the value, but can also set the internal state of the operator for
     * further use. Note that, the returned value of evaluation and forward propagation may not be the same,
     * for example, the DropOut operator. Back propagation takes the forward propagated value of this node, 
     * and the values of its child nodes, together with the internal state, propagates the received gradient
     * back to the child nodes of this operator.
     */
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
        /**
         * \brief Construct an operator with same process for evaluation and forward propagation.
         * \param evaluation The evaluation function.
         * \param backward The back propagation function.
         * \param shape The shape of the evaluation result.
         */
        Operator(Evaluation&& evaluation, Backward&& backward, const ArrayShape& shape) :
            evaluation_(evaluation),
            forward_([eval = std::move(evaluation)](const ForwardParams params){ return eval(params.childs); }),
            backward_(std::move(backward)),
            shape_(shape) {}
        /**
         * \brief Construct an operator with different processes for evaluation and forward propagation.
         * \param evaluation The evaluation function.
         * \param forward The forward propagation function.
         * \param backward The back propagation function.
         * \param shape The shape of the evaluation result.
         * \param state_shape The shape of the state array, defaults to the same as that of the result.
         */
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
        /** \brief Get the shape of the evaluation result. */
        const ArrayShape& shape() const { return shape_; }
        /**
         * \brief Evaluates this node given the values of its childs.
         * \param params Evaluated values of child nodes.
         * \return The evaluated value of this node.
         */
        OutParam evaluate(InParams params) const { return evaluation_(params); }
        /**
         * \brief Forward propagates the value through this node.
         * \param childs Forward propagated values of child nodes.
         * \return The forward propagated value of this node.
         */
        OutParam forward_propagate(InParams childs) { return forward_({ childs, state_ }); }
        /**
         * \brief Back propagate the gradient to the childs.
         * \param gradient Propagated gradient of the target node w.r.t. this node.
         * \param childs Forward propagated values of child nodes.
         * \param value The forward propagated value of this node.
         * \return Propagated gradient of the target node w.r.t. child nodes of this node.
         */
        OutParams back_propogate(InParam gradient, InParams childs, InParam value)
        {
            return backward_({ gradient, childs, value, state_ });
        }
    };
}
