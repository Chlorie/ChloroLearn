#include "node.h"

namespace chloro
{
    void Node::set_optimizer(const Optimizer& optimizer) { optimizer_ = optimizer; }

    void Node::clear_gradient()
    {
        switch (content_.index())
        {
        case 0: case 1: return; // Input, Constant
        case 2: case 3: gradient_.clear(); return; // Variable, Operator
        default: throw ArgumentOutOfRangeException("Current node is in invalid state");
        }
    }

    void Node::apply_gradient()
    {
        if (content_.index() != 2) return; // Not Variable
        std::get<VariableType>(content_).subtract_from_current(optimizer_(gradient_));
    }

    const Array<double>& Node::get_value()
    {
        switch (content_.index())
        {
        case 0: return std::get<0>(content_).value(); // Input
        case 1: return std::get<1>(content_).value(); // Constant
        case 2: return std::get<2>(content_).value(); // Variable
        case 3: // Operator
            if (!value_ready_)
            {
                std::vector<ArrayRef> params;
                for (NodeRef from : from_nodes_) params.emplace_back(from.get().get_value());
                operator_value_ = std::get<3>(content_).evaluate(params);
                value_ready_ = true;
            }
            return operator_value_;
        default: throw ArgumentOutOfRangeException("Current node is in invalid state");
        }
    }

    const Array<double>& Node::forward_propagate()
    {
        switch (content_.index())
        {
        case 0: return std::get<InputType>(content_).value(); // Input
        case 1: return std::get<ConstantType>(content_).value(); // Constant
        case 2: return std::get<VariableType>(content_).value(); // Variable
        case 3: // Operator
            if (!value_ready_)
            {
                std::vector<ArrayRef> params;
                for (NodeRef from : from_nodes_) params.emplace_back(from.get().forward_propagate());
                operator_value_ = std::get<OperatorType>(content_).forward_propagate(params);
                value_ready_ = true;
            }
            return operator_value_;
        default: throw ArgumentOutOfRangeException("Current node is in invalid state");
        }
    }

    void Node::back_propagate(const Array<double>& gradient)
    {
        switch (content_.index())
        {
        case 0: case 1: return; // Back propagation ends at constant values
        case 2: // Variable
            gradient_ += gradient.apply([](const double v) { return std::clamp(v, -5.0, 5.0); });
            updated_time_++;
            return;
        case 3: // Operator
            gradient_ += gradient.apply([](const double v) { return std::clamp(v, -5.0, 5.0); });
            updated_time_++;
            if (updated_time_ % update_time_ == 0)
            {
                Operator& content = std::get<OperatorType>(content_);
                std::vector<ArrayRef> params;
                for (NodeRef from : from_nodes_) params.emplace_back(from.get().get_value());
                std::vector<Array<double>> gradients = content.back_propogate(gradient_, params, forward_propagate());
                const size_t node_count = from_nodes_.size();
                for (size_t i = 0; i < node_count; i++) from_nodes_[i].get().back_propagate(gradients[i]);
            }
            return;
        default: throw ArgumentOutOfRangeException("Current node is in invalid state");
        }
    }

    const ArrayShape& Node::shape()
    {
        switch (content_.index())
        {
        case 0: return std::get<InputType>(content_).shape(); // Input
        case 1: return std::get<ConstantType>(content_).value().shape(); // Constant
        case 2: return std::get<VariableType>(content_).value().shape(); // Variable
        case 3: return std::get<OperatorType>(content_).shape(); // Operator
        default: throw ArgumentOutOfRangeException("Current node is in invalid state");
        }
    }
}
