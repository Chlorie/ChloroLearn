#pragma once

#include <vector>
#include <variant>
#include <functional>

#include "nodes/input.h"
#include "nodes/constant.h"
#include "nodes/variable.h"
#include "nodes/operator.h"

namespace chloro
{
    class Node;

    using NodeRef = std::reference_wrapper<Node>;
    using ArrayRef = std::reference_wrapper<const Array<double>>;

    class Node final
    {
        friend class Graph;
    private:
        Array<double> operator_value_;
        Array<double> gradient_;
        bool value_ready_ = false;
        int update_time_ = 0;
        int updated_time_ = 0;
        std::vector<NodeRef> from_nodes_;
        std::vector<NodeRef> to_nodes_;
        std::variant<Input, Constant, Variable, Operator> content_;
        void get_operator_value();
        void clear_gradient();
        void apply_gradient(double learning_rate);
        const Array<double>& get_value();
        void back_propagate(const Array<double>& gradient);
    public:
        Node() = delete;
        Node(Node&&) = default;
        explicit Node(Input& content) = delete;
        explicit Node(Constant& content) = delete;
        explicit Node(Variable& content) = delete;
        explicit Node(Operator& content, const std::vector<NodeRef>&) = delete;
        explicit Node(Input&& content) :content_(content) {}
        explicit Node(Constant&& content) :content_(content) {}
        explicit Node(Variable&& content) :content_(content) {}
        explicit Node(Operator&& content, const std::vector<NodeRef>& from_nodes);
        const ArrayShape& shape();
    };
}
