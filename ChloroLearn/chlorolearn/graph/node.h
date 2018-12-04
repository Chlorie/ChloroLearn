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

    /**
     * \brief A class representing a node in the DAG flow graph. Users should \b not construct objects
     * of this type manually, \b neither should users manually construct any of the four node content types
     * to be mentioned below. Use the \a add_... methods of class \c Graph instead. Nodes can contain
     * contents of types including \c Input, \c Constant, \c Variable and \c Operator. For more information on
     * these four node types, please see their own documentation and implementation respectively.
     */
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
        void clear_gradient();
        void apply_gradient(double learning_rate);
        const Array<double>& get_value();
        const Array<double>& forward_propagate();
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
