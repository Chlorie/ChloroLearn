#pragma once

#include <vector>
#include <variant>
#include <functional>

#include "optimizer.h"
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
     * \brief A class representing a node in the DAG flow graph.
     * \details Users should \b not construct objects of this type manually, \b neither should users
     * manually construct any of the four node content types to be mentioned below. Use the \a add_...
     * methods of class \c Graph instead. Nodes can contain contents of types including \c Input, \c
     * Constant, \c Variable and \c Operator. For more information on these four node types, please
     * refer to their own documentation and implementation respectively.
     */
    class Node final
    {
        friend class Graph;
    private:
        enum VariantType
        {
            InputType,
            ConstantType,
            VariableType,
            OperatorType
        };
        Array<double> operator_value_;
        Array<double> gradient_;
        Optimizer optimizer_;
        bool value_ready_ = false;
        int update_time_ = 0;
        int updated_time_ = 0;
        std::vector<NodeRef> from_nodes_;
        std::variant<Input, Constant, Variable, Operator> content_;
        void set_optimizer(const Optimizer& optimizer);
        void clear_gradient();
        void apply_gradient();
        const Array<double>& get_value();
        const Array<double>& forward_propagate();
        void back_propagate(const Array<double>& gradient);
    public:
        Node() = delete;
        Node(Node&&) = default; /**< \brief Move constructor. */
        explicit Node(Input& content) = delete;
        explicit Node(Constant& content) = delete;
        explicit Node(Variable& content) = delete;
        explicit Node(Operator& content, const std::vector<NodeRef>&) = delete;
        /** \brief Move construct a node of content type \c Input. */
        explicit Node(Input&& content) :content_(content) {}
        /** \brief Move construct a node of content type \c Constant. */
        explicit Node(Constant&& content) :content_(content) {}
        /** \brief Move construct a node of content type \c Variable. */
        explicit Node(Variable&& content) :content_(content) {}
        /** \brief Move construct a node of content type \c Operator, and link up the nodes. */
        explicit Node(Operator&& content, const std::vector<NodeRef>& from_nodes) :from_nodes_(from_nodes), content_(content) {}
        /** \brief Get the shape of the node content. */
        const ArrayShape& shape();
    };
}
