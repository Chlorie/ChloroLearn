#include <random>

#include "graph.h"
#include "nodes/input.h"
#include "nodes/variable.h"

namespace chloro
{
    void Graph::input(Node& node, const Array<double>& value) const
    {
        if (node.content_.index() != 0) throw IllegalOperationException("Current node isn't an input node");
        std::get<0>(node.content_).input(value);
    }

    void Graph::update_dag(Node& node)
    {
        node.update_time_++;
        if (node.content_.index() == 3) // Operator
            for (NodeRef from : node.from_nodes_)
                update_dag(from.get());
    }

    Node& Graph::add_input(const ArrayShape& size)
    {
        nodes_.emplace_back(Input(size));
        return nodes_.back();
    }

    Node& Graph::add_variable(const ArrayShape& size)
    {
        nodes_.emplace_back(Variable(size));
        Node& node = nodes_.back();
        node.gradient_ = Array<double>::zeros(size);
        return node;
    }

    Node& Graph::add_constant(const Array<double>& array)
    {
        nodes_.emplace_back(Constant(array));
        return nodes_.back();
    }

    Node& Graph::add_constant(Array<double>&& array)
    {
        nodes_.emplace_back(Constant(array));
        return nodes_.back();
    }

    Node& Graph::add_operator(Operand&& list)
    {
        std::vector<NodeRef> list_ref;
        list.for_each([&](ListedOperator& item)
            {
                std::vector<NodeRef> from;
                for (ListedOperator::Ref& ref : item.from_nodes)
                    if (ref.index() == 0) // size_t
                        from.push_back(list_ref[std::get<0>(ref)]);
                    else
                        from.push_back(std::get<1>(ref));
                nodes_.emplace_back(Node(std::move(item.content), from));
                Node& result = nodes_.back();
                result.gradient_ = Array<double>::zeros(result.shape());
                list_ref.emplace_back(result);
            });
        return nodes_.back();
    }

    const Array<double>& Graph::get_value(Node& node, const std::initializer_list<InputParam> input_params)
    {
        for (const InputParam& input_param : input_params) input(input_param.input_, input_param.value_);
        for (Node& graph_node : nodes_) graph_node.value_ready_ = false;
        return node.get_value();
    }

    void Graph::set_variable(Node& node, const Array<double>& value) const
    {
        if (node.content_.index() != 2) // Not a variable
            throw IllegalArgumentException("Current node is not a variable");
        std::get<2>(node.content_).set_value(value);
    }

    void Graph::optimize_once(Node& target, const std::initializer_list<InputParam> input_params,
        const double learning_rate)
    {
        if (target.content_.index() != 3) throw IllegalOperationException("Target should be an operator");
        for (Node& node : nodes_)
        {
            node.update_time_ = 0;
            node.updated_time_ = 0;
            node.clear_gradient();
        }
        update_dag(target);
        get_value(target, input_params);
        target.back_propagate(Array<double>::repeats(1.0, target.shape()));
        for (Node& node : nodes_) node.apply_gradient(learning_rate);
    }

    void Graph::optimize(const size_t batch_size, Node& target, const std::initializer_list<InputPack> input_pack,
        const double learning_rate, const bool not_update_dag)
    {
        if (target.content_.index() != 3) throw IllegalOperationException("Target should be an operator");
        for (Node& node : nodes_) node.updated_time_ = 0;
        size_t pack_size = 0;
        bool first = true;
        for (const InputPack& item : input_pack)
        {
            if (first) pack_size = item.pack_.size();
            if (pack_size != item.pack_.size()) throw MismatchedSizesException("Input packs should be of the same size");
            first = false;
        }
        static std::mt19937 generator{ std::random_device()() };
        const std::uniform_int_distribution<> dist(0, pack_size == 0 ? 0 : pack_size - 1);
        if (!not_update_dag)
        {
            for (Node& node : nodes_) node.update_time_ = 0;
            update_dag(target);
        }
        const Array<double> ones = Array<double>::repeats(1.0, target.shape());
        for (size_t i = 0; i < batch_size; i++)
        {
            for (Node& node : nodes_)
            {
                node.clear_gradient();
                node.value_ready_ = false;
            }
            if (pack_size != 0)
            {
                const int index = dist(generator);
                for (const InputPack& item : input_pack) input(item.input_, item.pack_[index]);
            }
            target.get_value();
            target.back_propagate(ones);
            for (Node& node : nodes_) node.apply_gradient(learning_rate);
        }
    }
}
