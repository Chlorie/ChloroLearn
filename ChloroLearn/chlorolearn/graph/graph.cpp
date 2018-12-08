#include <random>
#include <algorithm>

#include "graph.h"
#include "nodes/input.h"
#include "nodes/variable.h"
#include "../utility/binary_io.h"
#include "../utility/stopwatch.h"

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

    void Graph::forward_propagate(Node& node, std::initializer_list<InputParam> input_params)
    {
        for (const InputParam& input_param : input_params) input(input_param.input, input_param.value);
        for (Node& graph_node : nodes_) graph_node.value_ready_ = false;
        node.forward_propagate();
    }

    Node& Graph::add_input(const ArrayShape& shape)
    {
        nodes_.emplace_back(Input(shape));
        return nodes_.back();
    }

    Node& Graph::add_variable(const ArrayShape& shape)
    {
        nodes_.emplace_back(Variable(shape));
        Node& node = nodes_.back();
        node.gradient_ = Array<double>::zeros(shape);
        return node;
    }

    Node& Graph::add_constant(const Array<double>& array)
    {
        nodes_.emplace_back(Constant(array));
        return nodes_.back();
    }

    Node& Graph::add_constant(Array<double>&& array)
    {
        nodes_.emplace_back(Constant(std::move(array)));
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
        for (const InputParam& input_param : input_params) input(input_param.input, input_param.value);
        for (Node& graph_node : nodes_) graph_node.value_ready_ = false;
        return node.get_value();
    }

    void Graph::set_variable(Node& node, const Array<double>& value) const
    {
        if (node.content_.index() != 2) // Not a variable
            throw IllegalArgumentException("Current node is not a variable");
        std::get<2>(node.content_).set_value(value);
    }

    void Graph::randomize_variables(const double mean, const double stddev)
    {
        for (Node& node : nodes_)
            if (node.content_.index() == 2) // Variable node
            {
                Variable& variable = std::get<2>(node.content_);
                const ArrayShape& shape = variable.value().shape();
                variable.set_value(Array<double>::random(shape, mean, stddev));
            }
    }

    void Graph::optimize_once(Node& target, const std::initializer_list<InputParam> input_params,
        const Optimizer& optimizer)
    {
        if (target.content_.index() != 3) throw IllegalOperationException("Target should be an operator");
        for (Node& node : nodes_)
        {
            node.update_time_ = 0;
            node.updated_time_ = 0;
            node.clear_gradient();
            node.set_optimizer(optimizer);
        }
        update_dag(target);
        forward_propagate(target, input_params);
        target.back_propagate(Array<double>::repeats(1.0, target.shape()));
        for (Node& node : nodes_) node.apply_gradient();
    }

    void Graph::optimize(Node& target, const std::initializer_list<InputPack> input_pack,
        const Optimizer& optimizer, const size_t batch_size, Callback&& batch_callback, Callback&& epoch_callback)
    {
        if (target.content_.index() != 3) throw IllegalOperationException("Target should be an operator");
        for (Node& node : nodes_) node.updated_time_ = 0;
        size_t epoch_size = 0;
        bool first = true;
        for (const InputPack& item : input_pack)
        {
            if (first) epoch_size = item.pack.size();
            if (epoch_size != item.pack.size()) throw MismatchedSizesException("Input packs should be of the same size");
            first = false;
        }
        if (epoch_size == 0)
            throw IllegalOperationException("In order to perform batch updates and count epochs, there must be at least "
                "one input parameter.");
        static std::mt19937 generator{ std::random_device{}() };
        for (Node& node : nodes_)
        {
            node.update_time_ = 0;
            node.set_optimizer(optimizer);
        }
        update_dag(target);
        const Array<double> ones = Array<double>::repeats(1.0, target.shape());
        size_t counter = 0;
        Stopwatch batch_watch;
        while (true)
        {
            Stopwatch epoch_watch;
            std::vector<size_t> permutation(epoch_size);
            for (size_t i = 0; i < epoch_size; i++) permutation[i] = i;
            std::shuffle(permutation.begin(), permutation.end(), generator);
            for (size_t i = 0; i < epoch_size; i++)
            {
                for (Node& node : nodes_)
                {
                    node.clear_gradient();
                    node.value_ready_ = false;
                }
                for (const InputPack& item : input_pack) input(item.input, item.pack[permutation[i]]);
                target.forward_propagate();
                target.back_propagate(ones);
                for (Node& node : nodes_) node.apply_gradient();
                counter++;
                if (counter % batch_size == 0 && batch_callback)
                {
                    batch_watch.stop();
                    batch_callback(batch_watch.seconds());
                    batch_watch.restart();
                }
            }
            if (epoch_callback)
            {
                epoch_watch.stop();
                epoch_callback(epoch_watch.seconds());
            }
        }
    }

    void Graph::save_variables(const std::string& path) const
    {
        std::ofstream stream(path, std::ios::out | std::ios::binary);
        for (const Node& node : nodes_)
            if (node.content_.index() == 2)
            {
                const Array<double>& value = std::get<2>(node.content_).value();
                write_vector(stream, value.shape());
                write_vector(stream, value.data());
            }
        stream.close();
    }

    void Graph::load_variables(const std::string& path)
    {
        std::ifstream stream(path, std::ios::in | std::ios::binary);
        for (Node& node : nodes_)
            if (node.content_.index() == 2)
            {
                Variable& variable = std::get<2>(node.content_);
                ArrayShape shape;
                read_vector(stream, shape);
                if (!stream.good())
                    throw IllegalOperationException("Data in the file doesn't match the variable amount in the graph");
                Array<double> array = Array<double>::zeros(shape);
                std::vector<double> values;
                read_vector(stream, values);
                array = std::move(values);
                variable.set_value(std::move(array));
            }
        stream.close();
    }
}
