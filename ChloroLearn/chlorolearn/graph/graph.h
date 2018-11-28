#pragma once

#include <list>
#include <initializer_list>

#include "node.h"
#include "input_param.h"
#include "input_pack.h"
#include "operand.h"

namespace chloro
{
    class Graph final
    {
    private:
        std::list<Node> nodes_;
        void input(Node& node, const Array<double>& value) const;
        static void update_dag(Node& node);
    public:
        Graph() = default;
        Node& add_input() { return add_input({ 1 }); }
        Node& add_input(const ArrayShape& size);
        Node& add_variable() { return add_variable({ 1 }); }
        Node& add_variable(const ArrayShape& size);
        Node& add_constant(const double value) { return add_constant(Array<double>{ value }); }
        Node& add_constant(const Array<double>& array);
        Node& add_constant(Array<double>&& array);
        Node& add_operator(Operand&& list);
        const Array<double>& get_value(Node& node, std::initializer_list<InputParam> input_params = {});
        void set_variable(Node& node, const Array<double>& value) const;
        void optimize_once(Node& target, std::initializer_list<InputParam> input_params = {}, double learning_rate = 0.1);
        void optimize(size_t batch_size, Node& target, std::initializer_list<InputPack> input_pack = {},
            double learning_rate = 0.1, bool not_update_dag = false);
    };
}
