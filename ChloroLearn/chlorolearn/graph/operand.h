#pragma once

#include <vector>
#include <variant>
#include <functional>
#include <optional>

#include "node.h"

namespace chloro
{
    struct ListedOperator final
    {
        using Ref = std::variant<size_t, NodeRef>;
        Operator content;
        std::vector<Ref> from_nodes;
        ListedOperator(Operator&& content, std::vector<Ref>&& from_nodes)
            :content(std::move(content)), from_nodes(std::move(from_nodes)) {}
    };

    // ReSharper disable CppNonExplicitConvertingConstructor

    // The Operand class is sort of a syntax tree, this is for inserting multiple operators
    // together into a graph, like inserting an expression "x * (x + 2)", which composes
    // of two new operator nodes
    class Operand final
    {
        using Func = std::function<void(ListedOperator&)>;
        using Ref = ListedOperator::Ref;
    private:
        std::vector<ListedOperator> data_;
        std::optional<NodeRef> ref_;
        Ref root_node() const { return data_.empty() ? Ref(*ref_) : data_.size() - 1; }
        static void offset(std::vector<Ref>& refs, const size_t value)
        {
            for (Ref& ref : refs)
                if (ref.index() == 0) // size_t
                    ref = std::get<0>(ref) + value;
        }
        void offset_all(const size_t value) { for (ListedOperator& item : data_) offset(item.from_nodes, value); }
        Operand() = default;
    public:
        Operand(const Operand&) = default;
        Operand(Operand&&) = default;
        Operand(const NodeRef ref) :ref_(ref) {}
        const ArrayShape& shape() const { return data_.empty() ? ref_->get().shape() : data_.back().content.shape(); }
        void for_each(Func&& func) { for (ListedOperator& item : data_) func(item); }
        static Operand join(Operator&& value, std::initializer_list<Operand> childs)
        {
            std::vector<Operand> operands(std::make_move_iterator(childs.begin()), std::make_move_iterator(childs.end()));
            std::vector<Ref> new_refs;
            const size_t param_size = operands.size();
            size_t sum = 0;
            for (size_t i = 0; i < param_size; i++)
            {
                new_refs.push_back(operands[i].root_node());
                if (i > 0)
                {
                    sum += operands[i - 1].data_.size();
                    operands[i].offset_all(sum);
                }
            }
            Operand result;
            std::vector<ListedOperator>& data = result.data_;
            for (Operand& operand : operands)
                data.insert(data.end(), std::make_move_iterator(operand.data_.begin()),
                    std::make_move_iterator(operand.data_.end()));
            data.emplace_back(std::move(value), std::move(new_refs));
            return result;
        }
    };
}
