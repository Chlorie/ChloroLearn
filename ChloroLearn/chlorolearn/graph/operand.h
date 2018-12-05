#pragma once

#include <vector>
#include <variant>
#include <functional>
#include <optional>

#include "node.h"

namespace chloro
{
    /**
     * \brief This struct serves as the internal implementation for the \c Operand class.
     * \remark Users should not use this struct explicitly.
     */
    struct ListedOperator final
    {
        /** \brief A union of either a \c size_t value (operator reference) or a \c NodeRef. */
        using Ref = std::variant<size_t, NodeRef>;
        Operator content; /**< \brief The operator that is listed. */
        std::vector<Ref> from_nodes; /**< \brief Reference to the nodes connected to this operator. */
        /** \brief Move construct a listed operator. */
        ListedOperator(Operator&& content, std::vector<Ref>&& from_nodes)
            :content(std::move(content)), from_nodes(std::move(from_nodes)) {}
    };

    // ReSharper disable CppNonExplicitConvertingConstructor

    /**
     * \brief This class serves as a syntax tree for inserting multiple \c Operator objects
     * together into the same graph.
     * \details For example, the expression <em>x*(y+2)</em> composes of two <tt>Operator</tt>s, the
     * addition and the multiplication, together with references to nodes \a x and \a y. Explicitly
     * usage of this class is not recommended, except when you're trying to implement more operators
     * by yourself. In such cases, please refer to the implementations in namespace \c chloro::operators.
     * Explicit construction of this class is not recommended as well, since only rvalue references to
     * this object is used in this library.
     */
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
        Operand(const Operand&) = default; /**< \brief Copy constructor. */
        Operand(Operand&&) = default; /**< \brief Move constructor. */
        /**
         * \brief Construct an operand with only a node reference but no operators implicitly
         * using a node reference.
         */
        Operand(const NodeRef ref) :ref_(ref) {}
        /**
         * \brief Get the shape of the last operator listed in this operand. If there's not any,
         * return the shape of the last node reference.
         */
        const ArrayShape& shape() const { return data_.empty() ? ref_->get().shape() : data_.back().content.shape(); }
        /** \brief Call a function on each of the operators listed in this object. */
        void for_each(Func&& func) { for (ListedOperator& item : data_) func(item); }
        /**
         * \brief Join one or more operands together with an operator.
         * \details For example, for <em>x+y*z</em>, the plus operator joins two operands together:
         * The operand containing only a \c NodeRef \a x, together with the operand containing an 
         * multiplication operator and two <tt>NodeRef</tt>s to \a y and \a z. For more information
         * about this, see implementations in namespace \c chloro::operators.
         * \param value The operator joining the given operands.
         * \param childs The operands to join.
         * \return Joined operand.
         */
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
