#pragma once

#include "../../basic/array.h"

namespace chloro
{
    /**
     * \brief This kind of node content holds a constant array value, which can be queried
     * later in the evaluating process.
     * \remark Back-propagation won't pass to this kind of node.
     */
    class Constant final
    {
    private:
        Array<double> value_;
    public:
        Constant() = delete;
        /** \brief Construct a constant with the value. */
        explicit Constant(const Array<double>& value) :value_(value) {}
        /** \brief Move construct a value into the constant. */
        explicit Constant(Array<double>&& value) :value_(value) {}
        /** \brief Get the value saved in this constant. */
        const Array<double>& value() const { return value_; }
    };
}
