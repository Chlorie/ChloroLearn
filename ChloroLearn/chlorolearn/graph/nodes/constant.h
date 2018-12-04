#pragma once

#include "../../basic/array.h"

namespace chloro
{
    /**
     * \brief This kind of node content holds a constant array value, which can be queried
     * later in the evaluating process. Back-propagation won't pass to this kind of node.
     */
    class Constant final
    {
    private:
        Array<double> value_;
    public:
        Constant() = delete;
        /** \brief Set a value  */
        explicit Constant(const Array<double>& value) :value_(value) {}
        explicit Constant(Array<double>&& value) :value_(value) {}
        const Array<double>& value() const { return value_; }
    };
}
