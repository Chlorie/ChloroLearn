#pragma once

#include "../../basic/array.h"

namespace chloro
{
    /**
     * \brief This kind of node content holds a placeholder of an input array.
     * Before evaluating and back-propagating, the user should use input packs
     * or input parameters to specify the values of \c Input nodes, or the result
     * of the evaluation will be undefined.
     */
    class Input final
    {
    private:
        ArrayShape shape_;
        Array<double> value_;
    public:
        explicit Input(const ArrayShape& size) :shape_(size) {}
        void input(const Array<double>& input_value);
        const Array<double>& value() const;
        const ArrayShape& shape() const { return shape_; }
    };
}
