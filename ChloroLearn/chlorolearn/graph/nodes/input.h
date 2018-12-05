#pragma once

#include "../../basic/array.h"

namespace chloro
{
    /**
     * \brief This kind of node content holds a placeholder of an input array.
     * \details Before evaluating and back-propagating, the user should use input packs
     * or input parameters to specify the values of \c Input nodes, or the result
     * of the evaluation will be undefined.
     */
    class Input final
    {
    private:
        ArrayShape shape_;
        Array<double> value_;
    public:
        /** \brief Constructs an \c Input object of some specific shape. */
        explicit Input(const ArrayShape& shape) :shape_(shape) {}
        /** \brief Input a value into this object. */
        void input(const Array<double>& input_value);
        /** \brief Get the current saved value in this object. */
        const Array<double>& value() const;
        /** \brief Get the shape of the underlying array. */
        const ArrayShape& shape() const { return shape_; }
    };
}
