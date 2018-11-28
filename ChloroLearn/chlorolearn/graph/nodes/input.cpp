#include "input.h"
#include "../../basic/exceptions.h"

namespace chloro
{
    void Input::input(const Array<double>& input_value)
    {
        const size_t dimension = input_value.dimension();
        for (size_t i = 0; i < dimension; i++)
            if (input_value.length_at(i) != shape_[i])
                throw MismatchedSizesException("Input size doesn't match node size");
        value_ = input_value;
    }

    const Array<double>& Input::value() const
    {
        if (value_.size() == 0)
            throw EmptyValueException("There's no value in this node");
        return value_;
    }
}
