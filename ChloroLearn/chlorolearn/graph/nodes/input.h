#pragma once

#include "../../basic/array.h"

namespace chloro
{
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
