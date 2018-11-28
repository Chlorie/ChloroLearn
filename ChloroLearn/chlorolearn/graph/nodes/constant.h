#pragma once

#include "../../basic/array.h"

namespace chloro
{
    class Constant final
    {
    private:
        Array<double> value_;
    public:
        Constant() = delete;
        explicit Constant(const Array<double>& value) :value_(value) {}
        explicit Constant(Array<double>&& value) :value_(value) {}
        const Array<double>& value() const { return value_; }
    };
}
