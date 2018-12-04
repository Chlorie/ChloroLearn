#pragma once

#include <vector>

namespace chloro
{
    /**
     * \brief This kind of node content holds an array variable. The value of a variable
     * can be updated during back-propagation, so they represents adjustable parameters
     * in neural networks.
     */
    class Variable final
    {
    private:
        Array<double> value_;
    public:
        Variable() = delete;
        explicit Variable(const ArrayShape& size) :value_(Array<double>::random(size)) {}
        const Array<double>& value() const { return value_; }
        void set_value(const Array<double>& value) { value_ = value; }
        void set_value(Array<double>&& value) { value_ = std::move(value); }
        void substract_from_current(const Array<double>& decrement) { value_ -= decrement; }
    };
}
