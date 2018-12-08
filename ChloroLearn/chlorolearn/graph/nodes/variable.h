#pragma once

#include <vector>

namespace chloro
{
    /**
     * \brief This kind of node content holds an array variable.
     * \details The value of a variable can be updated during back-propagation, so they 
     * represents adjustable parameters in neural networks.
     */
    class Variable final
    {
    private:
        Array<double> value_;
    public:
        Variable() = delete;
        /**
         * \brief Construct a variable with the specific array size, and initialize the value
         * of it to zero.
         */
        explicit Variable(const ArrayShape& size) :value_(Array<double>::zeros(size)) {}
        /** \brief Get current value of this variable. */
        const Array<double>& value() const { return value_; }
        /** \brief Explicitly set the value of this variable to some array. */
        void set_value(const Array<double>& value) { value_ = value; }
        /** \brief Explicitly set the value of this variable by moving in some array. */
        void set_value(Array<double>&& value) { value_ = std::move(value); }
        /** \brief Substract an array value from current value element-wisely. */
        void substract_from_current(const Array<double>& decrement) { value_ -= decrement; }
    };
}
