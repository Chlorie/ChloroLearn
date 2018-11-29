#pragma once

#include <vector>
#include <initializer_list>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <random>

#include "exceptions.h"

namespace chloro
{
    using ArrayShape = std::vector<size_t>;
    inline static const ArrayShape scalar_shape{ 1 };

    template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
    class Array final
    {
        friend class Array;
    private:
        // Data members
        std::vector<T> data_;
        ArrayShape shape_;

        // Internal implementations
        void check_size_match(const Array& other) const
        {
            if (data_.size() != other.data_.size())
                throw MismatchedSizesException("Sizes of the two arrays don't match");
        }

    public:
        // Typedefs
        using ValueType = T;

        // Constructors
        Array() :data_(0), shape_{ 0 } {} // Default constructs an empty array with no data space
        Array(const Array&) = default; // Copy constructor
        Array(Array&&) = default; // Move constructor
        Array(const T value) :data_{ value }, shape_{ 1 } {}
        Array(std::initializer_list<T> list) :data_(list), shape_{ list.size() } {}
        Array(std::initializer_list<Array> lists)
        {
            bool first = true;
            for (const Array& list : lists)
            {
                if (first)
                    shape_ = list.shape_;
                else if (shape_ != list.shape_)
                    throw MismatchedSizesException("Shapes of the initializer lists don't match");
                data_.insert(data_.end(), list.data_.begin(), list.data_.end());
                first = false;
            }
            shape_.insert(shape_.begin(), lists.size());
        }
        template <typename U, typename = std::enable_if<std::is_convertible_v<U, T>>> // Implicit conversion
        Array(const Array<U>& other) : data_(other.data_.begin(), other.data_.end()), shape_(other.shape_) {}
        template <typename U, typename = std::enable_if<std::is_convertible_v<U, T>>> // Implicit conversion
        Array(Array<U>&& other) : data_(other.data_.begin(), other.data_.end()), shape_(std::move(other.shape_)) {}

        // Construct helpers
        static Array zeros(const ArrayShape& shape) // Constructs an array with given dimensions
        {
            size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            Array result;
            result.shape_ = shape;
            result.data_.resize(size);
            return result;
        }
        static Array random(const ArrayShape& shape) // Constructs an array with standard normally distributed numbers
        {
            static std::mt19937 generator{ std::random_device()() };
            static std::normal_distribution distribution;
            size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            Array result;
            result.shape_ = shape;
            result.data_.reserve(size);
            for (size_t i = 0; i < size; i++) result.data_.push_back(distribution(generator));
            return result;
        }
        static Array repeats(const T repeat, const ArrayShape& shape) // Repeats a value
        {
            size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            Array result;
            result.shape_ = shape;
            result.data_ = std::vector<T>(size, repeat);
            return result;
        }

        // Assign operators
        Array& operator=(const Array&) = default;
        Array& operator=(Array&&) = default;
        Array& operator=(const std::vector<T>& data)
        {
            if (data.size() == data_.size())
                data_ = data;
            else
                throw MismatchedSizesException("Size of the vector doesn't match that of the array");
            return *this;
        }
        Array& operator=(std::vector<T>&& data)
        {
            if (data.size() == data_.size())
                data_ = std::move(data);
            else
                throw MismatchedSizesException("Size of the vector doesn't match that of the array");
            return *this;
        }

        // Properties
        size_t size() const { return data_.size(); }
        size_t length_at(const size_t dimension) const
        {
            if (dimension >= shape_.size()) throw ArgumentOutOfRangeException("Index out of range");
            return shape_[dimension];
        }
        size_t dimension() const { return shape_.size(); }
        const ArrayShape& shape() const { return shape_; }

        // Accessors
        const std::vector<T>& data() const { return data_; }
        T& at(const std::initializer_list<size_t>& list) // Specify the index by an initializer_list
        {
            if (list.size() != shape_.size())
                throw MismatchedSizesException("Dimension of input is not the same as that of the array");
            int index = 0;
            int i = 0;
            for (size_t value : list)
            {
                if (value >= shape_[i]) throw ArgumentOutOfRangeException("Index out of range");
                index = index * shape_[i] + value;
                i++;
            }
            return data_[index];
        }
        const T& at(const std::initializer_list<size_t>& list) const // Const version
        {
            if (list.size() != shape_.size())
                throw MismatchedSizesException("Dimension of input is not the same as that of the array");
            int index = 0;
            int i = 0;
            for (size_t value : list)
            {
                if (value >= shape_[i]) throw ArgumentOutOfRangeException("Index out of range");
                index = index * shape_[i] + value;
                i++;
            }
            return data_[index];
        }
        T& operator()(const std::initializer_list<size_t>& list) { return at(list); }
        const T& operator()(const std::initializer_list<size_t>& list) const { return at(list); }
        T& operator[](const size_t index) { return data_[index]; }
        const T& operator[](const size_t index) const { return data_[index]; }
        void set_values(const std::vector<T>& values)
        {
            if (values.size() != data_.size())
                throw MismatchedSizesException("New value and old value are not of the same size");
            data_ = values;
        }
        void set_values(std::vector<T>&& values)
        {
            if (values.size() != data_.size())
                throw MismatchedSizesException("New value and old value are not of the same size");
            data_ = std::move(values);
        }

        // Element-wise operators
        Array& operator+=(T other)
        {
            for (T& value : data_) value += other;
            return *this;
        }
        Array& operator+=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] += other.data_[i];
            return *this;
        }
        Array& operator-=(T other)
        {
            for (T& value : data_) value -= other;
            return *this;
        }
        Array& operator-=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] -= other.data_[i];
            return *this;
        }
        Array& operator*=(T other)
        {
            for (T& value : data_) value *= other;
            return *this;
        }
        Array& operator*=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] *= other.data_[i];
            return *this;
        }
        Array& operator/=(T other)
        {
            for (T& value : data_) value /= other;
            return *this;
        }
        Array& operator/=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] /= other.data_[i];
            return *this;
        }
        Array operator-() const&
        {
            Array result(*this);
            for (T& value : result.data_) value = -value;
            return result;
        }
        Array operator-() &&
        {
            for (T& value : data_) value = -value;
            return std::move(*this);
        }

        // Friend operators
        friend Array operator+(T value, const Array& array)
        {
            Array result(array);
            return result += value;
        }
        friend Array operator+(const Array& array, T value) { return value + array; }
        friend Array operator+(T value, Array&& array) { return array += value; }
        friend Array operator+(Array&& array, T value) { return array += value; }
        friend Array operator+(const Array& left, const Array& right)
        {
            Array result(left);
            return result += right;
        }
        friend Array operator+(const Array& left, Array&& right) { return right += left; }
        friend Array operator+(Array&& left, const Array& right) { return std::move(left += right); }
        friend Array operator+(Array&& left, Array&& right) { return std::move(left += right); }
        friend Array operator-(T value, const Array& array)
        {
            Array result(-array);
            return result += value;
        }
        friend Array operator-(const Array& array, T value)
        {
            Array result(array);
            return result -= value;
        }
        friend Array operator-(T value, Array&& array) { return std::move(-array += value); }
        friend Array operator-(Array&& array, T value) { return std::move(array -= value); }
        friend Array operator-(const Array& left, const Array& right)
        {
            Array result(left);
            return result -= right;
        }
        friend Array operator-(const Array& left, Array&& right) { return std::move(-right += left); }
        friend Array operator-(Array&& left, const Array& right) { return std::move(left -= right); }
        friend Array operator-(Array&& left, Array&& right) { return std::move(left -= right); }
        friend Array operator*(T value, const Array& array)
        {
            Array result(array);
            return result *= value;
        }
        friend Array operator*(const Array& array, T value) { return value * array; }
        friend Array operator*(T value, Array&& array) { return std::move(array *= value); }
        friend Array operator*(Array&& array, T value) { return std::move(array *= value); }
        friend Array operator*(const Array& left, const Array& right)
        {
            Array result(left);
            return result *= right;
        }
        friend Array operator*(const Array& left, Array&& right) { return std::move(right *= left); }
        friend Array operator*(Array&& left, const Array& right) { return std::move(left *= right); }
        friend Array operator*(Array&& left, Array&& right) { return std::move(left *= right); }
        friend Array operator/(T value, const Array& array)
        {
            Array result = Array<T>::repeats(value, array.shape_);
            return result /= array;
        }
        friend Array operator/(const Array& array, T value)
        {
            Array result(array);
            return result /= value;
        }
        friend Array operator/(T value, Array&& array)
        {
            for (T& element : array.data_) element = T{ 1 } / element;
            return std::move(array *= value);
        }
        friend Array operator/(Array&& array, T value) { return std::move(array /= value); }
        friend Array operator/(const Array& left, const Array& right)
        {
            Array result(left);
            return result /= right;
        }
        friend Array operator/(const Array& left, Array&& right)
        {
            for (T& value : right.data_) value = T{ 1 } / value;
            return std::move(right *= left);
        }
        friend Array operator/(Array&& left, const Array& right) { return std::move(left /= right); }
        friend Array operator/(Array&& left, Array&& right) { return std::move(left /= right); }

        // Miscellaneous methods
        void clear() // Clear all elements to default value
        {
            const size_t size = data_.size();
            data_.clear();
            data_.resize(size);
        }
        void reshape(const ArrayShape& shape)
        {
            int automatic = -1;
            size_t size = 1;
            shape_.clear();
            for (size_t i = 0; i < shape.size(); i++)
            {
                if (shape[i] <= 0)
                {
                    // Automatic dimension
                    if (shape[i] == -1)
                    {
                        if (automatic == -1)
                            automatic = i;
                        else
                            throw IllegalArgumentException("Multiple automatic dimensions");
                    }
                    else
                        throw ArgumentOutOfRangeException("The lengths should be positive or -1 for automatic");
                }
                else
                    size *= shape[i];
                shape_.push_back(shape[i]);
            }
            const size_t data_size = data_.size();
            // No automatic dimension
            if (automatic == -1)
            {
                if (size != data_size)
                    throw MismatchedSizesException("Sizes don't match");
            }
            else
            {
                if (data_size % size != 0)
                    throw IllegalArgumentException("Automatic dimension is not an integer");
                shape_[automatic] = data_size / size;
            }
        }
        void force_reshape(const ArrayShape& shape)
        {
            shape_.clear();
            size_t size = 1;
            for (size_t value : shape)
            {
                if (value <= 0) throw ArgumentOutOfRangeException("Lengths should be positive");
                shape_.push_back(value);
                size *= value;
            }
            data_.resize(size);
        }
        template <typename Func>
        Array& apply_in_place(Func&& function)
        {
            for (T& value : data_) value = function(value);
            return *this;
        }
        template <typename Func>
        Array apply(Func&& function) const
        {
            Array result(*this);
            for (T& value : result.data_) value = function(value);
            return result;
        }
        template <typename Func = std::plus<T>>
        T accumulate(T initial, Func&& function = std::plus<T>()) const
        {
            return std::accumulate(data_.begin(), data_.end(), initial, function);
        }

        // Stream output
        friend std::ostream& operator<<(std::ostream& stream, const Array& array)
        {
            std::vector<size_t> periods(array.shape_);
            std::reverse(periods.begin(), periods.end());
            std::partial_sum(periods.begin(), periods.end(), periods.begin(), std::multiplies<size_t>());
            const size_t size = array.size();
            for (size_t i = 0; i < size; i++)
            {
                if (i != 0) stream << ", ";
                for (size_t period : periods)
                    if (i % period == 0)
                        stream << '[';
                stream << array.data_[i];
                for (size_t period : periods)
                    if ((i + 1) % period == 0)
                        stream << ']';
            }
            return stream;
        }
    };
}
