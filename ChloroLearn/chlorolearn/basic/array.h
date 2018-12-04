#pragma once

#include <vector>
#include <initializer_list>
#include <type_traits>
#include <algorithm>
#include <numeric>
#include <random>

#include "exceptions.h"

// ReSharper disable CppNonExplicitConvertingConstructor

namespace chloro
{
    using ArrayShape = std::vector<size_t>;
    using DefaultableArrayShape = std::vector<int>;
    inline static const ArrayShape scalar_shape{ 1 };

    /**
     * \brief A flexible multi-dimensional generic array that supports basic operations
     * and other functions like reshaping. Specifically, this type is broadly used in the
     * other parts of this library, more specifically, \c Array<double>.
     * \tparam T Type of data stored in the \c Array, should be an arithmatic type.
     */
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
        // Constructors

        Array() :data_(0), shape_{ 0 } {} /**< \brief Default constructs an empty array with no space for data. */
        Array(const Array&) = default; /**< \brief Copy constructor. */
        Array(Array&&) = default; /**< \brief Move constructor. */
        Array(const T value) :data_{ value }, shape_{ 1 } {} /**< \brief Construct an \c Array of shape 1 with a given value. */
        /** \brief Construct a row vector array with an \c std::initializer_list<T>. */
        Array(std::initializer_list<T> list) :data_(list), shape_{ list.size() } {}
        /** \brief Construct recursively an array with an \c std::initializer_list<Array>. */
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
        /** \brief Implicit converting constructor from an array of a different data type. */
        template <typename U, typename = std::enable_if<std::is_convertible_v<U, T>>>
        Array(const Array<U>& other) : data_(other.data_.begin(), other.data_.end()), shape_(other.shape_) {}
        /** \brief Implicit move converting constructor from an array of a different data type. */
        template <typename U, typename = std::enable_if<std::is_convertible_v<U, T>>>
        Array(Array<U>&& other) : data_(other.data_.begin(), other.data_.end()), shape_(std::move(other.shape_)) {}

        // Construct helpers

        /** \brief Constructs an array filled with zeros with the given shape. */
        static Array zeros(const ArrayShape& shape)
        {
            size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            Array result;
            result.shape_ = shape;
            result.data_.resize(size);
            return result;
        }
        /**
         * \brief Constructs an array filled with random numbers (normally distributed with mean of 0
         * and standard deviation of 1) with the given shape.
         */
        static Array random(const ArrayShape& shape)
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
        /**
         * \brief Constructs an array filled with a specific value with the given shape.
         * \param repeat The repeated value.
         * \param shape The shape of the array.
         * \return The constructed array.
         */
        static Array repeats(const T repeat, const ArrayShape& shape)
        {
            size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
            Array result;
            result.shape_ = shape;
            result.data_ = std::vector<T>(size, repeat);
            return result;
        }

        // Assign operators

        Array& operator=(const Array&) = default; /**< \brief Copy the values of another array into this one. */
        Array& operator=(Array&&) = default; /**< \brief Move the contents of another array into this one. */

        /** \brief Copy the values in a vector to this array. */
        Array& operator=(const std::vector<T>& data)
        {
            if (data.size() == data_.size())
                data_ = data;
            else
                throw MismatchedSizesException("Size of the vector doesn't match that of the array");
            return *this;
        }
        /** \brief Move the values in a vector into this array. */
        Array& operator=(std::vector<T>&& data)
        {
            if (data.size() == data_.size())
                data_ = std::move(data);
            else
                throw MismatchedSizesException("Size of the vector doesn't match that of the array");
            return *this;
        }

        // Properties

        /** \brief Get total element amount of the array. */
        size_t size() const { return data_.size(); }
        /** \brief Get the length of the array on a specific dimension. */
        size_t length_at(const size_t dimension) const
        {
            if (dimension >= shape_.size()) throw ArgumentOutOfRangeException("Index out of range");
            return shape_[dimension];
        }
        /** \brief Get the dimension amount of the array. */
        size_t dimension() const { return shape_.size(); }
        /** \brief Get the shape of the array. */
        const ArrayShape& shape() const { return shape_; }

        // Accessors

        /** \brief Get a read only \c std::vector containing the values in the array. */
        const std::vector<T>& data() const { return data_; }
        /** \brief Get a reference to the value at the given index. */
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
        /** \brief Get a const reference to the value at the given index. */
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
        /** \brief Get a reference to the value at the given index. */
        T& operator()(const std::initializer_list<size_t>& list) { return at(list); }
        /** \brief Get a const reference to the value at the given index. */
        const T& operator()(const std::initializer_list<size_t>& list) const { return at(list); }
        /**
         * \brief Get a reference to the value at the given index using single indexing.
         * Notice that this method does not check the validity of the input.
         */
        T& operator[](const size_t index) { return data_[index]; }
        /**
         * \brief Get a const reference to the value at the given index using single indexing.
         * Notice that this method does not check the validity of the input.
         */
        const T& operator[](const size_t index) const { return data_[index]; }

        // Element-wise operators

        /** \brief Add a value to each of the values in the array. */
        Array& operator+=(T other)
        {
            for (T& value : data_) value += other;
            return *this;
        }
        /** \brief Performs an element-wise add operation. */
        Array& operator+=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] += other.data_[i];
            return *this;
        }
        /** \brief Subtract a value from each of the values in the array. */
        Array& operator-=(T other)
        {
            for (T& value : data_) value -= other;
            return *this;
        }
        /** \brief Performs an element-wise subtract operation. */
        Array& operator-=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] -= other.data_[i];
            return *this;
        }
        /** \brief Multiply a value to each of the values in the array. */
        Array& operator*=(T other)
        {
            for (T& value : data_) value *= other;
            return *this;
        }
        /** \brief Performs an element-wise multiply operation. */
        Array& operator*=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] *= other.data_[i];
            return *this;
        }
        /** \brief Divide each of the values by a value in the array. */
        Array& operator/=(T other)
        {
            for (T& value : data_) value /= other;
            return *this;
        }
        /** \brief Performs an element-wise divide operation. */
        Array& operator/=(const Array& other)
        {
            check_size_match(other);
            const size_t size = data_.size();
            for (size_t i = 0; i < size; i++) data_[i] /= other.data_[i];
            return *this;
        }
        /** \brief Get the element-wise negation of this array. */
        Array operator-() const&
        {
            Array result(*this);
            for (T& value : result.data_) value = -value;
            return result;
        }
        /** \brief Negates every component of this temporary array and returning a temporary \c *this. */
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

        /** \brief Clear all the values to default value of \c T. */
        void clear()
        {
            const size_t size = data_.size();
            data_.clear();
            data_.resize(size);
        }
        /**
         * \brief Reshape the array to a different shape. You can use auto calculation (-1 for the
         * auto length calculation) on at most one dimension.
         */
        void reshape(const DefaultableArrayShape& shape)
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
        /** \brief Force reshaping the array into another shape. Padding and truncating might happen. */
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
        /**
         * \brief Apply a function element-wise in place.
         * \param function A function, taking a \c T as parameter, and returning another \c T as result.
         * \return The result array, which is just \c *this.
         */
        template <typename Func>
        Array& apply_in_place(Func&& function)
        {
            for (T& value : data_) value = function(value);
            return *this;
        }
        /**
         * \brief Apply a function element-wise, save the result in another array and return.
         * \param function A function, taking a \c T as parameter, and returning another \c T as result.
         * \return The result array.
         */
        template <typename Func>
        Array apply(Func&& function) const
        {
            Array result(*this);
            for (T& value : result.data_) value = function(value);
            return result;
        }
        /**
         * \brief Calls the standard \c std::accumulate function on the array.
         * \param initial Initial value of the accumulation.
         * \param function Specify another function other than the default \c std::plus<T>.
         * \return The result of the accumulation.
         */
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
