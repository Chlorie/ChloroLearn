#pragma once

#include <fstream>
#include <vector>

namespace chloro
{
    // Binary input

    /**
     * \brief Read a value from a binary stream.
     * \tparam T Type of the value to be read
     * \param stream The stream from which the value is read
     * \param value A non-const reference for returning the read data.
     */
    template <typename T>
    void read(std::ifstream& stream, T& value)
    {
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    }
    /**
     * \brief Read an \c std::vector from a binary stream. 
     * \tparam T Value type of the vector to be read
     * \param stream The stream from which the value is read
     * \param values A non-const reference for returning the read data.
     * \remark The binary structure of the vector in the stream should be a \c size_t value 
     * \a size indicating the size of the vector, followed by \a size values of type \c T, 
     * which are the contents of the vector.
     */
    template <typename T>
    void read_vector(std::ifstream& stream, std::vector<T>& values)
    {
        std::size_t size;
        read(stream, size);
        values.resize(size);
        stream.read(reinterpret_cast<char*>(&values[0]), size * sizeof(T));
    }

    // Binary output

    /**
     * \brief Write a value to a binary stream.
     * \tparam T Type of the value to be written
     * \param stream The stream to which the value is written
     * \param value The value to be written.
     */
    template <typename T>
    void write(std::ofstream& stream, const T& value)
    {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }
    /**
     * \brief Write an \c std::vector to a binary stream.
     * \details The function will first write a \c size_t, the size of the vector to the stream, 
     * and then the values saved in the vector.
     * \tparam T Type of the value to be written
     * \param stream The stream to which the value is written
     * \param values The vector to be written.
     */
    template <typename T>
    void write_vector(std::ofstream& stream, const std::vector<T>& values)
    {
        const size_t size = values.size();
        write(stream, size);
        stream.write(reinterpret_cast<const char*>(&values[0]), size * sizeof(T));
    }
}
