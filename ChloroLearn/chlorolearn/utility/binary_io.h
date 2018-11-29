#pragma once

#include <fstream>
#include <vector>

namespace chloro
{
    // Binary input
    template <typename T>
    void read(std::ifstream& stream, T& value)
    {
        stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    }

    template <typename T>
    void read_vector(std::ifstream& stream, std::vector<T>& values)
    {
        std::size_t size;
        read(stream, size);
        values.resize(size);
        stream.read(reinterpret_cast<char*>(&values[0]), size * sizeof(T));
    }

    // Binary output
    template <typename T>
    void write(std::ofstream& stream, const T& value)
    {
        stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    }

    template <typename T>
    void write_vector(std::ofstream& stream, const std::vector<T>& values)
    {
        const size_t size = values.size();
        write(stream, size);
        stream.write(reinterpret_cast<const char*>(&values[0]), size * sizeof(T));
    }
}
