#include <random>

#include "utility.h"

namespace chloro
{
    void train_test_split(const DataValues& data, const DataValues& labels,
        DataValues& train_data, DataValues& train_labels,
        DataValues& test_data, DataValues& test_labels,
        const double train_ratio)
    {
        static std::mt19937 generator{ std::random_device()() };
        const std::size_t data_size = data.size();
        std::vector<size_t> permutation(data_size);
        for (size_t i = 0; i < data_size; i++) permutation[i] = i;
        std::shuffle(permutation.begin(), permutation.end(), generator);
        const std::size_t train_size = std::size_t(data_size * train_ratio);
        const std::size_t test_size = data_size - train_size;
        train_data.clear(); train_labels.clear();
        train_data.reserve(train_size); train_labels.reserve(train_size);
        test_data.clear(); test_labels.clear();
        test_data.reserve(test_size); test_labels.reserve(test_size);
        for (size_t i = 0; i < train_size; i++)
        {
            train_data.push_back(data[permutation[i]]);
            train_labels.push_back(labels[permutation[i]]);
        }
        for (size_t i = train_size; i < data_size; i++)
        {
            test_data.push_back(data[permutation[i]]);
            test_labels.push_back(labels[permutation[i]]);
        }
    }
}
