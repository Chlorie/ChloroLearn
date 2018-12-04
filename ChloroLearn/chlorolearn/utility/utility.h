#pragma once

#include <vector>

#include "../basic/array.h"

namespace chloro
{
    using DataValues = std::vector<Array<double>>;
    /**
     * \brief Split the given data and labels to training set and test set.
     * \param data The input values that needs to be splitted.
     * \param labels The labels corresponding to the input data that needs to be splitted as well.
     * \param train_data Result parameter. Training set data after the split.
     * \param train_labels Result parameter. Training set labels after the split.
     * \param test_data Result parameter. Test set data after the split.
     * \param test_labels Result parameter. Test set labels after the split.
     * \param train_ratio The propotion of training set to the whole set of data and labels.
     */
    void train_test_split(const DataValues& data, const DataValues& labels,
        DataValues& train_data, DataValues& train_labels,
        DataValues& test_data, DataValues& test_labels,
        double train_ratio = 0.8);
}
