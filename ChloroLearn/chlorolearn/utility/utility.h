#pragma once

#include <vector>

#include "../basic/array.h"

namespace chloro
{
    using DataValues = std::vector<Array<double>>;
    void train_test_split(const DataValues& data, const DataValues& labels,
        DataValues& train_data, DataValues& train_labels,
        DataValues& test_data, DataValues& test_labels,
        double train_ratio = 0.8);
}
