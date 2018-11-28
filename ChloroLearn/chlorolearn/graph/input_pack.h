#pragma once

#include <vector>

#include "../basic/array.h"
#include "node.h"

namespace chloro
{
    struct InputPack final
    {
        Node& input_;
        std::vector<Array<double>>& pack_;
        InputPack() = delete;
        InputPack(Node& input, std::vector<Array<double>>& pack) :input_(input), pack_(pack) {}
    };
}
