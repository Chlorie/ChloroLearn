#pragma once

#include "../basic/array.h"
#include "node.h"

namespace chloro
{
    struct InputParam final
    {
        Node& input_;
        const Array<double>& value_;
        InputParam() = delete;
        InputParam(Node& input, const Array<double>& value) :input_(input), value_(value) {}
    };
}
