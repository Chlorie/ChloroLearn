#pragma once

#include "../basic/array.h"
#include "node.h"

namespace chloro
{
    /** \brief An aggregate struct for providing an input array for an \c Input node. */
    struct InputParam final
    {
        Node& input; /**< \brief Reference to the node containing an \c Input content. */
        const Array<double>& value; /**< Const reference to the array to input. */
        InputParam() = delete;
    };
}
