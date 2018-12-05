#pragma once

#include <exception>

namespace chloro
{
    /** \brief Base class for all the exceptions in this library. */
    class ChloroException : public std::exception
    {
    public:
        ChloroException() = default; /**< \brief Default constructor. */
        explicit ChloroException(const char* message) : std::exception(message) {} /**< Construct using a message string. */
    };

#define CHLORO_EXCEPTION(name) class name : public ChloroException \
    { \
    public: \
        name() = default; \
        explicit name(const char* message): ChloroException(message) {} \
    }

    CHLORO_EXCEPTION(ArgumentOutOfRangeException);
    CHLORO_EXCEPTION(EmptyValueException);
    CHLORO_EXCEPTION(MismatchedSizesException);
    CHLORO_EXCEPTION(IllegalArgumentException);
    CHLORO_EXCEPTION(IllegalOperationException);
    CHLORO_EXCEPTION(NotImplementedException);

#undef CHLORO_EXCEPTION
}
