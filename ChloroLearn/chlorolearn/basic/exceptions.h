#pragma once

#include <exception>

namespace chloro
{
    // Base class
    class ChloroException : public std::exception
    {
    public:
        ChloroException() = default;
        explicit ChloroException(const char* message) : std::exception(message) {}
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
