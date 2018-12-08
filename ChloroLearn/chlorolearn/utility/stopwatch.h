#pragma once

#include <chrono>

namespace chloro
{
    /** \brief An utility class for timing the execution time of a part of the program. */
    class Stopwatch final
    {
        using Clock = std::chrono::high_resolution_clock;
        using TimePoint = Clock::time_point;
    private:
        TimePoint start_;
        TimePoint stop_;
    public:
        /** \brief Constructs a \c Stopwatch object and start the clock. */
        Stopwatch() :start_(Clock::now()) {}
        /** \brief Reset the start point of this stopwatch. */
        void restart() { start_ = Clock::now(); }
        /** \brief Stop the stopwatch. */
        void stop() { stop_ = Clock::now(); }
        /**
         * \brief Get the time difference between the last time this stopwatch is stopped
         * and the time when this object is constructed (started).
         */
        std::chrono::nanoseconds elapsed_time() const { return stop_ - start_; }
        /** \brief Helper method for getting \c elapsed_time() and converting it into seconds. */
        double seconds() const { return elapsed_time().count() / 1e9; }
    };
}
