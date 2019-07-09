#pragma once

#include <chrono>
#include <stdint.h>

namespace pk
{
class PerfTimer {
public:
    PerfTimer();
    virtual ~PerfTimer();

    void   Start();
    void   Stop();
    void   Reset();
    double ElapsedSeconds();
    double ElapsedMilliseconds();
    double ElapsedNanoseconds();

    static void DelayMS( unsigned int msec );
    static uint64_t SystemTimeMilliseconds();
    static void SelfTest();

protected:
    std::chrono::high_resolution_clock::time_point m_startTick;
    std::chrono::high_resolution_clock::time_point m_stopTick;
};
} // namespace pk
