#include "perf_timer.h"

#include <chrono>
#include <thread>


namespace pk
{
PerfTimer::PerfTimer()
{
    Start();
}


PerfTimer::~PerfTimer()
{
}


void PerfTimer::Start()
{
    m_startTick = std::chrono::high_resolution_clock::now();
    m_stopTick  = std::chrono::high_resolution_clock::time_point( std::chrono::nanoseconds() );
}


void PerfTimer::Stop()
{
    m_stopTick = std::chrono::high_resolution_clock::time_point( std::chrono::nanoseconds() );
}


void PerfTimer::Reset()
{
    m_startTick = std::chrono::high_resolution_clock::now();
    m_stopTick  = std::chrono::high_resolution_clock::time_point( std::chrono::nanoseconds() );
}


double PerfTimer::ElapsedSeconds()
{
    std::chrono::high_resolution_clock::duration elapsedTicks;

    if ( m_stopTick.time_since_epoch().count() > 0 ) {
        elapsedTicks = m_stopTick - m_startTick;
    } else {
        elapsedTicks = std::chrono::high_resolution_clock::now() - m_startTick;
    }

    auto   duration = std::chrono::duration_cast<std::chrono::seconds>( elapsedTicks ).count();
    double seconds  = std::chrono::duration<double>( duration ).count();

    return seconds;
}


double PerfTimer::ElapsedMilliseconds()
{
    std::chrono::high_resolution_clock::duration elapsedTicks;

    if ( m_stopTick.time_since_epoch().count() > 0 ) {
        elapsedTicks = m_stopTick - m_startTick;
    } else {
        elapsedTicks = std::chrono::high_resolution_clock::now() - m_startTick;
    }

    auto   duration     = std::chrono::duration_cast<std::chrono::milliseconds>( elapsedTicks ).count();
    double milliseconds = std::chrono::duration<double>( duration ).count();

    return milliseconds;
}


double PerfTimer::ElapsedNanoseconds()
{
    std::chrono::high_resolution_clock::duration elapsedTicks;

    if ( m_stopTick.time_since_epoch().count() > 0 ) {
        elapsedTicks = m_stopTick - m_startTick;
    } else {
        elapsedTicks = std::chrono::high_resolution_clock::now() - m_startTick;
    }

    auto   duration     = std::chrono::duration_cast<std::chrono::nanoseconds>( elapsedTicks ).count();
    double nanoseconds = std::chrono::duration<double>( duration ).count();

    return nanoseconds;
}


void PerfTimer::DelayMS( unsigned int msec )
{
    std::this_thread::sleep_for( std::chrono::milliseconds( msec ) );
}


uint64_t PerfTimer::SystemTimeMilliseconds()
{
    return (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::system_clock::now().time_since_epoch() ).count();
}


void PerfTimer::SelfTest()
{
    pk::PerfTimer timer;

    int           delays_ns[] = { 1, 2, 5, 10, 100, 1000, 5000 };
    for ( int d : delays_ns ) {
        timer.Reset();
        std::this_thread::sleep_for( std::chrono::nanoseconds( d ) );
        printf( "%d = %f ns\n", d, timer.ElapsedNanoseconds() );
    }

    int           delays[] = { 1, 2, 5, 10, 100, 1000, 5000 };
    for ( int d : delays ) {
        timer.Reset();
        std::this_thread::sleep_for( std::chrono::milliseconds( d ) );
        printf( "%d = %f ms\n", d, timer.ElapsedMilliseconds() );
    }
}

} // namespace pk
