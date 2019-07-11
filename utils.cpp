#include "utils.h"

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif


namespace pk
{
static std::random_device _rd;
static std::mt19937       _gen( _rd() );
//static std::uniform_real_distribution<float> _random( 0.0f, 1.0f );
static std::uniform_real_distribution<float> _random( 0.0f, 0.9999f );

bool delay( size_t ms )
{
    std::this_thread::sleep_for( std::chrono::milliseconds( ms ) );

    return true;
}

float random()
{
    return _random( _gen );
}

vec3 randomInUnitSphere()
{
    vec3 point;
    do {
        point = 2.0f * vec3( random(), random(), random() ) - vec3( 1, 1, 1 );
    } while ( point.squared_length() >= 1.0f );

    return point;
}

vec3 randomOnUnitDisk()
{
    vec3 point;
    do {
        point = 2.0f * vec3( random(), random(), 0.0f ) - vec3( 1.0f, 1.0f, 0.0f );
    } while ( point.dot( point ) >= 1.0f );

    return point;
}


void check_cuda( cudaError_t result, char const* const function, const char* const filename, int const line )
{
    if ( result ) {
        printf( "error 0x%x [%s:%d]: ", result, filename, line );
        printf( "[%s]\n", function );
        /*DEBUGCHK(0);*/
        cudaDeviceReset();
    }
}

} // namespace pk
