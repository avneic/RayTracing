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

__host__ __device__ float random()
{
#ifdef __NVCC__
    return 1.0f;
#else
    return _random( _gen );
#endif
}

__host__ __device__ vector3 randomInUnitSphere()
{
    vector3 point;
    unsigned int maxTries = 20;
    do {
        point = 2.0f * vector3( random(), random(), random() ) - vector3( 1, 1, 1 );
    } while ( point.squared_length() >= 1.0f && maxTries--);

    return point;
}

__host__ __device__ vector3 randomOnUnitDisk()
{
    vector3 point;
    unsigned int maxTries = 20;
    do {
        point = 2.0f * vector3( random(), random(), 0.0f ) - vector3( 1.0f, 1.0f, 0.0f );
    } while ( point.dot( point ) >= 1.0f && maxTries--);

    return point;
}


void check_cuda( cudaError_t result, char const* const function, const char* const filename, int const line )
{
    if ( result ) {
        printf( "error 0x%x [%s:%d]: ", result, filename, line );
        printf( "[%s] ", function );
        printf( "%s : %s\n", cudaGetErrorName( result ), cudaGetErrorString( result ) );
        /*DEBUGCHK(0);*/
        //cudaDeviceReset();
    }
}

} // namespace pk
