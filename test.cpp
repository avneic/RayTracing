#include "test.h"

#include "perf_timer.h"
#include "threadpool.h"
#include "utils.h"

#include <atomic>
#include <cstdio>
#include <math.h>

namespace pk
{

//const unsigned int TOTAL_ELEMENTS = ( 1 << 20 );
const unsigned int TOTAL_ELEMENTS = ( 1 << 29 );
const unsigned int BLOCK_SIZE     = TOTAL_ELEMENTS / 4;
const unsigned int NUM_BLOCKS     = TOTAL_ELEMENTS / BLOCK_SIZE;

static void add( int start, int n, float* x, float* y )
{
    for ( int i = 0; i < n; i++ )
        y[ i ] = x[ i ] + y[ i ];
}

void testCPU()
{
    PerfTimer t;

    float* x = new float[ TOTAL_ELEMENTS ];
    float* y = new float[ TOTAL_ELEMENTS ];

    // initialize x and y arrays on the host
    for ( int i = 0; i < TOTAL_ELEMENTS; i++ ) {
        x[ i ] = 1.0f;
        y[ i ] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    add( 0, TOTAL_ELEMENTS, &x[ 0 ], &y[ 0 ] );

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for ( int i = 0; i < TOTAL_ELEMENTS; i++ )
        maxError = fmax( maxError, fabs( y[ i ] - 3.0f ) );

    printf( "Max error: %f\n", maxError );

    // Free memory
    delete[] x;
    delete[] y;

    printf( "Elapsed ms: %d\n", (uint32_t)t.ElapsedMilliseconds() );
}


typedef struct {
    int                start;
    int                num;
    float*             x;
    float*             y;
    int                total;
    std::atomic<bool>* complete;
} thread_context_t;


static void _cpu_add_thread( uint32_t tid, const void* context )
{
    thread_context_t* ctx = (thread_context_t*)context;
    //printf( "thread [%d] [%d : %d]\n", tid, ctx->start, ctx->start + ctx->num );

    add( ctx->start, ctx->num, ctx->x, ctx->y );

    if ( ctx->start + ctx->num >= ctx->total ) {
        *( ctx->complete ) = true;
    }
}


void testCPUThreaded()
{
    printf("Threaded: %d blocks of %d\n", NUM_BLOCKS, BLOCK_SIZE);

    thread_pool_t     tp       = threadPoolInit( 4 );
    thread_context_t* contexts = new thread_context_t[ NUM_BLOCKS ];

    PerfTimer t;

    float* x = new float[ TOTAL_ELEMENTS ];
    float* y = new float[ TOTAL_ELEMENTS ];

    // initialize x and y arrays on the host
    for ( int i = 0; i < TOTAL_ELEMENTS; i++ ) {
        x[ i ] = 1.0f;
        y[ i ] = 2.0f;
    }

    // Run kernel on 1M elements on the CPU
    std::atomic<bool> complete = false;
    for ( int i = 0; i < NUM_BLOCKS; i++ ) {
        thread_context_t& ctx = contexts[ i ];
        ctx.start             = i * BLOCK_SIZE;
        ctx.num               = BLOCK_SIZE;
        ctx.total             = TOTAL_ELEMENTS;
        ctx.x                 = &x[ctx.start];
        ctx.y                 = &y[ctx.start];
        ctx.complete          = &complete;

        bool blocking = false;

        threadPoolSubmitJob( tp, _cpu_add_thread, &ctx, blocking );
    }

    // Wait for threads to complete
    while ( !complete ) {
        //delay( 1 );
    }

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for ( int i = 0; i < TOTAL_ELEMENTS; i++ )
        maxError = fmax( maxError, fabs( y[ i ] - 3.0f ) );

    printf( "Max error: %f\n", maxError );

    // Free memory
    delete[] x;
    delete[] y;

    printf( "Elapsed ms: %d\n", (uint32_t)t.ElapsedMilliseconds() );
    threadPoolDeinit( tp );
}

void testCUDA()
{
}
} // namespace pk
