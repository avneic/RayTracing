//
// Trivial job system using thread pool
//

#include "threadpool.h"

#include "msgqueue.h"
#include "utils.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <mutex>
#include <thread>
#include <vector>

namespace pk
{

static const int MAX_THREAD_POOLS = 4;
static const int MAX_QUEUE_DEPTH  = 512;

typedef struct _job {
    jobFunction function;
    void*       context;

    _job() :
        function( nullptr ),
        context( nullptr ) {}
} _job_t;

typedef struct _thread {
    uint32_t          tid;
    thread_pool_t     pool;
    std::thread*      thread;
    std::atomic<bool> shouldExit;

    _thread() :
        tid( -1 ),
        thread( nullptr ),
        shouldExit( false )
    {
    }

    _thread( const _thread& rhs )
    {
        tid        = rhs.tid;
        pool       = rhs.pool;
        thread     = std::move( rhs.thread );
        shouldExit = false;
    }
} _thread_t;


typedef struct _thread_pool {
    std::mutex             mutex;
    std::vector<_thread_t> threads;
    queue_t                job_queue;
    bool                   free;

    _thread_pool() :
        free( true ),
        job_queue( INVALID_QUEUE ) {}
} _thread_pool_t;


static std::mutex     s_pools_mutex;
static _thread_pool_t s_pools[ MAX_THREAD_POOLS ];

static bool _valid( thread_pool_t pool );
static void _threadWorker( void* context );


//
// Public
//

thread_pool_t threadPoolInit( uint32_t numThreads )
{
    assert( numThreads );

    _thread_pool_t* tp     = nullptr;
    thread_pool_t   handle = INVALID_THREAD_POOL;

    std::lock_guard<std::mutex> lock( s_pools_mutex );

    for ( int i = 0; i < ARRAY_SIZE( s_pools ); i++ ) {
        std::lock_guard<std::mutex> lock_queue( s_pools[ i ].mutex );

        if ( s_pools[ i ].free ) {
            tp       = &s_pools[ i ];
            tp->free = false;
            handle   = (thread_pool_t)i;
            break;
        }
    }

    if ( !tp )
        return INVALID_THREAD_POOL;

    tp->threads.reserve( numThreads );

    uint8_t* buffer = new uint8_t[ sizeof( _job_t ) * MAX_QUEUE_DEPTH ];
    tp->job_queue   = queue_create( sizeof( _job_t ), MAX_QUEUE_DEPTH, buffer );

    for ( uint32_t i = 0; i < numThreads; i++ ) {
        _thread_t t;
        t.pool = handle;
        t.tid  = (uint32_t)i;

        tp->threads.push_back( t );
        tp->threads[ i ].thread = new std::thread( _threadWorker, (void*)&tp->threads[ i ] );
        assert( tp->threads[ i ].thread->joinable() );
    }

    printf( "Created pool %d, %d threads\n", handle, numThreads );

    return handle;
}

bool threadPoolSubmitJob( thread_pool_t pool, jobFunction function, void* context, bool blocking )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    _job_t job;
    job.function = function;
    job.context  = context;

    result rval = R_OK;
    if ( blocking ) {
        rval = queue_send_blocking( tp->job_queue, &job );
    } else {
        rval = queue_send( tp->job_queue, &job );
    }

    //printf( "_submit_job_t: 0x%p( 0x%p ) %d (%zd)\n", function, context, rval, queue_size( s_job_queue ) );

    return (rval == R_OK);
}


bool threadPoolDeinit( thread_pool_t pool )
{
    if ( !_valid( pool ) )
        return false;

    _thread_pool_t* tp = &s_pools[ pool ];

    for ( int i = 0; i < tp->threads.size(); i++ ) {
        _thread_t& t = tp->threads[ i ];
        t.shouldExit = true;
    }

    queue_notify_all( tp->job_queue );

    for ( int i = 0; i < tp->threads.size(); i++ ) {
        _thread_t& t = tp->threads[ i ];
        t.thread->join();

        //printf( "thread[%d] done\n", i );
    }

    return true;
}


//
// Private
//

static bool _valid( thread_pool_t pool )
{
    if ( pool == INVALID_THREAD_POOL || pool >= ARRAY_SIZE( s_pools ) ) {
        return false;
    }

    return true;
}


// Call the user-supplied function, passing a thread ID (informational) and the user-supplied function context
static void _threadWorker( void* context )
{
    _thread_t*      thread = (_thread_t*)context;
    _thread_pool_t* tp     = &s_pools[ thread->pool ];

    //printf( "_threadWorker[%d:%d] started\n", thread->pool, thread->tid );

    while ( true ) {
        _job_t job;

        if ( tp->threads[ thread->tid ].shouldExit ) {
            return;
        }

        if ( R_TIMEOUT != queue_receive( tp->job_queue, &job, sizeof( job ), std::numeric_limits<unsigned int>::max() ) ) {
            if ( job.function ) {
                uint32_t tid = uint32_t( thread->pool << 16 | thread->tid );
                job.function( tid, job.context );
            }
        }
    }
}


} // namespace pk
