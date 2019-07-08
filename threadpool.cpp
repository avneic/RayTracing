//
// Trivial job system using thread pool
//

#include "threadpool.h"

#include "msgqueue.h"

#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

namespace pk
{

typedef struct _Job {
    jobFunction function;
    void*       context;

    _Job() :
        function( nullptr ),
        context( nullptr ) {}
} Job;

typedef struct _thread_t {
    uint32_t          tid;
    std::thread*      thread;
    std::atomic<bool> shouldExit;

    _thread_t() :
        tid( -1 ),
        thread( nullptr ),
        shouldExit( false )
    {
    }

    _thread_t( const _thread_t& rhs )
    {
        tid        = rhs.tid;
        thread     = std::move( rhs.thread );
        shouldExit = false;
    }
} thread_t;

static std::vector<thread_t> s_threads;
static queue_t               s_job_queue     = INVALID_QUEUE;
static const int             MAX_QUEUE_DEPTH = 1024;

static void _threadWorker( void* context );


//
// Public
//

void threadPoolInit( uint32_t numThreads )
{
    assert( numThreads && s_threads.size() == 0 );
    s_threads.reserve( numThreads );

    uint8_t* buffer = new uint8_t[ sizeof( Job ) * MAX_QUEUE_DEPTH ];
    s_job_queue     = queue_create( sizeof( Job ), MAX_QUEUE_DEPTH, buffer );

    for ( uint32_t i = 0; i < numThreads; i++ ) {
        thread_t t;
        t.tid    = (uint32_t)i;
        t.thread = new std::thread( _threadWorker, (void*)t.tid );
        assert( t.thread->joinable() );
        s_threads.push_back( t );
    }

    printf( "Created %d threads\n", numThreads );
}

bool threadPoolSubmitJob( jobFunction function, void* context )
{
    Job job;
    job.function = function;
    job.context  = context;
    result rval  = queue_send( s_job_queue, &job );

    //printf( "_submitJob: 0x%p( 0x%p ) %d (%zd)\n", function, context, rval, queue_size( s_job_queue ) );

    return true;
}


bool threadPoolDeinit()
{
    //printf( "kill threads\n" );

    for ( int i = 0; i < s_threads.size(); i++ ) {
        thread_t& t  = s_threads[ i ];
        t.shouldExit = true;
    }

    queue_notify_all( s_job_queue );

    for ( int i = 0; i < s_threads.size(); i++ ) {
        thread_t& t = s_threads[ i ];
        t.thread->join();

        //printf( "thread[%d] done\n", i );
    }

    return true;
}


//
// Private
//

static void _threadWorker( void* context )
{
    uint32_t tid = (uint32_t)context;
    printf( "_threadWorker[%d] started\n", tid );

    while ( true ) {
        Job job;

        if ( s_threads[ tid ].shouldExit ) {
            return;
        }

        if ( R_TIMEOUT != queue_receive( s_job_queue, &job, sizeof( job ), std::numeric_limits<unsigned int>::max() ) ) {
            //RenderThreadContext* ctx = (RenderThreadContext*)job.context;

            if ( job.function ) {
                job.function( job.context );

                //std::atomic<uint32_t>* blockCount = ctx->blockCount;
                //uint32_t               count      = blockCount->fetch_add( 1 );
            }
        }
    }
}


} // namespace pk
