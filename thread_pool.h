#pragma once

//
// Trivial job system using thread pools
//

#include "result.h"

#include <cstdint>
#include <functional>

namespace pk
{

typedef uint32_t thread_pool_t;
typedef uint64_t job_t;
typedef uint64_t job_group_t;

#define INVALID_THREAD_POOL ( thread_pool_t( -1 ) )
#define INVALID_JOB ( job_t( -1 ) )
#define INVALID_JOB_GROUP ( job_group_t( -1 ) )
#define INFINITE_TIMEOUT ( uint32_t( -1 ) )

typedef bool ( *jobFunction )( void* context, uint32_t tid );

thread_pool_t threadPoolInit( uint32_t numThreads );

typedef enum {
    THREAD_POOL_SUBMIT_BLOCKING    = 0,
    THREAD_POOL_SUBMIT_NONBLOCKING = 1,
} thread_pool_blocking_t;

template<class TYPE>
job_t threadPoolSubmitJob( thread_pool_t pool, TYPE* object, bool ( TYPE::*method )( void*, uint32_t ), void* context, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );

template<class TYPE>
job_group_t threadPoolSubmitJobs( thread_pool_t pool, TYPE** objects, jobFunction* methods, void** contexts, size_t numJobs, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );

job_t       threadPoolSubmitJob( thread_pool_t pool, jobFunction function, void* context, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );
job_group_t threadPoolSubmitJobs( thread_pool_t pool, jobFunction* functions, void** contexts, size_t numJobs, thread_pool_blocking_t blocking = THREAD_POOL_SUBMIT_BLOCKING );

result      threadPoolWaitForJob( thread_pool_t pool, job_t, uint32_t timeout_ms = INFINITE_TIMEOUT );
result      threadPoolWaitForJobs( thread_pool_t pool, job_group_t, uint32_t timeout_ms = INFINITE_TIMEOUT );

bool        threadPoolDeinit( thread_pool_t pool );

void testThreadPool();

} // namespace pk
