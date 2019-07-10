#pragma once

//
// Trivial job system using thread pools
//

#include <cstdint>

namespace pk
{

typedef void ( *jobFunction )( uint32_t tid, const void* context );
typedef uint32_t thread_pool_t;
#define INVALID_THREAD_POOL ( thread_pool_t( -1 ) )

thread_pool_t threadPoolInit( uint32_t numThreads );
bool          threadPoolSubmitJob( thread_pool_t pool, jobFunction job, void* context, bool blocking = true );
bool          threadPoolDeinit( thread_pool_t pool );

} // namespace pk
