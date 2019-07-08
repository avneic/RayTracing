#pragma once

//
// Trivial job system using thread pool
//

#include <cstdint>

namespace pk
{

typedef void ( *jobFunction )( const void* );

void threadPoolInit( uint32_t numThreads );
bool threadPoolSubmitJob( jobFunction job, void* context );
bool threadPoolDeinit();

} // namespace pk
