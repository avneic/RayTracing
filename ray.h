#pragma once
#include "vector_cuda.h"

namespace pk
{

//#ifdef __CUDA__
#define LINKAGE __device__ 
//#else
//#define LINKAGE
//#endif

class ray {
public:
    LINKAGE ray() {};
    LINKAGE ray( const ray& rhs ) :
        origin(rhs.origin),
        direction(rhs.direction)
    {}
    LINKAGE ray( const vector3& origin, const vector3& direction ) { this->origin = origin, this->direction = direction; }

    LINKAGE vector3 point( float distance ) const { return origin + (distance * direction); }

    vector3 origin;
    vector3 direction;
};

} // namespace pk
