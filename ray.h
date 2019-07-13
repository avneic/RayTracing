#pragma once
#include "vector_cuda.h"

namespace pk
{

class ray {
public:
    __host__ __device__  ray() {};
    __host__ __device__  ray( const ray& rhs ) :
        origin(rhs.origin),
        direction(rhs.direction)
    {}
    __host__ __device__  ray( const vector3& origin, const vector3& direction ) { this->origin = origin, this->direction = direction; }

    __host__ __device__  vector3 point( float distance ) const { return origin + (distance * direction); }

    vector3 origin;
    vector3 direction;
};

} // namespace pk
