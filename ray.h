#pragma once
#include "vector.h"

namespace pk
{

class ray {
public:
    ray() = default;
    ray( const ray& rhs ) :
        origin(rhs.origin),
        direction(rhs.direction)
    {}
    ray( const vec3& origin, const vec3& direction ) { this->origin = origin, this->direction = direction; }

    vec3 point( float distance ) const { return origin + (distance * direction); }

    vec3 origin;
    vec3 direction;
};

} // namespace pk
