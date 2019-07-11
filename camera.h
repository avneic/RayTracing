#pragma once

#include "ray.h"
#include "utils.h"
#include "vector_cuda.h"

#include <cstdint>
#include <limits.h>
#include <cfloat> // for FLT_MAX
#include <cstdio>

namespace pk
{

class Camera {
public:
    LINKAGE Camera() :
        Camera( 50.0f, 2.0f ) {}

    LINKAGE Camera( float vfov, float aspect, float aperture = 1.0f, float focusDistance = FLT_MAX, const vector3& pos = vector3( 0, 0, 0 ), const vector3& up = vector3( 0, 1, 0 ), const vector3& lookat = vector3( 0, 0, -1 ) )
    {
        lensRadius = aperture / 2.0f;

        float theta      = RADIANS( vfov );
        float halfHeight = (float)tan( theta / 2 );
        float halfWidth  = aspect * halfHeight;

        origin = pos;
        w      = ( pos - lookat ).normalized();
        u      = up.cross( w ).normalized();
        v      = w.cross( u );

        leftCorner = origin - halfWidth * focusDistance * u - halfHeight * focusDistance * v - focusDistance * w;
        horizontal = 2 * halfWidth * focusDistance * u;
        vertical   = 2 * halfHeight * focusDistance * v;

        printf( "Camera: fov %4.1f aspect %4.1f aperture %4.1f (%f, %f, %f) -> (%f, %f, %f) (%f : %f)\n",
            vfov, aspect, aperture, origin.x, origin.y, origin.z, lookat.x, lookat.y, lookat.z, focusDistance, ( origin - lookat ).length() );
    }

    LINKAGE Camera( const Camera& rhs ) :
        origin(rhs.origin),
        leftCorner(rhs.leftCorner),
        horizontal(rhs.horizontal),
        vertical(rhs.vertical),
        u(rhs.u),
        v(rhs.v),
        w(rhs.w),
        lensRadius(rhs.lensRadius)
    {}

    LINKAGE ray getRay( float s, float t ) const
    {
        vector3 rand   = lensRadius * randomOnUnitDisk();
        vector3 offset = u * rand.x + v * rand.y;
        return ray( origin + offset, leftCorner + ( s * horizontal ) + ( (1.0f - t) * vertical ) - origin - offset );
    }

    vector3  origin;
    vector3  leftCorner;
    vector3  horizontal;
    vector3  vertical;
    vector3  u;
    vector3  v;
    vector3  w;
    float lensRadius;
};

} // namespace pk
