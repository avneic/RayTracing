#pragma once

#include "ray.h"
#include "utils.h"
#include "vec3.h"


namespace pk
{

class Camera {
public:
    Camera() :
        Camera( 50.0f, 2.0f ) {}

    Camera( float vfov, float aspect, float aperture = 1.0f, float focusDistance = std::numeric_limits<float>::max(), const vec3& pos = vec3( 0, 0, 0 ), const vec3& up = vec3( 0, 1, 0 ), const vec3& lookat = vec3( 0, 0, -1 ) )
    {
        lensRadius = aperture / 2.0f;

        float theta      = RADIANS( vfov );
        float halfHeight = tan( theta / 2 );
        float halfWidth  = aspect * halfHeight;

        origin = pos;
        w      = ( pos - lookat ).normalized();
        u      = up.cross( w ).normalized();
        v      = w.cross( u );

        leftCorner = origin - halfWidth * focusDistance * u - halfHeight * focusDistance * v - focusDistance * w;
        horizontal = 2 * halfWidth * focusDistance * u;
        vertical   = 2 * halfHeight * focusDistance * v;

        printf( "Camera: fov %f aspect %f aperture %f (%f, %f, %f) -> (%f, %f, %f) (%f : %f)\n",
            vfov, aspect, aperture, origin.x, origin.y, origin.z, lookat.x, lookat.y, lookat.z, focusDistance, ( origin - lookat ).length() );
    }

    Camera( const Camera& rhs ) :
        origin(rhs.origin),
        leftCorner(rhs.leftCorner),
        horizontal(rhs.horizontal),
        vertical(rhs.vertical),
        u(rhs.u),
        v(rhs.v),
        w(rhs.w),
        lensRadius(rhs.lensRadius)
    {}

    ray getRay( float s, float t ) const
    {
        //return ray( origin, leftCorner + ( s * horizontal ) + ( t * vertical ) - origin );

        vec3 rand   = lensRadius * randomOnUnitDisk();
        vec3 offset = u * rand.x + v * rand.y;
        return ray( origin + offset, leftCorner + ( s * horizontal ) + ( t * vertical ) - origin - offset );
    }

    vec3  origin;
    vec3  leftCorner;
    vec3  horizontal;
    vec3  vertical;
    vec3  u;
    vec3  v;
    vec3  w;
    float lensRadius;
};

} // namespace pk
