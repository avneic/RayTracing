#pragma once

#include "ray.h"
#include "vec3.h"


namespace pk
{

#define M_PI 3.14159265358979323846f
#define RADIANS( x ) ( (x)*M_PI / 180.0f )

class Camera {
public:
    //Camera() :
    //    origin( 0, 0, 0 ),
    //    leftCorner( -2, -1, -1 ),
    //    horizontal( 4.0, 0.0, 0.0 ),
    //    vertical( 0.0, 2.0, 0.0 )
    //{
    //}

    Camera() :
        Camera( 50.0f, 2.0f ) {}

    Camera( float vfov, float aspect, const vec3& pos = vec3( 0, 0, 0 ), const vec3& up = vec3( 0, 1, 0 ), const vec3& lookat = vec3( 0, 0, -1 ) )
    {
        float theta      = RADIANS( vfov );
        float halfHeight = tan( theta / 2 );
        float halfWidth  = aspect * halfHeight;

        origin = pos;
        vec3 w = ( pos - lookat ).normalized();
        vec3 u = up.cross( w ).normalized();
        vec3 v = w.cross( u );

        //leftCorner       = vec3( -halfWidth, -halfHeight, -1 );
        //horizontal       = vec3( 2 * halfWidth, 0, 0 );
        //vertical         = vec3( 0, 2 * halfHeight, 0 );
        leftCorner = origin - halfWidth * u - halfHeight * v - w;
        horizontal = 2 * halfWidth * u;
        vertical   = 2 * halfHeight * v;
    }

    ray getRay( float u, float v ) const { return ray( origin, leftCorner + ( u * horizontal ) + ( v * vertical ) - origin ); }

    vec3 origin;
    vec3 leftCorner;
    vec3 horizontal;
    vec3 vertical;
};

} // namespace pk
