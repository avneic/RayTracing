#pragma once

#include "ray.h"
#include "vec3.h"


namespace pk {

    class Camera {
    public:
        Camera() :
            origin(0,0,0),
            left_corner(-2,-1,-1),
            horizontal(4.0, 0.0, 0.0),
            vertical(0.0, 2.0, 0.0)
        {
        }

        //ray getRay(float u, float v) const { return ray(origin, upper_left_corner + (u*horizontal) + (v*vertical) - origin); }
        ray getRay(float u, float v) const { return ray(origin, left_corner + (u*horizontal) + (v*vertical) - origin); }

        vec3 origin;
        vec3 left_corner;
        vec3 horizontal;
        vec3 vertical;
    };

}
