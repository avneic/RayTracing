#pragma once

#include "material.h"
#include "ray.h"

#include <stdbool.h>
#include <vector>

namespace pk
{

// DEPRECATED
class IVisible {
public:
    IVisible() {};
    virtual ~IVisible(){};
    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const = 0;
};


// DEPRECATED
class Scene : virtual public IVisible {
public:
    Scene() {};
    virtual ~Scene() {};

    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const
    {
        bool     rval = false;
        hit_info hit;
        float    closestSoFar = max;

        for ( IVisible* obj : objects ) {
            hit_info tmp;
            if ( obj->hit( r, min, closestSoFar, &tmp ) ) {
                rval         = true;
                closestSoFar = tmp.distance;
                hit          = tmp;
            }
        }

        *p_hit = hit;
        return rval;
    }

    std::vector<IVisible*> objects;
};

} // namespace pk
