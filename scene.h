#pragma once

#include "ray.h"
#include "vector_cuda.h"

#include <stdbool.h>
#include <vector>

namespace pk
{

//#ifdef __CUDA__
//#define LINKAGE __device__
//#else
//#define LINKAGE
//#endif

class IMaterial;

typedef struct _hit {
    float      distance;
    vector3    point;
    vector3    normal;
    IMaterial* material;

    LINKAGE _hit() :
        distance( 0.0f ),
        point( 0, 0, 0 ),
        normal( 0, 0, 0 ),
        material( nullptr ) {}
} hit_info;

class IVisible {
public:
    LINKAGE IVisible() {};
    virtual LINKAGE ~IVisible(){};
    virtual LINKAGE bool hit( const ray& r, float min, float max, hit_info* p_hit ) const = 0;
};


class Scene : virtual public IVisible {
public:
    LINKAGE Scene() {};
    LINKAGE virtual ~Scene() {};

    virtual LINKAGE bool hit( const ray& r, float min, float max, hit_info* p_hit ) const
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
