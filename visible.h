#pragma once

#include "ray.h"
#include "vector.h"

#include <stdbool.h>
#include <vector>

namespace pk
{

class IMaterial;

typedef struct _hit {
    float      t;
    vec3       point;
    vec3       normal;
    IMaterial* material;

    _hit() :
        t(0.0f),
        point(0, 0, 0),
        normal(0, 0, 0),
        material(nullptr) {}
} hit_info;

class IVisible {
public:
    virtual ~IVisible(){};
    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const = 0;
};


class Scene : virtual public IVisible {
public:
    Scene()          = default;
    virtual ~Scene() = default;

    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const
    {
        bool     rval = false;
        hit_info hit;

        for ( IVisible* obj : objects ) {
            hit_info tmp;
            if ( obj->hit( r, min, max, &tmp ) ) {
                rval = true;
                hit  = tmp;
            }
        }

        *p_hit = hit;
        return rval;
    }

    std::vector<IVisible*> objects;
};

} // namespace pk
