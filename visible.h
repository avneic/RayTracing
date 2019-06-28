#pragma once

#include <vector>
#include <stdbool.h>
#include "ray.h"
#include "vector.h"

namespace pk
{

typedef struct _hit {
    float t;
    vec3  point;
    vec3  normal;
} hit_info;

class IVisible {
public:
    virtual ~IVisible(){};
    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const = 0;
};


class Scene : virtual public IVisible {
public:
    Scene() = default;
    virtual ~Scene() = default;

    virtual bool hit(const ray& r, float min, float max, hit_info* p_hit) const
    {
        bool rval = false;
        hit_info hit;

        for (IVisible* obj : objects) {
            hit_info tmp;
            if (obj->hit( r, 0, std::numeric_limits<float>::max(), &tmp )) {
                rval = true;
                hit = tmp;
            }
        }

        *p_hit = hit;
        return rval;
    }

    std::vector<IVisible*> objects;
};

} // namespace pk
