#pragma once

#include "scene.h"

namespace pk
{

class Sphere : virtual public IVisible {
public:
    Sphere();
    Sphere(const vec3& pos, float radius, IMaterial *material);
    Sphere(const Sphere& rhs) = default;
    virtual ~Sphere() = default;
    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const;

    vec3  center;
    float radius;
    IMaterial *material;
};


} // namespace pk
