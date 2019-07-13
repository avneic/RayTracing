#pragma once

#include "scene.h"
#include "material.h"

namespace pk
{

class Sphere : virtual public IVisible {
public:
    Sphere();
    Sphere( const vector3& pos, float radius, material_t* material );
    Sphere( const Sphere& rhs ) = default;
    virtual ~Sphere()           = default;
    virtual bool hit( const ray& r, float min, float max, hit_info* p_hit ) const;

    vector3    center;
    float      radius;
    material_t* material;
};


} // namespace pk
