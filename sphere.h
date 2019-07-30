#pragma once

#include "scene.h"
#include "material.h"

namespace pk
{

// DEPRECATED:
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


typedef struct _sphere {
    vector3    center;
    float      radius;
    material_t material; // TODO: should be an index into a materials array
} sphere_t;

bool sphereHit(const sphere_t &sphere, const ray& r, float min, float max, hit_info* p_hit);

} // namespace pk
