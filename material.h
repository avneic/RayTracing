#pragma once

#include "ray.h"
#include "scene.h"
#include "vector_cuda.h"

namespace pk
{

typedef enum {
    MATERIAL_DIFFUSE = 0,
    MATERIAL_METAL   = 1,
    MATERIAL_GLASS   = 2,
} material_type_t;

typedef struct _material {
    material_type_t type;
    vector3         albedo;
    float           blur;
    float           refractionIndex;

    _material() :
        type(MATERIAL_DIFFUSE),
        albedo(1,0,1),
        blur(1.0f),
        refractionIndex(1.0f)
    {}

    _material(material_type_t type, vector3 albedo = vector3(1,0,1), float blur = 1.0f, float refractionIndex = 1.0f) :
        type(type),
        albedo(albedo),
        blur(blur),
        refractionIndex(refractionIndex) {}

} material_t;

__host__ __device__ bool materialScatter( const material_t* m, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered );

} // namespace pk
