#pragma once

#include "ray.h"
#include "vector_cuda.h"

namespace pk
{

typedef enum {
    MATERIAL_NONE    = 0,
    MATERIAL_DIFFUSE = 1,
    MATERIAL_METAL   = 2,
    MATERIAL_GLASS   = 3,
} material_type_t;


typedef struct _material {
    material_type_t type;
    vector3         albedo;
    float           blur;
    float           refractionIndex;

    __host__ __device__ _material() :
        //type(MATERIAL_DIFFUSE),
        type( MATERIAL_NONE ),
        albedo( 1, 0, 1 ),
        blur( 1.0f ),
        refractionIndex( 1.0f )
    {
    }

    __host__ __device__ _material( material_type_t type, vector3 albedo = vector3( 1, 0, 1 ), float blur = 1.0f, float refractionIndex = 1.0f ) :
        type( type ),
        albedo( albedo ),
        blur( blur ),
        refractionIndex( refractionIndex ) {}

} material_t;


typedef struct _hit {
    float      distance;
    vector3    point;
    vector3    normal;
    material_t material;

    __host__ __device__ _hit() :
        distance( 0.0f ),
        point( 0, 0, 0 ),
        normal( 0, 0, 0 )
    {
    }
} hit_info;


__host__ __device__ bool materialScatter( const material_t& m, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered );

} // namespace pk
