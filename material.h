#pragma once

#include "ray.h"
#include "scene.h"
#include "vector_cuda.h"

namespace pk
{

class IMaterial {
public:
    virtual bool scatter( const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered ) const = 0;
};


class Diffuse : virtual public IMaterial {
public:
    Diffuse() :
        albedo( 1.0f, 1.0f, 1.0f ){};
    Diffuse( float r, float g, float b ) :
        albedo( r, g, b ) {}
    Diffuse( vector3 a ) :
        albedo( a ) {}

    virtual bool scatter( const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered ) const;

    vector3 albedo;
};


class Metal : virtual public IMaterial {
public:
    Metal() :
        albedo( 1.0f, 1.0f, 1.0f ),
        blur( 0.0f ){};
    Metal( float r, float g, float b, float blur = 0.0f ) :
        albedo( r, g, b ),
        blur( blur ) {}
    Metal( vector3 a, float blur = 0.0f ) :
        albedo( a ),
        blur( blur ) {}

    virtual bool scatter( const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered ) const;

    vector3 albedo;
    float   blur;
};

class Glass : virtual public IMaterial {
public:
    Glass() :
        refractionIndex( 1.0f ) {}
    Glass( float ri ) :
        refractionIndex( ri ) {}

    virtual bool scatter( const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered ) const;

    float refractionIndex;
};

} // namespace pk
