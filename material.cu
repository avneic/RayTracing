#include "material.h"
#include "utils.h"

#include <cassert>

namespace pk
{

__host__ __device__ static vector3 _reflect( const vector3& v, const vector3& normal );
__host__ __device__ static bool    _refract( const vector3& v, const vector3& normal, float ni_over_nt, vector3* refracted );
__host__ __device__ static float   _schlick( float cosine, float refractionIndex );

__host__ __device__ static bool _diffuseScatter( const material_t& d, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered );
__host__ __device__ static bool _metalScatter( const material_t& m, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered );
__host__ __device__ static bool _glassScatter( const material_t& g, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered );

__host__ __device__ bool materialScatter( const material_t& m, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered )
{
    bool rval = false;

    switch ( m.type ) {
        case MATERIAL_DIFFUSE:
            rval = _diffuseScatter( m, r, hit, attenuation, scattered );
            break;

        case MATERIAL_METAL:
            rval = _metalScatter( m, r, hit, attenuation, scattered );
            break;

        case MATERIAL_GLASS:
            rval = _glassScatter( m, r, hit, attenuation, scattered );
            break;

        default:
            printf("unknown material %d\n", m.type);
            assert( 0 );
    }

    return rval;
}

//
// Material implementations
//

#pragma nv_exec_check_disable
__host__ __device__ static bool _diffuseScatter( const material_t& d, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered )
{
    vector3 target    = hit.point + hit.normal + randomInUnitSphere();
    *scattered        = ray( hit.point, target - hit.point );
    *attenuation      = d.albedo;

    return true;
}


#pragma nv_exec_check_disable
__host__ __device__ static bool _metalScatter( const material_t& m, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered )
{
    vector3 reflected = _reflect( r.direction.normalized(), hit.normal );
    *scattered        = ray( hit.point, reflected + ( m.blur * randomInUnitSphere() ) );
    *attenuation      = m.albedo;

    return ( scattered->direction.dot( hit.normal ) > 0 );
}


#pragma nv_exec_check_disable
__host__ __device__ static bool _glassScatter( const material_t& g, const ray& r, const hit_info& hit, vector3* attenuation, ray* scattered )
{
    vector3 outwardNormal;
    vector3 reflected = _reflect( r.direction, hit.normal );
    float   niOverNt;
    vector3 refracted;
    float   probability;
    float   cosine;

    *attenuation = vector3( 1.0, 1.0, 1.0 ); // no color shift; "white" glass

    if ( r.direction.dot( hit.normal ) > 0 ) {
        outwardNormal = -hit.normal;
        niOverNt      = g.refractionIndex;
        cosine        = g.refractionIndex * r.direction.dot( hit.normal ) / r.direction.length();
    } else {
        outwardNormal = hit.normal;
        niOverNt      = 1.0f / g.refractionIndex;
        cosine        = -r.direction.dot( hit.normal ) / r.direction.length();
    }

    if ( _refract( r.direction, outwardNormal, niOverNt, &refracted ) ) {
        probability = _schlick( cosine, g.refractionIndex );
    } else {
        probability = 1.0f;
    }

    float p = random();

    if ( p < probability ) {
        *scattered = ray( hit.point, reflected );
    } else {
        *scattered = ray( hit.point, refracted );
    }

    return true;
}


//
// Helper functions
//

__host__ __device__ static vector3 _reflect( const vector3& v, const vector3& normal )
{
    return v - 2 * v.dot( normal ) * normal;
}


__host__ __device__ static bool _refract( const vector3& v, const vector3& normal, float ni_over_nt, vector3* refracted )
{
    vector3 _v           = v.normalized();
    float   dt           = _v.dot( normal );
    float   discriminant = 1.0f - ni_over_nt * ni_over_nt * ( 1.0f - dt * dt );
    if ( discriminant > 0 ) {
        *refracted = ni_over_nt * ( _v - normal * dt ) - normal * sqrt( discriminant );
        return true;
    } else {
        return false;
    }
}


__host__ __device__ static float _schlick( float cosine, float refractionIndex )
{
    float r0 = ( 1.0f - refractionIndex ) / ( 1.0f + refractionIndex );
    r0       = r0 * r0;

    return r0 + ( 1.0f - r0 ) * pow( ( 1.0f - cosine ), 5 );
}


} // namespace pk
