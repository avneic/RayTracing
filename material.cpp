#include "material.h"
#include "utils.h"


namespace pk
{

static vec3  _reflect( const vec3& v, const vec3& normal );
static bool  _refract( const vec3& v, const vec3& normal, float ni_over_nt, vec3* refracted );
static float _schlick( float cosine, float refractionIndex );

static vec3 _reflect( const vec3& v, const vec3& normal )
{
    return v - 2 * v.dot( normal ) * normal;
}

static bool _refract( const vec3& v, const vec3& normal, float ni_over_nt, vec3* refracted )
{
    vec3  _v           = v.normalized();
    float dt           = _v.dot( normal );
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * ( 1.0f - dt * dt );
    if ( discriminant > 0 ) {
        *refracted = ni_over_nt * ( _v - normal * dt ) - normal * sqrt( discriminant );
        return true;
    } else {
        return false;
    }
}

static float _schlick(float cosine, float refractionIndex)
{
    float r0 = (1.0f - refractionIndex) / (1.0f + refractionIndex);
    r0 = r0 * r0;

    return r0 + (1.0f - r0) * pow((1.0f - cosine), 5);
}


bool Diffuse::scatter( const ray& r, const hit_info& hit, vec3* attenuation, ray* scattered ) const
{
    vec3 target  = hit.point + hit.normal + randomInUnitSphere();
    *scattered   = ray( hit.point, target - hit.point );
    *attenuation = albedo;

    return true;
}

bool Metal::scatter( const ray& r, const hit_info& hit, vec3* attenuation, ray* scattered ) const
{
    vec3 reflected = _reflect( r.direction.normalized(), hit.normal );
    *scattered     = ray( hit.point, reflected + ( blur * randomInUnitSphere() ) );
    *attenuation   = albedo;

    return ( scattered->direction.dot( hit.normal ) > 0 );
}

bool Glass::scatter( const ray& r, const hit_info& hit, vec3* attenuation, ray* scattered ) const
{
    vec3  outwardNormal;
    vec3  reflected = _reflect( r.direction, hit.normal );
    float niOverNt;
    vec3  refracted;
    float probability;
    float cosine;

    *attenuation = vec3( 1.0, 1.0, 1.0 ); // no color shift; "white" glass

    if ( r.direction.dot( hit.normal ) > 0 ) {
        outwardNormal = -hit.normal;
        niOverNt      = refractionIndex;
        cosine        = refractionIndex * r.direction.dot( hit.normal ) / r.direction.length();
    } else {
        outwardNormal = hit.normal;
        niOverNt      = 1.0f / refractionIndex;
        cosine        = -r.direction.dot( hit.normal ) / r.direction.length();
    }

    if ( _refract( r.direction, outwardNormal, niOverNt, &refracted ) ) {
        probability = _schlick( cosine, refractionIndex );
    } else {
        probability = 1.0f;
    }

    if ( random() < probability ) {
        *scattered = ray( hit.point, reflected );
    } else {
        *scattered = ray( hit.point, refracted );
    }

    return true;
}

} // namespace pk
