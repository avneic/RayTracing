#include "material.h"
#include <random>

namespace pk {

static std::random_device rd;
static std::mt19937 gen(rd());
static std::uniform_real_distribution<float> random(0.0f, 1.0f);

static vec3 _reflect( const vec3& v, const vec3& normal );
static vec3 _randomInUnitSphere();


static vec3 _reflect(const vec3& v, const vec3& normal)
{
    return v - 2 * dot(v, normal) * normal;
}

static vec3 _randomInUnitSphere()
{
    vec3 point;
    do {
        point = 2.0f * vec3(random(gen), random(gen), random(gen)) - vec3(1, 1, 1);
    } while (point.squared_length() >= 1.0);

    return point;
}

bool Diffuse::scatter(const ray& r, const hit_info& hit, vec3* attenuation, ray* scattered) const
{
    vec3 target = hit.point + hit.normal + _randomInUnitSphere();
    *scattered = ray(hit.point, target - hit.point);
    *attenuation = albedo;

    return true;
}

bool Metal::scatter(const ray& r, const hit_info& hit, vec3* attenuation, ray* scattered) const
{
    vec3 reflected = _reflect(r.direction.normalized(), hit.normal);
    *scattered = ray(hit.point, reflected + (blur * _randomInUnitSphere()));
    *attenuation = albedo;

    return (dot(scattered->direction, hit.normal) > 0);
}

}
