#include "utils.h"
#include <random>

namespace pk
{
static std::random_device                    _rd;
static std::mt19937                          _gen( _rd() );
static std::uniform_real_distribution<float> _random( 0.0f, 1.0f );

float random()
{
    return _random( _gen );
}

vec3 randomInUnitSphere()
{
    vec3 point;
    do {
        point = 2.0f * vec3( random(), random(), random() ) - vec3( 1, 1, 1 );
    } while ( point.squared_length() >= 1.0f );

    return point;
}

vec3 randomOnUnitDisk()
{
    vec3 point;
    do {
        point = 2.0f * vec3( random(), random(), 0.0f ) - vec3( 1.0f, 1.0f, 0.0f );
    } while (point.dot(point) >= 1.0f);

    return point;
}

} // namespace pk
