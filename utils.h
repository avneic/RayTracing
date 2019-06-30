#pragma once

#include "vec3.h"

namespace pk
{
#define M_PI 3.14159265358979323846f
#define RADIANS( x ) ( (x)*M_PI / 180.0f )

float random();
vec3  randomInUnitSphere();
vec3  randomOnUnitDisk();

} // namespace pk
