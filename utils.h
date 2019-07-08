#pragma once

#include "vec3.h"

namespace pk
{
#define M_PI 3.14159265358979323846f
#define RADIANS( x ) ( (x)*M_PI / 180.0f )

#define ARRAY_SIZE(x) (sizeof(x) / sizeof((x)[0]))

bool delay(size_t ms);
float random();
vec3  randomInUnitSphere();
vec3  randomOnUnitDisk();

} // namespace pk
