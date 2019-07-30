#pragma once

#include "camera.h"
#include "material.h"
#include "sphere.h"

#include <atomic>
#include <stdint.h>
#include <string.h>

namespace pk
{

//#define DIFFUSE_SHADE
//#define NORMAL_SHADE

int renderScene( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples = 4, unsigned max_ray_depth = 50, unsigned numThreads = 1, unsigned blockSize = 64, bool debug = false, bool recursive = true );
int renderSceneCUDA( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples = 4, unsigned max_ray_depth = 50, unsigned numThreads = 1, unsigned blockSize = 64, bool debug = false, bool recursive = true );
int renderSceneISPC( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples = 4, unsigned max_ray_depth = 50, unsigned numThreads = 1, unsigned blockSize = 64, bool debug = false, bool recursive = true );

} // namespace pk
