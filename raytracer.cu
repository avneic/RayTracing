#include "raytracer.h"

#include "material.h"
#include "ray.h"
#include "scene.h"
#include "vector.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace pk
{

__global__ void render( float *framebuffer, uint32_t max_x, uint32_t max_y )
{
    unsigned  x   = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned  y   = threadIdx.y + blockIdx.y + blockDim.y;
    const int bpp = 3;

    if ( x >= max_x || y >= max_y )
        return;

    unsigned p = ( y * max_x * bpp ) + ( x * bpp );

    // Render a gradient
    framebuffer[ p + 0 ] = float( x ) / max_x;
    framebuffer[ p + 1 ] = float( y ) / max_y;
    framebuffer[ p + 0 ] = 0.2f;
}

int renderSceneCUDA( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{

    return 0;
}

} // namespace pk
