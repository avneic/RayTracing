#include "raytracer.h"

#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "scene.h"
#include "vector_cuda.h"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

namespace pk
{

__global__ void render( uint32_t *framebuffer, uint32_t max_x, uint32_t max_y )
{
    unsigned  x   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned  y   = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= max_x || y >= max_y )
        return;

    uint8_t r = uint8_t( (float( x ) / max_x) * 255.0f);
    uint8_t g = uint8_t( (float( y ) / max_y) * 255.0f);
    uint8_t b = uint8_t((0.2f * 255.0f));
    uint32_t rgb = ( (uint32_t)r << 24 ) | ( (uint32_t)g << 16 ) | ( (uint32_t)b << 8 );

    unsigned p = (y * max_x) + ( x );
    framebuffer[p] = rgb;
}

int renderSceneCUDA( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    blockSize = 16;

    // Add +1 to blockSize in case image is not a multiple of blockSize
    dim3 blocks(cols / blockSize + 1, rows / blockSize + 1);
    dim3 threads(blockSize, blockSize);

    printf("renderSceneCUDA(): blocks %d,%d,%d threads %d,%d\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y);

    render<<<blocks, threads >>>(frameBuffer, cols, rows);
    //render<<<(rows*cols), 1>>>(frameBuffer, cols, rows);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    printf("renderSceneCUDA: %f ms\n", t.ElapsedMilliseconds());

    return 0;
}

} // namespace pk
