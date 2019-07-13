#include "raytracer.h"

#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "scene.h"
#include "threadpool.h"
#include "vector_cuda.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

namespace pk
{

static vector3 _color_recursive( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth );
static vector3 _color( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth );
static vector3 _background( const ray& r );

typedef struct _RenderThreadContext {
    const Scene*           scene;
    const Camera*          camera;
    uint32_t*              frameBuffer;
    uint32_t               blockID;
    uint32_t               blockSize;
    uint32_t               xOffset;
    uint32_t               yOffset;
    uint32_t               rows;
    uint32_t               cols;
    uint32_t               num_aa_samples;
    uint32_t               max_ray_depth;
    std::atomic<uint32_t>* blockCount;
    uint32_t               totalBlocks;
    bool                   debug;
    bool                   recursive;

    _RenderThreadContext() :
        scene( nullptr ),
        camera( nullptr ),
        frameBuffer( nullptr ),
        blockSize( 0 ),
        xOffset( 0 ),
        yOffset( 0 ),
        debug( false ),
        recursive( false )
    {
    }
} RenderThreadContext;
static void _renderThread( uint32_t tid, const void* context );


int renderScene( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* frameBuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    // Spin up a pool of render threads, one per block
    // Allocate width+1 and height+1 blocks to handle case where image is not an even multiple of block size
    uint32_t      widthBlocks  = uint32_t( float( cols / blockSize ) ) + 1;
    uint32_t      heightBlocks = uint32_t( float( rows / blockSize ) ) + 1;
    uint32_t      numBlocks    = heightBlocks * widthBlocks;
    thread_pool_t tp           = threadPoolInit( numThreads );

    printf( "Render %d x %d: blockSize %d x %d, %d blocks debug %d threads [%d:%d]\n",
        cols, rows, blockSize, blockSize, numBlocks, debug, tp, numThreads );

    RenderThreadContext* contexts = new RenderThreadContext[ numBlocks ];

    std::atomic<uint32_t> blockCount = 0;
    uint32_t              blockID    = 0;
    uint32_t              yOffset    = 0;
    for ( uint32_t y = 0; y < heightBlocks; y++ ) {
        uint32_t xOffset = 0;
        for ( uint32_t x = 0; x < widthBlocks; x++ ) {
            RenderThreadContext* ctx = &contexts[ blockID ];
            ctx->scene               = &scene;
            ctx->camera              = &camera;
            ctx->frameBuffer         = frameBuffer;
            ctx->blockID             = blockID;
            ctx->blockSize           = blockSize;
            ctx->xOffset             = xOffset;
            ctx->yOffset             = yOffset;
            ctx->rows                = rows;
            ctx->cols                = cols;
            ctx->num_aa_samples      = num_aa_samples;
            ctx->max_ray_depth       = max_ray_depth;
            ctx->blockCount          = &blockCount;
            ctx->totalBlocks         = numBlocks;
            ctx->debug               = debug;
            ctx->recursive           = recursive;

            threadPoolSubmitJob( tp, _renderThread, ctx );

            //printf( "Submit block %d of %d\n", blockID, numBlocks );

            blockID++;
            xOffset += blockSize;
        }
        yOffset += blockSize;
    }

    // Wait for threads to complete
    while ( blockCount != numBlocks ) {
        delay( 1000 );
    }

    threadPoolDeinit( tp );
    delete[] contexts;

    printf( "renderScene: %f s\n", t.ElapsedSeconds() );

    return 0;
}

static void _renderThread( uint32_t tid, const void* context )
{
    UNUSED( tid );

    const RenderThreadContext* ctx = (const RenderThreadContext*)context;

    //printf( "start %dx%d block %d of %d AA:%d MD:%d R:%d %d x %d x %d\n",
    //    ctx->cols, ctx->rows,
    //    ctx->blockID, ctx->totalBlocks, ctx->num_aa_samples, ctx->max_ray_depth, ctx->recursive,
    //    ctx->xOffset, ctx->yOffset, ctx->blockSize
    //    );

    for ( uint32_t y = ctx->yOffset; y < ctx->yOffset + ctx->blockSize; y++ ) {
        // Don't render out of bounds (in case where image is not an even multiple of block size)
        if ( y >= ctx->rows )
            break;

        for ( uint32_t x = ctx->xOffset; x < ctx->xOffset + ctx->blockSize; x++ ) {
            // Don't render out of bounds (in case where image is not an even multiple of block size)
            if ( x >= ctx->cols )
                break;

            // TEST
            if ( ctx->debug && ( y == ctx->yOffset || y == ctx->yOffset + ctx->blockSize - 1 || x == ctx->xOffset || x == ctx->xOffset + ctx->blockSize - 1 ) ) {
                ctx->frameBuffer[ y * ctx->cols + x ] = 0xFF000000;
                continue;
            }

            // Sample each pixel in image space, with anti-aliasing

            vector3 color( 0, 0, 0 );
            for ( uint32_t s = 0; s < ctx->num_aa_samples; s++ ) {
                float u = float( x + random() ) / float( ctx->cols );
                float v = float( y + random() ) / float( ctx->rows );
                ray   r = ctx->camera->getRay( u, v, nullptr );

                if ( ctx->recursive ) {
                    color += _color_recursive( r, ctx->scene, 0, ctx->max_ray_depth );
                } else {
                    color += _color( r, ctx->scene, 0, ctx->max_ray_depth );
                }
            }
            color /= float( ctx->num_aa_samples );

            // Apply 2.0 Gamma correction
            color = vector3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

            uint8_t  _r  = ( uint8_t )( 255.99 * color.x );
            uint8_t  _g  = ( uint8_t )( 255.99 * color.y );
            uint8_t  _b  = ( uint8_t )( 255.99 * color.z );
            uint32_t rgb = ( (uint32_t)_r << 24 ) | ( (uint32_t)_g << 16 ) | ( (uint32_t)_b << 8 );

            ctx->frameBuffer[ y * ctx->cols + x ] = rgb;
        }
    }

    //printf( "block %d of %d DONE\n", ctx->blockID, ctx->totalBlocks );

    // Notify main thread that we have completed the work
    if ( ctx->blockID == ctx->totalBlocks - 1 ) {
        std::atomic<uint32_t>* blockCount = ctx->blockCount;
        blockCount->exchange( ctx->totalBlocks );
    }

    return;
}

// Recursively trace each ray through objects/materials
static vector3 _color_recursive( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth )
{
    hit_info hit;

    if ( scene->hit( r, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
        vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
        return 0.5f * vector3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
        if ( depth < max_depth ) {
            vector3 target = hit.point + hit.normal + randomInUnitSphere();
            return 0.5f * _color_recursive( ray( hit.point, target - hit.point ), scene, depth + 1, max_depth );
        } else {
            return vector3( 0, 0, 0 );
        }
#else
        ray     scattered;
        vector3 attenuation;
        if ( depth < max_depth && hit.material && materialScatter( hit.material, r, hit, &attenuation, &scattered, nullptr ) ) {
            return attenuation * _color( scattered, scene, depth + 1, max_depth );
        } else {
            return vector3( 0, 0, 0 );
        }
#endif
    }

    return _background( r );
}

// Non-recursive version
static vector3 _color( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( scene->hit( scattered, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
            vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            return 0.5f * vector3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
            vector3 target = hit.point + hit.normal + randomInUnitSphere();
            scattered      = ray( hit.point, target - hit.point );
            color *= 0.5f;
#else
            if ( hit.material && materialScatter( hit.material, scattered, hit, &attenuation, &scattered, nullptr ) ) {
                color *= attenuation;
            } else {
                break;
            }
#endif
        } else {
            color *= _background( scattered );
            break;
        }
    }

    return color;
}


static vector3 _background( const ray& r )
{
    vector3 unitDirection = r.direction.normalized();
    float   t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vector3( 1.0f, 1.0f, 1.0f ) + t * vector3( 0.5f, 0.7f, 1.0f );
}

} // namespace pk
