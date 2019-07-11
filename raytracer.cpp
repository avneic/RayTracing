#include "raytracer.h"

#include "material.h"
#include "ray.h"
#include "scene.h"
#include "threadpool.h"
#include "vector.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

namespace pk
{

static vec3 _color_recursive( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth );
static vec3 _color( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth );
static vec3 _background( const ray& r );

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
    // Spin up a pool of render threads, one per tile
    uint32_t      numBlocks = ( rows / blockSize ) * ( rows / blockSize );
    thread_pool_t tp        = threadPoolInit( numThreads );

    printf( "Render: blockSize %d x %d, %d blocks debug %d threads [%d:%d]\n", blockSize, blockSize, numBlocks, debug, tp, numThreads );

    RenderThreadContext* contexts = new RenderThreadContext[ numBlocks ];

    std::atomic<uint32_t> blockCount = 0;
    uint32_t              blockID    = 0;
    uint32_t              yOffset    = 0;
    for ( uint32_t y = 0; y < rows / blockSize; y++ ) {
        uint32_t xOffset = 0;
        for ( uint32_t x = 0; x < cols / blockSize; x++ ) {
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

    return 0;
}

static void _renderThread( uint32_t tid, const void* context )
{
    UNUSED( tid );

    const RenderThreadContext* ctx = (const RenderThreadContext*)context;

    for ( uint32_t y = ctx->yOffset; y < ctx->yOffset + ctx->blockSize; y++ ) {
        for ( uint32_t x = ctx->xOffset; x < ctx->xOffset + ctx->blockSize; x++ ) {

            // TEST
            if ( ctx->debug && ( y == ctx->yOffset || y == ctx->yOffset + ctx->blockSize - 1 || x == ctx->xOffset || x == ctx->xOffset + ctx->blockSize - 1 ) ) {
                ctx->frameBuffer[ y * ctx->cols + x ] = 0xFF000000;
                continue;
            }

            // Sample each pixel in image space, with anti-aliasing

            vec3 color( 0, 0, 0 );
            for ( uint32_t s = 0; s < ctx->num_aa_samples; s++ ) {
                float u = float( x + random() ) / float( ctx->cols );
                float v = float( y + random() ) / float( ctx->rows );
                ray   r = ctx->camera->getRay( u, v );

                if ( ctx->recursive ) {
                    color += _color_recursive( r, ctx->scene, 0, ctx->max_ray_depth );
                } else {
                    color += _color( r, ctx->scene, 0, ctx->max_ray_depth );
                }
            }
            color /= float( ctx->num_aa_samples );

            // Apply 2.0 Gamma correction
            color = vec3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

            uint8_t  _r  = ( uint8_t )( 255.99 * color.x );
            uint8_t  _g  = ( uint8_t )( 255.99 * color.y );
            uint8_t  _b  = ( uint8_t )( 255.99 * color.z );
            uint32_t rgb = ( (uint32_t)_r << 24 ) | ( (uint32_t)_g << 16 ) | ( (uint32_t)_b << 8 );

            ctx->frameBuffer[ y * ctx->cols + x ] = rgb;
        }
    }

    // Notify caller that we have completed one work item
    if ( ctx->blockID == ctx->totalBlocks - 1 ) {
        std::atomic<uint32_t>* blockCount = ctx->blockCount;
        //uint32_t               count = blockCount->fetch_add(1);
        blockCount->exchange( ctx->totalBlocks );
    }

    return;
}

// Recursively trace each ray through objects/materials
static vec3 _color_recursive( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth )
{
    hit_info hit;

    if ( scene->hit( r, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
        vec3 normal = ( r.point( hit.distance ) - vec3( 0, 0, -1 ) ).normalized();
        return 0.5f * vec3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
        if ( depth < max_depth ) {
            vec3 target = hit.point + hit.normal + randomInUnitSphere();
            return 0.5f * _color_recursive( ray( hit.point, target - hit.point ), scene, depth + 1 );
        } else {
            return vec3( 0, 0, 0 );
        }
#else
        ray  scattered;
        vec3 attenuation;
        if ( depth < max_depth && hit.material && hit.material->scatter( r, hit, &attenuation, &scattered ) ) {
            return attenuation * _color( scattered, scene, depth + 1, max_depth );
        } else {
            return vec3( 0, 0, 0 );
        }
#endif
    }

    return _background( r );
}

// Non-recursive version
static vec3 _color( const ray& r, const Scene* scene, unsigned depth, unsigned max_depth )
{
    hit_info hit;
    vec3     attenuation;
    ray      scattered = r;
    vec3     color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( scene->hit( scattered, 0.001f, ( std::numeric_limits<float>::max )(), &hit ) ) {
#if defined( NORMAL_SHADE )
            vec3 normal = ( r.point( hit.distance ) - vec3( 0, 0, -1 ) ).normalized();
            return 0.5f * vec3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
            if ( depth < max_depth ) {
                vec3 target = hit.point + hit.normal + randomInUnitSphere();
                return 0.5f * _color( ray( hit.point, target - hit.point ), scene, depth + 1 );
            } else {
                return vec3( 0, 0, 0 );
            }
#else
            if ( hit.material && hit.material->scatter( scattered, hit, &attenuation, &scattered ) ) {
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

static vec3 _background( const ray& r )
{
    vec3  unitDirection = r.direction.normalized();
    float t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * vec3( 0.5f, 0.7f, 1.0f );
}

} // namespace pk