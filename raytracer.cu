#include "material.h"
#include "perf_timer.h"
#include "ray.h"
#include "raytracer.h"
#include "scene.h"
#include "sphere.h"
#include "vector_cuda.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <curand_kernel.h>

namespace pk
{


typedef struct _sphere {
    vector3    center;
    float      radius;
    material_t material;

    //__host__ __device__ _sphere() :
    //    center( 0, 0, 0 ), radius( 0.0f ) {}
} sphere_t;


typedef struct _RenderThreadContext {
    const Camera*   camera;
    const sphere_t* scene;
    uint32_t        sceneSize;
    curandState*    random;
    uint32_t*       framebuffer;
    uint32_t        rows;
    uint32_t        cols;
    uint32_t        num_aa_samples;
    uint32_t        max_ray_depth;
    bool            debug;

    _RenderThreadContext() :
        scene( nullptr ),
        camera( nullptr ),
        framebuffer( nullptr ),
        debug( false )
    {
    }
} RenderThreadContext;


static __global__ void _createCamera( Camera* pdCamera );
static __global__ void _renderInit( RenderThreadContext* pdContext );
static __global__ void _render( RenderThreadContext* pdContext );
static __device__ vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned max_depth, curandState* rand );
static __device__ vector3 _background( const ray& r );
static __device__ bool    _sphereHit( const sphere_t& sphere, const ray& r, float min, float max, hit_info* p_hit );
static __device__ bool    _sceneHit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit );


int renderSceneCUDA( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    // Add +1 to block dims in case image is not a multiple of blockSize
    //blockSize = 16;
    dim3 blocks( cols / blockSize + 1, rows / blockSize + 1 );
    dim3 threads( blockSize, blockSize );
    printf( "renderSceneCUDA(): blocks %d,%d,%d threads %d,%d\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y );

    // Allocate random number generator for each ray
    // This is HUGE but seems to be idiomatic in CUDA
    size_t       numPixels     = rows * cols;
    curandState* pdRandomState = nullptr;
    CHECK_CUDA( cudaMalloc( &pdRandomState, sizeof( curandState ) * numPixels ) );
    printf( "Allocated %zd device bytes for pdrandomState\n", sizeof( curandState ) * numPixels );

    // Create a copy of the Camera on the device [ gross hack because Camera is created in main() since before I refactored for CUDA ]
    Camera* pdCamera = nullptr;
    CHECK_CUDA( cudaMallocManaged( &pdCamera, sizeof( Camera ) * 2 ) );
    memcpy( &pdCamera[ 1 ], &camera, sizeof( camera ) );
    printf( "Allocated %zd device bytes for camera\n", sizeof( camera ) );
    _createCamera<<<1, 1>>>( pdCamera );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    // Copy the Scene to device
    // Flatten the Scene object to an array of sphere_t, which is what Scene should've been in the first place
    sphere_t* pdScene   = nullptr;
    size_t    sceneSize = sizeof( sphere_t ) * scene.objects.size();
    CHECK_CUDA( cudaMallocManaged( &pdScene, sceneSize ) );
    printf( "Allocated %zd device bytes / %zd objects\n", sceneSize, scene.objects.size() );

    sphere_t* p = pdScene;
    for ( IVisible* obj : scene.objects ) {
        Sphere*   s1 = dynamic_cast<Sphere*>( obj );
        sphere_t* s2 = (sphere_t*)p;
        s2->center   = s1->center;
        s2->radius   = s1->radius;

        // Deep copy material to the GPU
        s2->material = *( s1->material );

        p++;
    }
    printf( "Copied %zd objects to device\n", scene.objects.size() );


    // Allocate a render context to pass information to the GPU
    RenderThreadContext* pdContext = nullptr;
    CHECK_CUDA( cudaMallocManaged( &pdContext, sizeof( RenderThreadContext ) ) );
    printf( "Allocated %zd device bytes for context\n", sizeof( RenderThreadContext ) );
    pdContext->camera         = pdCamera;
    pdContext->scene          = pdScene;
    pdContext->sceneSize      = (uint32_t)scene.objects.size();
    pdContext->framebuffer    = framebuffer;
    pdContext->rows           = rows;
    pdContext->cols           = cols;
    pdContext->num_aa_samples = num_aa_samples;
    pdContext->max_ray_depth  = max_ray_depth;
    pdContext->debug          = debug;
    pdContext->random         = pdRandomState;

    // Render the scene
    _renderInit<<<blocks, threads>>>( pdContext );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    //_render<<<blocks, threads>>>( pdCamera, framebuffer, cols, rows, pdScene, (uint32_t)scene.objects.size(), pdRandomState );
    _render<<<blocks, threads>>>( pdContext );
    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );

    CHECK_CUDA( cudaFree( pdCamera ) );
    CHECK_CUDA( cudaFree( pdRandomState ) );
    CHECK_CUDA( cudaFree( pdScene ) );
    CHECK_CUDA( cudaFree( pdContext ) );

    printf( "renderSceneCUDA: %f s\n", t.ElapsedSeconds() );

    return 0;
}


static __global__ void _createCamera( Camera* pdCamera )
{
    // Gross hack: allocate a GPU camera by creating a local copy of the host camera's data
    Camera* prototype = &pdCamera[ 1 ];
    new ( pdCamera ) Camera( *prototype );
}


static __global__ void _renderInit( RenderThreadContext* ctx )
{
    unsigned x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned y = threadIdx.y + blockIdx.y * blockDim.y;

    if ( x >= ctx->cols || y >= ctx->rows )
        return;

    unsigned pixel_index = ( y * ctx->cols ) + x;

    // NOTE: tutorial says to pass a unique sequence number but this crashes curand_init() hard (and is insanely slow).
    // See: https://devtalk.nvidia.com/default/topic/1028057/curand_init-sequence-number-problem/
    //curand_init( 1984, pixel_index, 0, &pdRandomState[ pixel_index ] );
    curand_init( 1984 + pixel_index, 0, 0, &ctx->random[ pixel_index ] );
}

static __global__ void _render( RenderThreadContext* ctx )
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= ctx->cols || y >= ctx->rows )
        return;

    unsigned    p      = ( y * ctx->cols ) + ( x );
    curandState random = ctx->random[ p ];
    vector3     color( 0, 0, 0 );

    for ( uint32_t s = 0; s < ctx->num_aa_samples; s++ ) {
        float u = float( x + curand_uniform( &random ) ) / float( ctx->cols );
        float v = float( y + curand_uniform( &random ) ) / float( ctx->rows );
        ray   r = ctx->camera->getRay( u, v, &random );

        color += _color( r, ctx->scene, ctx->sceneSize, ctx->max_ray_depth, &random );
    }
    color /= float( ctx->num_aa_samples );

    // Apply 2.0 Gamma correction
    color = vector3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

    ctx->framebuffer[ p ] = ( ( uint32_t )( color.x * 255.99f ) << 24 ) | ( ( uint32_t )( color.y * 255.99f ) << 16 ) | ( ( uint32_t )( color.z * 255.99f ) << 8 );
}


// Non-recursive version
static __device__ vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned max_depth, curandState* rand )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( _sceneHit( scene, sceneSize, scattered, 0.001f, FLT_MAX, &hit ) ) {
#if defined( NORMAL_SHADE )
            vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            return 0.5f * vector3( normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f );
#elif defined( DIFFUSE_SHADE )
            vector3 target = hit.point + hit.normal + randomInUnitSphereCUDA( rand );
            scattered      = ray( hit.point, target - hit.point );
            color *= 0.5f;
#else
            if ( hit.material && materialScatter( hit.material, scattered, hit, &attenuation, &scattered, rand ) ) {
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


static __device__ vector3 _background( const ray& r )
{
    vector3 unitDirection = r.direction.normalized();
    float   t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vector3( 1.0f, 1.0f, 1.0f ) + t * vector3( 0.5f, 0.7f, 1.0f );
}


static __device__ bool _sphereHit( const sphere_t& sphere, const ray& r, float min, float max, hit_info* p_hit )
{
    assert( p_hit );

    vector3 oc = r.origin - sphere.center;
    float   a  = r.direction.dot( r.direction );
    float   b  = oc.dot( r.direction );
    float   c  = oc.dot( oc ) - ( sphere.radius * sphere.radius );

    float discriminant = b * b - a * c;

    if ( discriminant > 0 ) {
        float t = ( -b - sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - sphere.center ) / sphere.radius;
            p_hit->material = &sphere.material;
            return true;
        }

        t = ( -b + sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - sphere.center ) / sphere.radius;
            p_hit->material = &sphere.material;
            return true;
        }
    }

    return false;
}

static __device__ bool _sceneHit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit )
{
    bool     rval         = false;
    float    closestSoFar = max;
    hit_info hit;

    for ( int i = 0; i < sceneSize; i++ ) {
        const sphere_t& sphere = scene[ i ];

        hit_info tmp;
        if ( _sphereHit( sphere, r, min, closestSoFar, &tmp ) ) {
            rval         = true;
            closestSoFar = tmp.distance;
            hit          = tmp;
        }
    }

    *p_hit = hit;
    return rval;
}


//static __device__ vector3 _randomInUnitSphere( curandState* random )
//{
//    vector3      point;
//    unsigned int maxTries = 20;
//    do {
//        point = 2.0f * vector3( curand_uniform( random ), curand_uniform( random ), curand_uniform( random ) ) - vector3( 1, 1, 1 );
//    } while ( point.squared_length() >= 1.0f && maxTries-- );
//
//    return point;
//}

} // namespace pk
