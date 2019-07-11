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

namespace pk
{


#define NORMAL_SHADE
//#define DIFFUSE_SHADE

typedef struct _sphere {
    vector3    center;
    float      radius;
    uint32_t   material;
    __device__ __host__ _sphere() :
        center( 0, 0, 0 ), radius( 0.0f ), material( 0 ) {}
} sphere_t;


static __global__ void render( uint32_t* framebuffer, uint32_t max_x, uint32_t max_y, const sphere_t* scene, uint32_t sceneSize );
static __device__ vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth );
static __device__ vector3 _background( const ray& r );
static __device__ bool    _sphere_hit( const vector3& center, float radius, const ray& r, float min, float max, hit_info* p_hit );
static __device__ bool    _scene_hit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit );


int renderSceneCUDA( const Scene& scene, const Camera& camera, unsigned rows, unsigned cols, uint32_t* framebuffer, unsigned num_aa_samples, unsigned max_ray_depth, unsigned numThreads, unsigned blockSize, bool debug, bool recursive )
{
    PerfTimer t;

    // Allocate device-side buffer to hold the scene
    sphere_t* pdScene   = nullptr;
    size_t    sceneSize = sizeof( sphere_t ) * scene.objects.size();
    CHECK_CUDA( cudaMallocManaged( &pdScene, sceneSize ) );
    printf( "Allocated %zd device bytes / %zd objects\n", sceneSize, scene.objects.size() );

    // Copy the Scene to device
    // Flatten the Scene object to an array of sphere_t, which is what Scene should've been in the first place
    sphere_t* p = pdScene;
    for ( IVisible* obj : scene.objects ) {
        Sphere*   s1 = dynamic_cast<Sphere*>( obj );
        sphere_t* s2 = (sphere_t*)p;
        s2->center   = s1->center;
        s2->radius   = s1->radius;
        p++;
    }
    printf( "Copied %zd objects to device\n", scene.objects.size() );

    blockSize = 16;

    // Add +1 to blockSize in case image is not a multiple of blockSize
    dim3 blocks( cols / blockSize + 1, rows / blockSize + 1 );
    dim3 threads( blockSize, blockSize );

    printf( "renderSceneCUDA(): blocks %d,%d,%d threads %d,%d\n", blocks.x, blocks.y, blocks.z, threads.x, threads.y );

    //render<<<blocks, threads>>>( framebuffer, cols, rows );
    //render<<<blocks, threads>>>( framebuffer, cols, rows, camera.origin, camera.leftCorner, camera.horizontal, camera.vertical );
    render<<<blocks, threads>>>( framebuffer, cols, rows, pdScene, (uint32_t)scene.objects.size() );

    CHECK_CUDA( cudaGetLastError() );
    CHECK_CUDA( cudaDeviceSynchronize() );
    CHECK_CUDA( cudaFree( pdScene ) );

    printf( "renderSceneCUDA: %f ms\n", t.ElapsedMilliseconds() );

    return 0;
}

//__global__ void render( uint32_t* framebuffer, uint32_t max_x, uint32_t max_y, const vector3& origin, const vector3& leftCorner, const vector3& horizontal, const vector3& vertical )
__global__ void render( uint32_t* framebuffer, uint32_t max_x, uint32_t max_y, const sphere_t* scene, uint32_t sceneSize )
{
    unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( x >= max_x || y >= max_y )
        return;

    float u = (float)x / (float)max_x;
    float v = (float)y / (float)max_y;

    vector3 origin( 0, 0, 0 );
    vector3 leftCorner( -2.0f, 1.0f, -1.0f );
    vector3 horizontal( 4.0f, 0.0f, 0.0f );
    vector3 vertical( 0.0f, -2.0f, 0.0f );

    ray     r( origin, leftCorner + u * horizontal + v * vertical );
    vector3 rgb = _color( r, scene, sceneSize, 0, 50 );

    unsigned p       = ( y * max_x ) + ( x );
    framebuffer[ p ] = ( ( uint32_t )( rgb.x * 255.99f ) << 24 ) | ( ( uint32_t )( rgb.y * 255.99f ) << 16 ) | ( ( uint32_t )( rgb.z * 255.99f ) << 8 );

    //printf("%d,%d: (%f, %f, %f) [%f %f %f] 0x%x\n",
    //    x, y,
    //    r.direction.x, r.direction.y, r.direction.z,
    //    rgb.x, rgb.y, rgb.z,
    //    framebuffer[p]);
}


//static __device__ vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth )
//{
//    float    min = 0.001;
//    float    max = FLT_MAX;
//    hit_info hit;
//
//    vector3 color = _background(r);
//
//    for ( int i = 0; i < sceneSize; i++ ) {
//        const sphere_t* sphere = &scene[ i ];
//
//        if (_sphere_hit( sphere->center, sphere->radius, r, min, max, &hit)) {
//            color = vector3(1, 0, 0);
//        }
//    }
//
//    return color;
//}


// Non-recursive version
static __device__ vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned depth, unsigned max_depth )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( _scene_hit( scene, sceneSize, r, 0.001f, FLT_MAX, &hit ) ) {
#if defined( NORMAL_SHADE )
            vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            return 0.5f * vector3( normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f );
#elif defined( DIFFUSE_SHADE )
            if ( depth < max_depth ) {
                vec3 target = hit.point + hit.normal + randomInUnitSphere();
                return 0.5f * _color( ray( hit.point, target - hit.point ), scene, depth + 1 );
            } else {
                return vec3( 0, 0, 0 );
            }
#else
            //if (hit.material && hit.material->scatter(scattered, hit, &attenuation, &scattered)) {
            //    color *= attenuation;
            //}
            //else {
            //    break;
            //}

            color = vector3( 1, 0, 0 );
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


static __device__ bool _sphere_hit( const vector3& center, float radius, const ray& r, float min, float max, hit_info* p_hit )
{
    assert( p_hit );

    vector3 oc = r.origin - center;
    float   a  = r.direction.dot( r.direction );
    float   b  = oc.dot( r.direction );
    float   c  = oc.dot( oc ) - ( radius * radius );

    float discriminant = b * b - a * c;

    if ( discriminant > 0 ) {
        float t = ( -b - sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - center ) / radius;
            //p_hit->material = material;
            return true;
        }

        t = ( -b + sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - center ) / radius;
            //p_hit->material = material;
            return true;
        }
    }

    return false;
}

static __device__ bool _scene_hit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit )
{
    bool     rval         = false;
    float    closestSoFar = max;
    hit_info hit;

    for ( int i = 0; i < sceneSize; i++ ) {
        const sphere_t* sphere = &scene[ i ];

        hit_info tmp;
        if ( _sphere_hit( sphere->center, sphere->radius, r, min, closestSoFar, &tmp ) ) {
            rval         = true;
            closestSoFar = tmp.distance;
            hit          = tmp;
        }
    }

    *p_hit = hit;
    return rval;
}

} // namespace pk
