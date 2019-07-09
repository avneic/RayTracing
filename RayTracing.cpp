// Ray Tracing In One Weekend
//

#include "argsparser.h"
#include "camera.h"
#include "material.h"
#include "msgqueue.h"
#include "ray.h"
#include "sphere.h"
#include "threadpool.h"
#include "utils.h"
#include "vec3.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

using namespace pk;

//#define DIFFUSE_SHADE
//#define NORMAL_SHADE

//const unsigned int COLS = 1200;
//const unsigned int ROWS = 800;
const unsigned int COLS = 1280;
const unsigned int ROWS = 1280;

// Anti-aliasing
const unsigned int NUM_AA_SAMPLES = 10;

// Num bounces per ray
const unsigned int MAX_RAY_DEPTH = 50;

static int    _renderScene( const std::string& filename, const Scene& scene, const Camera& camera, uint32_t* frameBuffer, unsigned blockSize = 64, bool debug = false );
static vec3   _color( const ray& r, const Scene* scene, float depth );
static vec3   _background( const ray& r );
static Scene* _randomScene();

typedef struct _RenderThreadContext {
    const Scene*           scene;
    const Camera*          camera;
    uint32_t*              frameBuffer;
    uint32_t               blockID;
    uint32_t               blockSize;
    uint32_t               xOffset;
    uint32_t               yOffset;
    std::atomic<uint32_t>* blockCount;
    bool                   debug;

    _RenderThreadContext() :
        scene( nullptr ),
        camera( nullptr ),
        frameBuffer( nullptr ),
        blockSize( 0 ),
        xOffset( 0 ),
        yOffset( 0 ),
        debug( false )
    {
    }
} RenderThreadContext;
static void _renderThread( const void* context );

//
// Simple Ray Tracer
//

int main( int argc, char** argv )
{
    ArgsParser args( argc, argv );

    Scene* scene = _randomScene();

    vec3  origin( 13, 2, 3 );
    vec3  lookat( 0, 0, 0 );
    vec3  up( 0, 1, 0 );
    float vfov   = 20;
    float aspect = float( COLS ) / float( ROWS );

    //float focusDistance = (origin - lookat).length();
    float focusDistance = 10.0f;
    float aperture      = 0.1f;

    Camera camera( vfov, aspect, aperture, focusDistance, origin, up, lookat );

    // Render to RAM (multi-threaded) then flush to disk
    uint32_t* frameBuffer = nullptr;
    frameBuffer           = new uint32_t[ ROWS * COLS ];
    memset( frameBuffer, 0xCC, sizeof( uint32_t ) * ROWS * COLS );

    int numThreads = std::thread::hardware_concurrency();
    if ( args.cmdOptionExists( "-t" ) ) {
        const std::string& arg = args.getCmdOption( "-t" );
        numThreads             = std::stoi( arg );
    }

    int blockSize = 64;
    if ( args.cmdOptionExists( "-b" ) ) {
        const std::string& arg = args.getCmdOption( "-b" );
        blockSize             = std::stoi( arg );
    }

    bool debug = false;
    if ( args.cmdOptionExists( "-d" ) ) {
        debug = true;
    }

    threadPoolInit( numThreads );

    _renderScene( "foo.ppm", *scene, camera, frameBuffer, blockSize, debug );

    delete scene;
    delete[] frameBuffer;

    return 0;
}

static int _renderScene( const std::string& filename, const Scene& scene, const Camera& camera, uint32_t* frameBuffer, unsigned blockSize, bool debug )
{
    FILE*   file = nullptr;
    errno_t err  = fopen_s( &file, filename.c_str(), "w" );
    if ( !file || err != 0 ) {
        printf( "Error: failed to open [%s] for writing errno %d.\n", filename.c_str(), err );
        return -1;
    }

    // Print PPM header
    fprintf( file, "P3\n" );
    fprintf( file, "%d %d\n", COLS, ROWS );
    fprintf( file, "255\n" );

    // Spin up a pool of render threads, one per tile
    uint32_t numThreads = std::thread::hardware_concurrency();
    uint32_t numBlocks  = ( ROWS / blockSize ) * ( ROWS / blockSize );

    printf( "Render: blockSize %d x %d, %d blocks debug %d\n", blockSize, blockSize, numBlocks, debug );

    RenderThreadContext* contexts = new RenderThreadContext[ numBlocks ];

    std::atomic<uint32_t> blockCount = 0;
    uint32_t              blockID    = 0;
    uint32_t              yOffset    = 0;
    for ( uint32_t y = 0; y < ROWS / blockSize; y++ ) {
        uint32_t xOffset = 0;
        for ( uint32_t x = 0; x < COLS / blockSize; x++ ) {
            RenderThreadContext* ctx = &contexts[ blockID ];
            ctx->scene               = &scene;
            ctx->camera              = &camera;
            ctx->frameBuffer         = frameBuffer;
            ctx->blockID             = blockID;
            ctx->blockSize           = blockSize;
            ctx->xOffset             = xOffset;
            ctx->yOffset             = yOffset;
            ctx->blockCount          = &blockCount;
            ctx->debug               = debug;

            threadPoolSubmitJob( _renderThread, ctx );

            blockID++;
            xOffset += blockSize;
        }
        yOffset += blockSize;
    }

    // Wait for threads to complete
    while ( blockCount != numBlocks ) {
        delay( 1000 );
    }

    threadPoolDeinit();
    delete[] contexts;

    // Write to disk
    for ( uint32_t y = 0; y < ROWS; y++ ) {
        for ( uint32_t x = 0; x < COLS; x++ ) {
            uint32_t rgb = frameBuffer[ y * COLS + x ];
            uint8_t  _r  = ( uint8_t )( ( rgb & 0xFF000000 ) >> 24 );
            uint8_t  _g  = ( uint8_t )( ( rgb & 0x00FF0000 ) >> 16 );
            uint8_t  _b  = ( uint8_t )( ( rgb & 0x0000FF00 ) >> 8 );

            fprintf( file, "%d %d %d\n", _r, _g, _b );
        }
    }

    fflush( file );
    fclose( file );

    return 0;
}

static void _renderThread( const void* context )
{
    const RenderThreadContext* ctx = (const RenderThreadContext*)context;

    for ( uint32_t y = ctx->yOffset; y < ctx->yOffset + ctx->blockSize; y++ ) {
        for ( uint32_t x = ctx->xOffset; x < ctx->xOffset + ctx->blockSize; x++ ) {

            // TEST
            if ( ctx->debug && ( y == ctx->yOffset || y == ctx->yOffset + ctx->blockSize - 1 || x == ctx->xOffset || x == ctx->xOffset + ctx->blockSize - 1 ) ) {
                ctx->frameBuffer[ y * COLS + x ] = 0xFF000000;
                continue;
            }

            // Sample each pixel in image space, with anti-aliasing

            vec3 color( 0, 0, 0 );
            for ( int s = 0; s < NUM_AA_SAMPLES; s++ ) {
                float u = float( x + random() ) / float( COLS );
                float v = float( y + random() ) / float( ROWS );
                ray   r = ctx->camera->getRay( u, v );
                color += _color( r, ctx->scene, 0 );
            }
            color /= float( NUM_AA_SAMPLES );

            // Apply 2.0 Gamma correction
            color = vec3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

            uint8_t _r = ( uint8_t )( 255.99 * color.x );
            uint8_t _g = ( uint8_t )( 255.99 * color.y );
            uint8_t _b = ( uint8_t )( 255.99 * color.z );

            uint32_t rgb                     = ( (uint32_t)_r << 24 ) | ( (uint32_t)_g << 16 ) | ( (uint32_t)_b << 8 );
            ctx->frameBuffer[ y * COLS + x ] = rgb;
        }
    }

    // Notify caller that we have completed one work item
    std::atomic<uint32_t>* blockCount = ctx->blockCount;
    uint32_t               count      = blockCount->fetch_add( 1 );

    return;
}

// Recursively trace each ray through objects/materials
static vec3 _color( const ray& r, const Scene* scene, float depth )
{
    hit_info hit;

    if ( scene->hit( r, 0.001f, std::numeric_limits<float>::max(), &hit ) ) {
#if defined( NORMAL_SHADE )
        vec3 normal = ( r.point( hit.t ) - vec3( 0, 0, -1 ) ).normalized();
        return 0.5f * vec3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
        if ( depth < MAX_RAY_DEPTH ) {
            vec3 target = hit.point + hit.normal + randomInUnitSphere();
            return 0.5f * _color( ray( hit.point, target - hit.point ), scene, depth + 1 );
        } else {
            return vec3( 0, 0, 0 );
        }
#else
        ray  scattered;
        vec3 attenuation;
        if ( depth < MAX_RAY_DEPTH && hit.material && hit.material->scatter( r, hit, &attenuation, &scattered ) ) {
            return attenuation * _color( scattered, scene, depth + 1 );
        } else {
            return vec3( 0, 0, 0 );
        }

#endif
    }

    return _background( r );
}

static vec3 _background( const ray& r )
{
    vec3  unitDirection = r.direction.normalized();
    float t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * vec3( 0.5f, 0.7f, 1.0f );
}

static Scene* _randomScene()
{
    Scene* scene = new Scene();

    scene->objects.push_back( new Sphere( vec3( 0, -1000.0f, 0 ), 1000, new Diffuse( vec3( 0.5f, 0.5f, 0.5f ) ) ) );

    //scene->objects.push_back( new Sphere( vec3( -4, 1, 0 ), 1.0f, new Diffuse( vec3( 0.4f, 0.2f, 0.1f ) ) ) );
    //scene->objects.push_back( new Sphere( vec3( 0, 1, 0 ), 1.0f, new Glass( 1.5f ) ) );
    //scene->objects.push_back( new Sphere( vec3( 4, 1, 0 ), 1.0f, new Metal( vec3( 0.7f, 0.6f, 0.5f ), 0.0f ) ) );

    for ( int a = -11; a < 11; a++ ) {
        for ( int b = -11; b < 11; b++ ) {
            float material = random();
            vec3  center( a + 0.9f * random(), 0.2f, b + 0.9f * random() );

            if ( ( center - vec3( 4.0f, 0.2f, 0.0f ) ).length() > 0.9f ) {
                if ( material < 0.8f ) {
                    scene->objects.push_back( new Sphere( center, 0.2f, new Diffuse( vec3( random() * random(), random() * random(), random() * random() ) ) ) );
                } else if ( material > 0.95f ) {
                    scene->objects.push_back( new Sphere( center, 0.2f, new Metal( vec3( 0.5f * ( 1 + random() ), 0.5f * ( 1 + random() ), 0.5f * ( 1 + random() ) ), 0.5f * random() ) ) );
                } else {
                    scene->objects.push_back( new Sphere( center, 0.2f, new Glass( 1.5f ) ) );
                }
            }
        }
    }

    scene->objects.push_back( new Sphere( vec3( -4, 1, 0 ), 1.0f, new Diffuse( vec3( 0.4f, 0.2f, 0.1f ) ) ) );
    scene->objects.push_back( new Sphere( vec3( 0, 1, 0 ), 1.0f, new Glass( 1.5f ) ) );
    scene->objects.push_back( new Sphere( vec3( 4, 1, 0 ), 1.0f, new Metal( vec3( 0.7f, 0.6f, 0.5f ), 0.0f ) ) );

    return scene;
}
