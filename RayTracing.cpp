// Ray Tracing In One Weekend
//

#include "argsparser.h"
#include "camera.h"
#include "material.h"
#include "msgqueue.h"
#include "perf_timer.h"
#include "ray.h"
#include "raytracer.h"
#include "scene.h"
#include "sphere.h"
#include "test.h"
#include "utils.h"
#include "vector_cuda.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace pk;

const unsigned int COLS = 2000;
const unsigned int ROWS = 1000;

// Anti-aliasing
// NOTE: if a CUDA program takes > 2 seconds to execute, Windows may kill it.
const unsigned int NUM_AA_SAMPLES = 25;

// Max bounces per ray
const unsigned int MAX_RAY_DEPTH = 5;

static Scene* _randomScene();

//
// Simple Ray Tracer
//

int main( int argc, char** argv )
{
    //
    // Parse command-line args
    //
    ArgsParser args( argc, argv );

    bool cuda = true;
    //bool cuda = false;
    if ( args.cmdOptionExists( "-c" ) ) {
        cuda = false;
    }

    int aaSamples = NUM_AA_SAMPLES;
    if ( args.cmdOptionExists( "-a" ) ) {
        const std::string& arg = args.getCmdOption( "-a" );
        aaSamples              = std::stoi( arg );
    }

    int blockSize = cuda ? 16 : 64;
    if ( args.cmdOptionExists( "-b" ) ) {
        const std::string& arg = args.getCmdOption( "-b" );
        blockSize              = std::stoi( arg );
    }

    bool debug = false;
    if ( args.cmdOptionExists( "-d" ) ) {
        debug = true;
    }

    std::string filename = "foo.ppm";
    if ( args.cmdOptionExists( "-f" ) ) {
        filename = args.getCmdOption( "-f" );
    }

    int maxBounce = MAX_RAY_DEPTH;
    if ( args.cmdOptionExists( "-m" ) ) {
        const std::string& arg = args.getCmdOption( "-m" );
        maxBounce              = std::stoi( arg );
    }

    bool recursive = false;
    if ( args.cmdOptionExists( "-r" ) ) {
        recursive = true;
    }

    int numThreads = std::thread::hardware_concurrency() - 1;
    if ( args.cmdOptionExists( "-t" ) ) {
        const std::string& arg = args.getCmdOption( "-t" );
        numThreads             = std::stoi( arg );
    }


    //
    // Define the scene and camera
    //
    Scene* scene = _randomScene();

    vector3 origin( 13, 2, 3 );
    vector3 lookat( 0, 0, 0 );
    vector3 up( 0, 1, 0 );
    float   vfov   = 20;
    float   aspect = float( COLS ) / float( ROWS );

    //float focusDistance = ( origin - lookat ).length();
    float focusDistance = 10.0f;
    float aperture      = 0.1f;

    Camera camera( vfov, aspect, aperture, focusDistance, origin, up, lookat );

    //
    // Allocate frame buffer
    // TODO: floating point instead of 8-bit RGB
    //
    uint32_t* frameBuffer = nullptr;
    if ( cuda ) {
        CHECK_CUDA( cudaMallocManaged( (void**)&frameBuffer, ROWS * COLS * sizeof( uint32_t ) ) );
    } else {
        frameBuffer = new uint32_t[ ROWS * COLS ];
        memset( frameBuffer, 0xCC, sizeof( uint32_t ) * ROWS * COLS );
    }

    //
    // Open file to save image
    //
    FILE*   file = nullptr;
    errno_t err  = fopen_s( &file, filename.c_str(), "w" );
    if ( !file || err != 0 ) {
        printf( "Error: failed to open [%s] for writing errno %d.\n", filename.c_str(), err );
        return -1;
    }

    if ( cuda ) {
        renderSceneCUDA( *scene, camera, ROWS, COLS, frameBuffer, aaSamples, maxBounce, numThreads, blockSize, debug, recursive );
    } else {
        renderScene( *scene, camera, ROWS, COLS, frameBuffer, aaSamples, maxBounce, numThreads, blockSize, debug, recursive );
    }

    //
    // Save image
    //
    fprintf( file, "P3\n" );
    fprintf( file, "%d %d\n", COLS, ROWS );
    fprintf( file, "255\n" );

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

    delete scene;

    if ( cuda ) {
        CHECK_CUDA( cudaFree( frameBuffer ) );
        CHECK_CUDA( cudaDeviceReset() );
    } else {
        delete[] frameBuffer;
    }

    return 0;
}

static Scene* _randomScene()
{
    Scene* scene = new Scene();

    scene->objects.push_back( new Sphere( vector3( 0, -1000.0f, 0 ), 1000, new material_t( MATERIAL_DIFFUSE, vector3( 0.5f, 0.5f, 0.5f ) ) ) );

    for ( int a = -11; a < 11; a++ ) {
        for ( int b = -11; b < 11; b++ ) {
            float   material = random();
            vector3 center( a + 0.9f * random(), 0.2f, b + 0.9f * random() );

            if ( ( center - vector3( 4.0f, 0.2f, 0.0f ) ).length() > 0.9f ) {
                if ( material < 0.8f ) {
                    scene->objects.push_back( new Sphere( center, 0.2f, new material_t( MATERIAL_DIFFUSE, vector3( random() * random(), random() * random(), random() * random() ) ) ) );
                } else if ( material > 0.95f ) {
                    scene->objects.push_back( new Sphere( center, 0.2f, new material_t( MATERIAL_METAL, vector3( 0.5f * ( 1 + random() ), 0.5f * ( 1 + random() ), 0.5f * ( 1 + random() ) ), 0.5f * random() ) ) );
                } else {
                    scene->objects.push_back( new Sphere( center, 0.2f, new material_t( MATERIAL_GLASS, vector3( 1, 1, 1 ), 1.0f, 1.5f ) ) );
                }
            }
        }
    }

    scene->objects.push_back( new Sphere( vector3( -4, 1, 0 ), 1.0f, new material_t( MATERIAL_DIFFUSE, vector3( 0.4f, 0.2f, 0.1f ) ) ) );
    scene->objects.push_back( new Sphere( vector3( 0, 1, 0 ), 1.0f, new material_t( MATERIAL_GLASS, vector3( 0.4f, 0.2f, 0.1f ), 1.0f, 1.5f ) ) );
    scene->objects.push_back( new Sphere( vector3( 4, 1, 0 ), 1.0f, new material_t( MATERIAL_METAL, vector3( 0.7f, 0.6f, 0.5f ), 0.0f ) ) );

    //scene->objects.push_back( new Sphere( vector3( 0, 0, -1 ), 0.5f, new material_t( MATERIAL_DIFFUSE, vector3( 0.4f, 0.2f, 0.1f ) ) ) );
    //scene->objects.push_back( new Sphere( vector3( 0, -100.5f, -1 ), 100.0f, new material_t( MATERIAL_DIFFUSE, vector3( 0.5f, 0.5f, 0.5f ) ) ) );

    return scene;
}
