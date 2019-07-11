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
#include "vector.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <thread>

using namespace pk;

const unsigned int COLS = 1280;
const unsigned int ROWS = 1280;

// Anti-aliasing
const unsigned int NUM_AA_SAMPLES = 50;

// Max bounces per ray
const unsigned int MAX_RAY_DEPTH = 50;

static Scene* _randomScene();

//
// Simple Ray Tracer
//

int main( int argc, char** argv )
{
    ArgsParser args( argc, argv );

    // TEST
    //testCPU();
    //testCPUThreaded();
    //testCUDA();
    //return 0;

    Scene* scene = _randomScene();

    vec3  origin( 13, 2, 3 );
    vec3  lookat( 0, 0, 0 );
    vec3  up( 0, 1, 0 );
    float vfov   = 20;
    float aspect = float( COLS ) / float( ROWS );

    //float focusDistance = ( origin - lookat ).length();
    float focusDistance = 10.0f;
    float aperture      = 0.1f;

    Camera camera( vfov, aspect, aperture, focusDistance, origin, up, lookat );

    // Render to RAM (multi-threaded) then flush to disk
    uint32_t* frameBuffer = nullptr;
    frameBuffer           = new uint32_t[ ROWS * COLS ];
    memset( frameBuffer, 0xCC, sizeof( uint32_t ) * ROWS * COLS );

    int numThreads = std::thread::hardware_concurrency() - 1;
    if ( args.cmdOptionExists( "-t" ) ) {
        const std::string& arg = args.getCmdOption( "-t" );
        numThreads             = std::stoi( arg );
    }

    std::string filename = "foo.ppm";
    if ( args.cmdOptionExists( "-f" ) ) {
        filename = args.getCmdOption( "-f" );
    }

    int blockSize = 64;
    if ( args.cmdOptionExists( "-b" ) ) {
        const std::string& arg = args.getCmdOption( "-b" );
        blockSize              = std::stoi( arg );
    }

    bool debug = false;
    if ( args.cmdOptionExists( "-d" ) ) {
        debug = true;
    }

    bool recursive = false;
    if ( args.cmdOptionExists( "-r" ) ) {
        recursive = true;
    }

    FILE*   file = nullptr;
    errno_t err  = fopen_s( &file, filename.c_str(), "w" );
    if ( !file || err != 0 ) {
        printf( "Error: failed to open [%s] for writing errno %d.\n", filename.c_str(), err );
        return -1;
    }

    renderScene( *scene, camera, ROWS, COLS, frameBuffer, NUM_AA_SAMPLES, MAX_RAY_DEPTH, numThreads, blockSize, debug, recursive );

    // Write to disk
    fprintf( file, "P3\n" );
    fprintf( file, "%d %d\n", ROWS, COLS );
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
    delete[] frameBuffer;

    return 0;
}

static Scene* _randomScene()
{
    Scene* scene = new Scene();

    scene->objects.push_back( new Sphere( vec3( 0, -1000.0f, 0 ), 1000, new Diffuse( vec3( 0.5f, 0.5f, 0.5f ) ) ) );

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
