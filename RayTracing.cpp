// Ray Tracing In One Weekend
//

#include "camera.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "vec3.h"

#include <limits>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <string>
using namespace pk;

#define ARRAY_SIZE( x ) ( sizeof( x ) / sizeof( x[ 0 ] ) )

const unsigned int COLS = 200;
const unsigned int ROWS = 100;

// Anti-aliasing
const unsigned int NUM_AA_SAMPLES = 100;

// Num bounces per ray
const unsigned int MAX_RAY_DEPTH = 50;


static int  _generateTestPPM( const std::string& filename, const Scene& scene, const Camera& camera );
static vec3 _color( const ray& r, const Scene& scene, float depth );
static vec3 _background( const ray& r );
static vec3 _randomInUnitSphere();

static std::random_device                    rd;
static std::mt19937                          gen( rd() );
static std::uniform_real_distribution<float> random( 0.0f, 1.0f );


int main()
{
    Scene scene;
    scene.objects.push_back( new Sphere( vec3( 0, -100.5f, -1 ), 100.0f, new Diffuse( vec3( 0.8f, 0.8f, 0.0f ) ) ) );
    scene.objects.push_back( new Sphere( vec3( 1, 0, -1 ), 0.5f, new Metal( vec3( 0.8f, 0.6f, 0.2f ), 0.0f ) ) );
    //scene.objects.push_back( new Sphere( vec3( -1, 0, -1 ), 0.5f, new Glass( 1.5f ) ) );
    scene.objects.push_back( new Sphere( vec3( -1, 0, -1 ), -0.45f, new Glass( 1.5f ) ) );
    scene.objects.push_back( new Sphere( vec3( 0, 0, -1 ), 0.5f, new Diffuse( vec3( 0.1f, 0.2f, 0.5f ) ) ) );

    vec3 origin(0, 0, 0);
    vec3 lookat(0, 0, -1);
    vec3 up(0, 1, 0);

    Camera camera( 90, float( COLS ) / float( ROWS ), origin, up, lookat );

    return _generateTestPPM( "foo.ppm", scene, camera );
}

static int _generateTestPPM( const std::string& filename, const Scene& scene, const Camera& camera )
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


    for ( int y = ROWS - 1; y >= 0; y-- ) {
        for ( int x = 0; x < COLS; x++ ) {

            // Sample each pixel in image space, with anti-aliasing

            vec3 color( 0, 0, 0 );
            for ( int s = 0; s < NUM_AA_SAMPLES; s++ ) {
                float u = float( x + random( gen ) ) / float( COLS );
                float v = float( y + random( gen ) ) / float( ROWS );
                ray   r = camera.getRay( u, v );
                color += _color( r, scene, 0 );
            }
            color /= NUM_AA_SAMPLES;

            // Apply 2.0 Gamma correction
            color = vec3( sqrt( color.r() ), sqrt( color.g() ), sqrt( color.b() ) );

            uint8_t _r = ( uint8_t )( 255.99 * color.x );
            uint8_t _g = ( uint8_t )( 255.99 * color.y );
            uint8_t _b = ( uint8_t )( 255.99 * color.z );

            fprintf( file, "%d %d %d\n", _r, _g, _b );
        }
    }

    fflush( file );
    fclose( file );

    return 0;
}

// Recursively trace each ray through objects/materials
static vec3 _color( const ray& r, const Scene& scene, float depth )
{
    hit_info hit;

    if ( scene.hit( r, 0.001f, std::numeric_limits<float>::max(), &hit ) ) {
#if defined( NORMAL_SHADE )
        vec3 normal = ( r.point( hit.t ) - vec3( 0, 0, -1 ) ).normalized();
        return 0.5f * vec3( normal.x + 1, normal.y + 1, normal.z + 1 );
#elif defined( DIFFUSE_SHADE )
        vec3 target = hit.point + hit.normal + _randomInUnitSphere();
        return 0.5f * _color( ray( hit.point, target - hit.point ), scene );
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

    //static vec3 color( random( gen ), random( gen ), random( gen ) );
    //return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * color;

    return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * vec3( 0.5f, 0.7f, 1.0f );
}

static vec3 _randomInUnitSphere()
{
    vec3 point;
    do {
        point = 2.0f * vec3( random( gen ), random( gen ), random( gen ) ) - vec3( 1, 1, 1 );
    } while ( point.squared_length() >= 1.0 );

    return point;
}
