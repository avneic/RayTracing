// Ray Tracing In One Weekend
//

#include "camera.h"
#include "material.h"
#include "ray.h"
#include "sphere.h"
#include "utils.h"
#include "vec3.h"

#include <limits>
#include <stdint.h>
#include <stdio.h>
#include <string>
using namespace pk;

#define ARRAY_SIZE( x ) ( sizeof( x ) / sizeof( x[ 0 ] ) )

//#define DIFFUSE_SHADE
//#define NORMAL_SHADE

//const unsigned int COLS = 1200;
//const unsigned int ROWS = 800;
const unsigned int COLS = 1200;
const unsigned int ROWS = 800;

// Anti-aliasing
const unsigned int NUM_AA_SAMPLES = 10;

// Num bounces per ray
const unsigned int MAX_RAY_DEPTH = 50;


static int    _generateTestPPM( const std::string& filename, const Scene& scene, const Camera& camera );
static vec3   _color( const ray& r, const Scene& scene, float depth );
static vec3   _background( const ray& r );
static Scene* _randomScene();


int main()
{
    //Scene scene;
    //scene.objects.push_back( new Sphere( vec3( 0, -100.5f, -1 ), 100.0f, new Diffuse( vec3( 0.8f, 0.8f, 0.0f ) ) ) );
    //scene.objects.push_back( new Sphere( vec3( -1, 0, -1 ), -0.45f, new Glass( 1.5f ) ) );
    //scene.objects.push_back( new Sphere( vec3( 0, 0, -1 ), 0.5f, new Diffuse( vec3( 0.1f, 0.2f, 0.5f ) ) ) );
    //scene.objects.push_back( new Sphere( vec3( 1, 0, -1 ), 0.5f, new Metal( vec3( 0.8f, 0.6f, 0.2f ), 0.0f ) ) );

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

    _generateTestPPM( "foo.ppm", *scene, camera );

    delete scene;

    return 0;
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
                float u = float( x + random() ) / float( COLS );
                float v = float( y + random() ) / float( ROWS );
                ray   r = camera.getRay( u, v );
                color += _color( r, scene, 0 );
            }
            color /= float( NUM_AA_SAMPLES );

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

    //static vec3 color( random( gen ), random( gen ), random( gen ) );
    //return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * color;

    return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * vec3( 0.5f, 0.7f, 1.0f );
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
