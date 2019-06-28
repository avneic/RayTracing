// Ray Tracing In One Weekend
//

#include "pch.h"
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

static int   _generateTestPPM( const std::string& filename, const Scene& scene );
static vec3  _color( const ray& r, const Scene& scene );
static vec3  _background( const ray& r );
//static float _rayIntersectsSphere( const ray& r, const vec3& center, float radius );

int main()
{
    Scene scene;
    scene.objects.push_back( new Sphere( vec3( 0, -100.5f, -1 ), 100.0f ) );
    scene.objects.push_back( new Sphere( vec3( 0, 0, -1 ), 0.5f ) );

    return _generateTestPPM( "foo.ppm", scene );
}

static int _generateTestPPM( const std::string& filename, const Scene& scene )
{
    const unsigned int COLS = 200;
    const unsigned int ROWS = 100;

     FILE* file = nullptr;
    errno_t err = fopen_s( &file, filename.c_str(), "w" );
    if ( !file || err != 0 ) {
        printf( "Error: failed to open [%s] for writing errno %d.\n", filename.c_str(), err );
        return -1;
    }

    // Print PPM header
    fprintf( file, "P3\n" );
    fprintf( file, "%d %d\n", COLS, ROWS );
    fprintf( file, "255\n" );

    vec3 origin( 0, 0, 0 );
    vec3 horizontal( 4, 0, 0 );
    vec3 vertical( 0, 2, 0 );
    //vec3 lowerLeftCorner(-2, -1, -1);
    vec3 upperLeftCorner( -2, 1, -1 );

    for ( unsigned int y = 0; y < ROWS; y++ ) {
    //for ( int y = ROWS-1; y >= 0; y-- ) {
        for ( int x = 0; x < COLS; x++ ) {
            float u = float( x ) / float(COLS);
            float v = float( y ) / float(ROWS);

            //ray r(origin, lowerLeftCorner + (u * horizontal) + (v * vertical));
            ray r( origin, upperLeftCorner + u * horizontal + -v * vertical );

            vec3 color = _color( r, scene );

            uint8_t _r = ( uint8_t )( 255 * color.x );
            uint8_t _g = ( uint8_t )( 255 * color.y );
            uint8_t _b = ( uint8_t )( 255 * color.z );

            fprintf( file, "%d %d %d\n", _r, _g, _b );
        }
    }

    fflush( file );
    fclose( file );

    return 0;
}

static vec3 _color( const ray& r, const Scene& scene )
{
    hit_info hit;

    if ( scene.hit( r, 0, std::numeric_limits<float>::max(), &hit ) ) {
        vec3 normal = ( r.point( hit.t ) - vec3( 0, 0, -1 ) ).normalized();
        return 0.5f * vec3( normal.x + 1, normal.y + 1, normal.z + 1 );
    }

    return _background( r );
}

static vec3 _background( const ray& r )
{
    vec3  unitDirection = r.direction.normalized();
    float t             = 0.5f * ( unitDirection.y + 1.0f );

    static std::random_device                    rd;
    static std::mt19937                          gen( rd() );
    static std::uniform_real_distribution<float> random( 0.0f, 1.0f );
    static vec3                                  color( random( gen ), random( gen ), random( gen ) );

    return ( 1.0f - t ) * vec3( 1.0f, 1.0f, 1.0f ) + t * color;
}

//static float _rayIntersectsSphere( const ray& r, const vec3& center, float radius )
//{
//    vec3  oc           = r.origin - center;
//    float a            = r.direction.dot( r.direction );
//    float b            = 2.0f * oc.dot( r.direction );
//    float c            = oc.dot( oc ) - radius * radius;
//    float discriminant = b * b - 4 * a * c;
//
//    if ( discriminant < 0 ) {
//        return -1;
//    }
//
//    return ( -b - sqrt( discriminant ) ) / ( 2.0f * a );
//}
