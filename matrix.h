#pragma once

#include "vector.hpp"

#include <math.h>

namespace pk
{

template<typename T>
struct Matrix2 {
    Matrix2()
    {
        x.x = 1;
        x.y = 0;
        y.x = 0;
        y.y = 1;
    }
    Matrix2( const T* m )
    {
        x.x = m[ 0 ];
        x.y = m[ 1 ];
        y.x = m[ 2 ];
        y.y = m[ 3 ];
    }
    vec2 x;
    vec2 y;
};

template<typename T>
struct Matrix3 {
    Matrix3()
    {
        x.x = 1;
        x.y = 0;
        x.z = 0;
        y.x = 0;
        y.y = 1;
        y.z = 0;
        z.x = 0;
        z.y = 0;
        z.z = 1;
    }
    Matrix3( const T* m )
    {
        x.x = m[ 0 ];
        x.y = m[ 1 ];
        x.z = m[ 2 ];
        y.x = m[ 3 ];
        y.y = m[ 4 ];
        y.z = m[ 5 ];
        z.x = m[ 6 ];
        z.y = m[ 7 ];
        z.z = m[ 8 ];
    }
    Matrix3 transposed() const
    {
        Matrix3 m;
        m.x.x = x.x;
        m.x.y = y.x;
        m.x.z = z.x;
        m.y.x = x.y;
        m.y.y = y.y;
        m.y.z = z.y;
        m.z.x = x.z;
        m.z.y = y.z;
        m.z.z = z.z;
        return m;
    }
    const T* pointer() const
    {
        return &x.x;
    }
    vec3 x;
    vec3 y;
    vec3 z;
};

template<typename T>
struct Matrix4 {
    Matrix4()
    {
        x.x = 1;
        x.y = 0;
        x.z = 0;
        x.w = 0;
        y.x = 0;
        y.y = 1;
        y.z = 0;
        y.w = 0;
        z.x = 0;
        z.y = 0;
        z.z = 1;
        z.w = 0;
        w.x = 0;
        w.y = 0;
        w.z = 0;
        w.w = 1;
    }
    Matrix4( const Matrix3<T>& m )
    {
        x.x = m.x.x;
        x.y = m.x.y;
        x.z = m.x.z;
        x.w = 0;
        y.x = m.y.x;
        y.y = m.y.y;
        y.z = m.y.z;
        y.w = 0;
        z.x = m.z.x;
        z.y = m.z.y;
        z.z = m.z.z;
        z.w = 0;
        w.x = 0;
        w.y = 0;
        w.z = 0;
        w.w = 1;
    }
    Matrix4( const T* m )
    {
        x.x = m[ 0 ];
        x.y = m[ 1 ];
        x.z = m[ 2 ];
        x.w = m[ 3 ];
        y.x = m[ 4 ];
        y.y = m[ 5 ];
        y.z = m[ 6 ];
        y.w = m[ 7 ];
        z.x = m[ 8 ];
        z.y = m[ 9 ];
        z.z = m[ 10 ];
        z.w = m[ 11 ];
        w.x = m[ 12 ];
        w.y = m[ 13 ];
        w.z = m[ 14 ];
        w.w = m[ 15 ];
    }
    Matrix4 operator*( const Matrix4& b ) const
    {
        Matrix4 m;
        m.x.x = x.x * b.x.x + x.y * b.y.x + x.z * b.z.x + x.w * b.w.x;
        m.x.y = x.x * b.x.y + x.y * b.y.y + x.z * b.z.y + x.w * b.w.y;
        m.x.z = x.x * b.x.z + x.y * b.y.z + x.z * b.z.z + x.w * b.w.z;
        m.x.w = x.x * b.x.w + x.y * b.y.w + x.z * b.z.w + x.w * b.w.w;
        m.y.x = y.x * b.x.x + y.y * b.y.x + y.z * b.z.x + y.w * b.w.x;
        m.y.y = y.x * b.x.y + y.y * b.y.y + y.z * b.z.y + y.w * b.w.y;
        m.y.z = y.x * b.x.z + y.y * b.y.z + y.z * b.z.z + y.w * b.w.z;
        m.y.w = y.x * b.x.w + y.y * b.y.w + y.z * b.z.w + y.w * b.w.w;
        m.z.x = z.x * b.x.x + z.y * b.y.x + z.z * b.z.x + z.w * b.w.x;
        m.z.y = z.x * b.x.y + z.y * b.y.y + z.z * b.z.y + z.w * b.w.y;
        m.z.z = z.x * b.x.z + z.y * b.y.z + z.z * b.z.z + z.w * b.w.z;
        m.z.w = z.x * b.x.w + z.y * b.y.w + z.z * b.z.w + z.w * b.w.w;
        m.w.x = w.x * b.x.x + w.y * b.y.x + w.z * b.z.x + w.w * b.w.x;
        m.w.y = w.x * b.x.y + w.y * b.y.y + w.z * b.z.y + w.w * b.w.y;
        m.w.z = w.x * b.x.z + w.y * b.y.z + w.z * b.z.z + w.w * b.w.z;
        m.w.w = w.x * b.x.w + w.y * b.y.w + w.z * b.z.w + w.w * b.w.w;
        return m;
    }
    Matrix4& operator*=( const Matrix4& b )
    {
        Matrix4 m = *this * b;
        return ( *this = m );
    }
    Matrix4 transposed() const
    {
        Matrix4 m;
        m.x.x = x.x;
        m.x.y = y.x;
        m.x.z = z.x;
        m.x.w = w.x;
        m.y.x = x.y;
        m.y.y = y.y;
        m.y.z = z.y;
        m.y.w = w.y;
        m.z.x = x.z;
        m.z.y = y.z;
        m.z.z = z.z;
        m.z.w = w.z;
        m.w.x = x.w;
        m.w.y = y.w;
        m.w.z = z.w;
        m.w.w = w.w;
        return m;
    }
    Matrix3<T> toMat3() const
    {
        Matrix3<T> m;
        m.x.x = x.x;
        m.y.x = y.x;
        m.z.x = z.x;
        m.x.y = x.y;
        m.y.y = y.y;
        m.z.y = z.y;
        m.x.z = x.z;
        m.y.z = y.z;
        m.z.z = z.z;
        return m;
    }
    const T* pointer() const
    {
        return &x.x;
    }

    static Matrix4<T> identity()
    {
        return Matrix4();
    }

    static Matrix4<T> translate( T x, T y, T z )
    {
        Matrix4 m;
        m.x.x = 1;
        m.x.y = 0;
        m.x.z = 0;
        m.x.w = 0;
        m.y.x = 0;
        m.y.y = 1;
        m.y.z = 0;
        m.y.w = 0;
        m.z.x = 0;
        m.z.y = 0;
        m.z.z = 1;
        m.z.w = 0;
        m.w.x = x;
        m.w.y = y;
        m.w.z = z;
        m.w.w = 1;
        return m;
    }

    static Matrix4<T> scale( T s )
    {
        Matrix4 m;
        m.x.x = s;
        m.x.y = 0;
        m.x.z = 0;
        m.x.w = 0;
        m.y.x = 0;
        m.y.y = s;
        m.y.z = 0;
        m.y.w = 0;
        m.z.x = 0;
        m.z.y = 0;
        m.z.z = s;
        m.z.w = 0;
        m.w.x = 0;
        m.w.y = 0;
        m.w.z = 0;
        m.w.w = 1;
        return m;
    }

    static Matrix4<T> scale( T sx, T sy, T sz )
    {
        Matrix4 m;
        m.x *= sx;
        m.y *= sy;
        m.z *= sz;
        return m;
    }

    static Matrix4<T> rotate( float angle, T x, T y, T z )
    {
        Matrix4 rotMat;

        float sinAngle, cosAngle;
        float mag = sqrtf( x * x + y * y + z * z );

        sinAngle = sinf( angle * 3.14159f / 180.0f );
        cosAngle = cosf( angle * 3.14159f / 180.0f );
        if ( mag > 0.0f ) {
            float xx, yy, zz, xy, yz, zx, xs, ys, zs;
            float oneMinusCos;

            x /= mag;
            y /= mag;
            z /= mag;

            xx          = x * x;
            yy          = y * y;
            zz          = z * z;
            xy          = x * y;
            yz          = y * z;
            zx          = z * x;
            xs          = x * sinAngle;
            ys          = y * sinAngle;
            zs          = z * sinAngle;
            oneMinusCos = 1.0f - cosAngle;

            rotMat.x.x = ( oneMinusCos * xx ) + cosAngle;
            rotMat.x.y = ( oneMinusCos * xy ) - zs;
            rotMat.x.z = ( oneMinusCos * zx ) + ys;
            rotMat.x.w = 0.0F;

            rotMat.y.x = ( oneMinusCos * xy ) + zs;
            rotMat.y.y = ( oneMinusCos * yy ) + cosAngle;
            rotMat.y.z = ( oneMinusCos * yz ) - xs;
            rotMat.y.w = 0.0F;

            rotMat.z.x = ( oneMinusCos * zx ) - ys;
            rotMat.z.y = ( oneMinusCos * yz ) + xs;
            rotMat.z.z = ( oneMinusCos * zz ) + cosAngle;
            rotMat.z.w = 0.0F;

            rotMat.w.x = 0.0F;
            rotMat.w.y = 0.0F;
            rotMat.w.z = 0.0F;
            rotMat.w.w = 1.0F;
        }

        return rotMat;
    }

    static Matrix4<T> rotateX( T degrees )
    {
        T radians = degrees * 3.14159f / 180.0f;
        T s       = sin( radians );
        T c       = cos( radians );

        Matrix4 m;
        m.x.x = 1;
        m.x.y = 0;
        m.x.z = 0;
        m.x.w = 0;
        m.y.x = 0;
        m.y.y = c;
        m.y.z = -s;
        m.y.w = 0;
        m.z.x = 0;
        m.z.y = s;
        m.z.z = c;
        m.z.w = 0;
        m.w.x = 0;
        m.w.y = 0;
        m.w.z = 0;
        m.w.w = 1;
        return m;
    }
    static Matrix4<T> rotateY( T degrees )
    {
        T radians = degrees * 3.14159f / 180.0f;
        T s       = sin( radians );
        T c       = cos( radians );

        Matrix4 m;
        m.x.x = c;
        m.x.y = 0;
        m.x.z = s;
        m.x.w = 0;
        m.y.x = 0;
        m.y.y = 1;
        m.y.z = 0;
        m.y.w = 0;
        m.z.x = -s;
        m.z.y = 0;
        m.z.z = c;
        m.z.w = 0;
        m.w.x = 0;
        m.w.y = 0;
        m.w.z = 0;
        m.w.w = 1;
        return m;
    }
    static Matrix4<T> rotateZ( T degrees )
    {
        T radians = degrees * 3.14159f / 180.0f;
        T s       = sin( radians );
        T c       = cos( radians );

        Matrix4 m;
        m.x.x = c;
        m.x.y = -s;
        m.x.z = 0;
        m.x.w = 0;
        m.y.x = s;
        m.y.y = c;
        m.y.z = 0;
        m.y.w = 0;
        m.z.x = 0;
        m.z.y = 0;
        m.z.z = 1;
        m.z.w = 0;
        m.w.x = 0;
        m.w.y = 0;
        m.w.z = 0;
        m.w.w = 1;
        return m;
    }

    static Matrix4<T> frustum( float left, float right, float bottom, float top, float nearZ, float farZ )
    {
        Matrix4<T> frust;

        if ( right - left <= 0 || top - bottom <= 0 || farZ - nearZ <= 0 )
            return frust;

        float a = 2.0f * nearZ / ( right - left );
        float b = 2.0f * nearZ / ( top - bottom );
        float c = ( right + left ) / ( right - left );
        float d = ( top + bottom ) / ( top - bottom );
        float e = -( farZ + nearZ ) / ( farZ - nearZ );
        float f = -2.0f * farZ * nearZ / ( farZ - nearZ );

        frust.x.x = a;
        frust.x.y = 0;
        frust.x.z = 0;
        frust.z.w = 0;
        frust.y.x = 0;
        frust.y.y = b;
        frust.y.z = 0;
        frust.y.w = 0;
        frust.z.x = c;
        frust.z.y = d;
        frust.z.z = e;
        frust.z.w = -1;
        frust.w.x = 0;
        frust.w.y = 0;
        frust.w.z = f;
        frust.w.w = 0; // the OpenGL ES book says w.w = 1 in, but that was a printing error.

        return frust;
    }

    static Matrix4<T> perspective( float fFov, float aspectRatio, float nearZ, float farZ )
    {
        float frustumW, frustumH;

        frustumH = tanf( fFov / 360.0f * 3.14159f ) * nearZ;
        frustumW = frustumH * aspectRatio;

        return Matrix4<T>::frustum( -frustumW, frustumW, -frustumH, frustumH, nearZ, farZ );
    }

    static Matrix4<T> ortho( float left, float right, float bottom, float top, float nearZ, float farZ )
    {
        float      deltaX = right - left;
        float      deltaY = top - bottom;
        float      deltaZ = farZ - nearZ;
        Matrix4<T> ortho;

        if ( ( deltaX == 0.0f ) || ( deltaY == 0.0f ) || ( deltaZ == 0.0f ) )
            return ortho;

        ortho.x.x = 2.0f / deltaX;
        ortho.w.x = -( right + left ) / deltaX;
        ortho.y.y = 2.0f / deltaY;
        ortho.w.y = -( top + bottom ) / deltaY;
        ortho.z.z = -2.0f / deltaZ;
        ortho.w.z = -( nearZ + farZ ) / deltaZ;

        return ortho;
    }

    vec4 operator*( const vec4& vector ) const
    {
        vec4 result;

        result.x = ( vector.x * x.x ) + ( vector.y * y.x ) + ( vector.z * z.x ) + ( vector.w * w.x );
        result.y = ( vector.x * x.y ) + ( vector.y * y.y ) + ( vector.z * z.y ) + ( vector.w * w.y );
        result.z = ( vector.x * x.z ) + ( vector.y * y.z ) + ( vector.z * z.z ) + ( vector.w * w.z );
        result.w = ( vector.x * x.w ) + ( vector.y * y.w ) + ( vector.z * z.w ) + ( vector.w * w.w );

        return result;
    }

    vec3 operator*( const vec3& vector ) const
    {
        vec3 result;

        result.x = ( vector.x * x.x ) + ( vector.y * y.x ) + ( vector.z * z.x );
        result.y = ( vector.x * x.y ) + ( vector.y * y.y ) + ( vector.z * z.y );
        result.z = ( vector.x * x.z ) + ( vector.y * y.z ) + ( vector.z * z.z );

        return result;
    }

    vec4 x;
    vec4 y;
    vec4 z;
    vec4 w;
};

typedef Matrix2<float> mat2;
typedef Matrix3<float> mat3;
typedef Matrix4<float> mat4;

} // namespace pk
