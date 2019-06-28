#pragma once

#include <math.h>

namespace pk
{

template<typename _TYPE>
class Vector2 {
public:
    Vector2() { x = 0, y = 0; }

    Vector2( _TYPE x, _TYPE y )
    {
        this->x = x;
        this->y = y;
    }

    ~Vector2() {}

    // Return a normalized copy of this vector
    Vector2 normalized() const
    {
        Vector2 v = *this;
        v.normalize();

        return v;
    }

    // Normalize this vector
    void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y );
        x /= length;
        y /= length;
    }

    float length() const
    {
        return sqrt( x * x + y * y );
    }

    _TYPE dot( const Vector2& v ) const
    {
        return x * v.x + y * v.y;
    }

    _TYPE angle( const Vector2& v ) const
    {
        return acos( v.Dot( *this ) );
    }

    Vector2 operator-() const
    {
        return Vector2( -x, -y );
    }

    Vector2 operator-( float f ) const
    {
        return Vector2( x - f, y - f );
    }

    Vector2 operator-( int i ) const
    {
        return Vector2( x - i, y - i );
    }

    Vector2 operator+( float f ) const
    {
        return Vector2( x + f, y + f );
    }

    Vector2 operator+( int i ) const
    {
        return Vector2( x + i, y + i );
    }

    Vector2 operator+( const Vector2& v ) const
    {
        return Vector2( x + v.x, y + v.y );
    }

    Vector2 operator-( const Vector2& v ) const
    {
        return Vector2( x - v.x, y - v.y );
    }

    Vector2 operator*( _TYPE f )
    {
        return Vector2( x * f, y * f );
    }

    Vector2 operator/( _TYPE f )
    {
        return Vector2( x / f, y / f );
    }

    bool operator==( const Vector2& v ) const
    {
        return x == v.x && y == v.y;
    }

    bool operator!=( const Vector2& v ) const
    {
        return x != v.x || y != v.y;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    bool operator<( const Vector2& v ) const
    {
        return length() < v.length();
    }

    bool operator>( const Vector2& v ) const
    {
        return length() > v.length();
    }

    void operator+=( const Vector2& v )
    {
        x += v.x;
        y += v.y;
    }

    void operator-=( const Vector2& v )
    {
        x -= v.x;
        y -= v.y;
    }

    void operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
    }

    _TYPE x;
    _TYPE y;
};

Vector2<float> operator*(float s, const Vector2<float>& v);
Vector2<float> operator/(float s, const Vector2<float>& v);
Vector2<int> operator*(int s, const Vector2<int>& v);
Vector2<int> operator/(int s, const Vector2<int>& v);


template<typename _TYPE>
class Vector3 {
public:
    Vector3() { x = 0, y = 0, z = 0; }

    Vector3( _TYPE x, _TYPE y, _TYPE z )
    {
        this->x = x;
        this->y = y;
        this->z = z;
    }

    ~Vector3() {}

    Vector3 normalized() const
    {
        Vector3 v = *this;
        v.normalize();

        return v;
    }

    void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y + z * z );
        x /= length;
        y /= length;
        z /= length;
    }

    float length() const
    {
        return sqrt( x * x + y * y + z * z );
    }

    Vector3 cross( const Vector3& v ) const
    {
        return Vector3( y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x );
    }

    _TYPE dot( const Vector3& v ) const
    {
        return (x * v.x) + (y * v.y) + (z * v.z);
    }

    _TYPE angle( const Vector3& v ) const
    {
        return acos( v.dot( *this ) );
    }

    Vector3 operator+( const Vector3& v ) const
    {
        return Vector3( x + v.x, y + v.y, z + v.z );
    }

    void operator+=( const Vector3& v )
    {
        x += v.x;
        y += v.y;
        z += v.z;
    }

    void operator-=( const Vector3& v )
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
    }

    void operator/=( _TYPE s )
    {
        x /= s;
        y /= s;
        z /= s;
    }

    void operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
        z *= s;
    }

    Vector3 operator-( const Vector3& v ) const
    {
        return Vector3( x - v.x, y - v.y, z - v.z );
    }

    Vector3 operator-( float f ) const
    {
        return Vector3( x - f, y - f, z - f );
    }

    Vector3 operator-( int i ) const
    {
        return Vector3( x - i, y - i, z - i );
    }

    Vector3 operator+( float f ) const
    {
        return Vector3( x + f, y + f, z + f );
    }

    Vector3 operator+( int i ) const
    {
        return Vector3( x + i, y + i, z + i );
    }

    Vector3 operator-() const
    {
        return Vector3( -x, -y, -z );
    }

    Vector3 operator*( _TYPE s ) const
    {
        return Vector3( x * s, y * s, z * s );
    }

    Vector3 operator/( _TYPE s ) const
    {
        return Vector3( x / s, y / s, z / s );
    }

    bool operator==( const Vector3& v ) const
    {
        return x == v.x && y == v.y && z == v.z;
    }

    bool operator!=( const Vector3& v ) const
    {
        return x != v.x || y != v.y || z != v.z;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    bool operator<( const Vector3& v ) const
    {
        return length() < v.length();
    }

    bool operator>( const Vector3& v ) const
    {
        return length() > v.length();
    }

    operator const _TYPE*() const
    {
        return static_cast<const _TYPE*>( &x );
    }

    _TYPE x;
    _TYPE y;
    _TYPE z;
};


Vector3<float> operator*(float s, const Vector3<float>& v);
Vector3<float> operator/(float s, const Vector3<float>& v);
Vector3<int> operator*(int s, const Vector3<int>& v);
Vector3<int> operator/(int s, const Vector3<int>& v);


template<typename _TYPE>
class Vector4 {
public:
    Vector4() { x = 0, y = 0, z = 0, w = 0; }

    Vector4( _TYPE x, _TYPE y, _TYPE z, _TYPE w )
    {
        this->x = x;
        this->y = y;
        this->z = z;
        this->w = w;
    }

    ~Vector4() {}

    Vector4 normalized() const
    {
        Vector4 v = *this;
        v.normalize();

        return v;
    }

    void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y + z * z + w * w );
        x /= length;
        y /= length;
        z /= length;
        w /= length;
    }

    float length()
    {
        return sqrt( x * x + y * y + z * z + w * w );
    }

    Vector4 operator-() const
    {
        return Vector4( -x, -y, -z, -w );
    }

    Vector4 operator-( float f ) const
    {
        return Vector4( x - f, y - f, z - f, w - f );
    }

    Vector4 operator-( int i ) const
    {
        return Vector4( x - i, y - i, z - i, w - i );
    }

    Vector4 operator-( const Vector4& v ) const
    {
        return Vector4( x - v.x, y - v.y, z - v.z, w - v.w );
    }

    Vector4 operator+( float f ) const
    {
        return Vector4( x + f, y + f, z + f, w + f );
    }

    Vector4 operator+( int i ) const
    {
        return Vector4( x + i, y + i, z + i, w + i );
    }

    Vector4 operator+( const Vector4& v ) const
    {
        return Vector4( x + v.x, y + v.y, z + v.z, w + v.w );
    }

    Vector4 operator*( _TYPE s ) const
    {
        return Vector4( x * s, y * s, z * s, w * s );
    }

    Vector4 operator/( _TYPE s ) const
    {
        return Vector4( x / s, y / s, z / s, w / s );
    }

    bool operator==( const Vector4& v ) const
    {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    bool operator!=( const Vector4& v ) const
    {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    bool operator<( const Vector4& v ) const
    {
        return length() < v.length();
    }

    bool operator>( const Vector4& v ) const
    {
        return length() > v.length();
    }

    void operator+=( const Vector4& v )
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
    }

    void operator-=( const Vector4& v )
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
    }

    void operator/=( _TYPE s )
    {
        x /= s;
        y /= s;
        z /= s;
    }

    void operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
    }

    _TYPE x;
    _TYPE y;
    _TYPE z;
    _TYPE w;
};

Vector4<float> operator*(float s, const Vector4<float>& v);
Vector4<float> operator/(float s, const Vector4<float>& v);
Vector4<int> operator*(int s, const Vector4<int>& v);
Vector4<int> operator/(int s, const Vector4<int>& v);

//typedef Vector2<int> ivec2;
//typedef Vector3<int> ivec3;
//typedef Vector4<int> ivec4;
//
//typedef Vector2<float> vec2;
//typedef Vector3<float> vec3;
//typedef Vector4<float> vec4;

} // namespace pk
