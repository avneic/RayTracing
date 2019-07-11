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
    inline Vector2 normalized() const
    {
        Vector2 v = *this;
        v.normalize();

        return v;
    }

    // Normalize this vector
    inline void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y );
        x /= length;
        y /= length;
    }

    inline float length() const
    {
        return sqrt( x * x + y * y );
    }

    inline float squared_length() const
    {
        return x * x + y * y;
    }

    inline _TYPE dot( const Vector2& v ) const
    {
        return x * v.x + y * v.y;
    }

    inline _TYPE angle( const Vector2& v ) const
    {
        return acos( v.Dot( *this ) );
    }

    inline Vector2 operator-() const
    {
        return Vector2( -x, -y );
    }

    inline Vector2 operator-( float f ) const
    {
        return Vector2( x - f, y - f );
    }

    inline Vector2 operator-( int i ) const
    {
        return Vector2( x - i, y - i );
    }

    inline Vector2 operator+( float f ) const
    {
        return Vector2( x + f, y + f );
    }

    inline Vector2 operator+( int i ) const
    {
        return Vector2( x + i, y + i );
    }

    inline Vector2 operator+( const Vector2& v ) const
    {
        return Vector2( x + v.x, y + v.y );
    }

    inline Vector2 operator-( const Vector2& v ) const
    {
        return Vector2( x - v.x, y - v.y );
    }

    inline Vector2 operator*( _TYPE f )
    {
        return Vector2( x * f, y * f );
    }

    inline Vector2 operator/( _TYPE f )
    {
        return Vector2( x / f, y / f );
    }

    inline bool operator==( const Vector2& v ) const
    {
        return x == v.x && y == v.y;
    }

    inline bool operator!=( const Vector2& v ) const
    {
        return x != v.x || y != v.y;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    inline bool operator<( const Vector2& v ) const
    {
        return length() < v.length();
    }

    inline bool operator>( const Vector2& v ) const
    {
        return length() > v.length();
    }

    inline Vector2& operator+=( const Vector2& v )
    {
        x += v.x;
        y += v.y;
        return *this;
    }

    inline Vector2& operator-=( const Vector2& v )
    {
        x -= v.x;
        y -= v.y;
        return *this;
    }

    inline Vector2& operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
        return *this;
    }

    inline Vector2& operator/=( _TYPE s )
    {
        x /= s;
        y /= s;
        return *this;
    }

    inline Vector2& operator*=(const Vector2& v)
    {
        x *= v.x;
        y *= v.y;
        return *this;
    }

    _TYPE x;
    _TYPE y;
};

inline Vector2<float> operator*( float s, const Vector2<float>& v )
{
    return Vector2<float>( v.x * s, v.y * s );
}

inline Vector2<float> operator/( float s, const Vector2<float>& v )
{
    return Vector2<float>( v.x / s, v.y / s );
}

inline Vector2<int> operator*( int s, const Vector2<int>& v )
{
    return Vector2<int>( v.x * s, v.y * s );
}

inline Vector2<int> operator/( int s, const Vector2<int>& v )
{
    return Vector2<int>( v.x / s, v.y / s );
}

inline Vector2<float> operator*(const Vector2<float>& u, const Vector2<float>& v)
{
    return Vector2<float>(u.x * v.x, u.y * v.y);
}

inline Vector2<int> operator*(const Vector2<int>& u, const Vector2<int>& v)
{
    return Vector2<int>(u.x * v.x, u.y * v.y);
}



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

    inline Vector3 normalized() const
    {
        Vector3 v = *this;
        v.normalize();

        return v;
    }

    inline void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y + z * z );
        x /= length;
        y /= length;
        z /= length;
    }

    inline float length() const
    {
        return (float)sqrt( x * x + y * y + z * z );
    }

    inline float squared_length() const
    {
        return x * x + y * y + z * z;
    }
    
    inline Vector3 cross( const Vector3& v ) const
    {
        return Vector3( y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x );
    }

    inline _TYPE dot( const Vector3& v ) const
    {
        return (x * v.x) + (y * v.y) + (z * v.z);
    }

    inline _TYPE angle( const Vector3& v ) const
    {
        return acos( v.dot( *this ) );
    }

    inline Vector3 operator+( const Vector3& v ) const
    {
        return Vector3( x + v.x, y + v.y, z + v.z );
    }

    inline Vector3 operator-( const Vector3& v ) const
    {
        return Vector3( x - v.x, y - v.y, z - v.z );
    }

    inline Vector3 operator-( float f ) const
    {
        return Vector3( x - f, y - f, z - f );
    }

    inline Vector3 operator-( int i ) const
    {
        return Vector3( x - i, y - i, z - i );
    }

    inline Vector3 operator+( float f ) const
    {
        return Vector3( x + f, y + f, z + f );
    }

    inline Vector3 operator+( int i ) const
    {
        return Vector3( x + i, y + i, z + i );
    }

    inline Vector3 operator-() const
    {
        return Vector3( -x, -y, -z );
    }

    inline Vector3 operator*( _TYPE s ) const
    {
        return Vector3( x * s, y * s, z * s );
    }

    inline Vector3 operator/( _TYPE s ) const
    {
        return Vector3( x / s, y / s, z / s );
    }

    inline bool operator==( const Vector3& v ) const
    {
        return x == v.x && y == v.y && z == v.z;
    }

    inline bool operator!=( const Vector3& v ) const
    {
        return x != v.x || y != v.y || z != v.z;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    inline bool operator<( const Vector3& v ) const
    {
        return length() < v.length();
    }

    inline bool operator>( const Vector3& v ) const
    {
        return length() > v.length();
    }

    inline Vector3& operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }

    inline Vector3& operator/=( _TYPE s )
    {
        x /= s;
        y /= s;
        z /= s;
        return *this;
    }

    inline Vector3& operator+=( const Vector3& v )
    {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    inline Vector3& operator-=( const Vector3& v )
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    inline Vector3& operator*=(const Vector3& v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        return *this;
    }

    inline Vector3& operator/=(const Vector3& v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        return *this;
    }

    inline operator const _TYPE*() const
    {
        return static_cast<const _TYPE*>( &x );
    }

    inline float r() const { return x; }
    inline float g() const { return y; }
    inline float b() const { return z; }

    _TYPE x;
    _TYPE y;
    _TYPE z;
};


inline Vector3<float> operator*( float s, const Vector3<float>& v )
{
    return Vector3<float>( v.x * s, v.y * s, v.z * s );
}

inline Vector3<float> operator/( float s, const Vector3<float>& v )
{
    return Vector3<float>( v.x / s, v.y / s, v.z / s );
}

inline Vector3<int> operator*( int s, const Vector3<int>& v )
{
    return Vector3<int>( v.x * s, v.y * s, v.z * s );
}

inline Vector3<int> operator/( int s, const Vector3<int>& v )
{
    return Vector3<int>( v.x / s, v.y / s, v.z / s );
}

inline Vector3<float> operator*(const Vector3<float>& u, const Vector3<float>& v)
{
    return Vector3<float>(u.x * v.x, u.y * v.y, u.z * v.z);
}

inline Vector3<int> operator*(const Vector3<int>& u, const Vector3<int>& v)
{
    return Vector3<int>(u.x * v.x, u.y * v.y, u.z * v.z);
}


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

    inline Vector4 normalized() const
    {
        Vector4 v = *this;
        v.normalize();

        return v;
    }

    inline void normalize()
    {
        _TYPE length = (_TYPE)sqrt( x * x + y * y + z * z + w * w );
        x /= length;
        y /= length;
        z /= length;
        w /= length;
    }

    inline float length() const
    {
        return sqrt( x * x + y * y + z * z + w * w );
    }

    inline float squared_length() const
    {
        return x * x + y * y + z * z + w * w;
    }
    
    inline Vector4 operator-() const
    {
        return Vector4( -x, -y, -z, -w );
    }

    inline Vector4 operator-( float f ) const
    {
        return Vector4( x - f, y - f, z - f, w - f );
    }

    inline Vector4 operator-( int i ) const
    {
        return Vector4( x - i, y - i, z - i, w - i );
    }

    inline Vector4 operator-( const Vector4& v ) const
    {
        return Vector4( x - v.x, y - v.y, z - v.z, w - v.w );
    }

    inline Vector4 operator+( float f ) const
    {
        return Vector4( x + f, y + f, z + f, w + f );
    }

    inline Vector4 operator+( int i ) const
    {
        return Vector4( x + i, y + i, z + i, w + i );
    }

    inline Vector4 operator+( const Vector4& v ) const
    {
        return Vector4( x + v.x, y + v.y, z + v.z, w + v.w );
    }

    inline Vector4 operator*( _TYPE s ) const
    {
        return Vector4( x * s, y * s, z * s, w * s );
    }

    inline Vector4 operator/( _TYPE s ) const
    {
        return Vector4( x / s, y / s, z / s, w / s );
    }

    inline bool operator==( const Vector4& v ) const
    {
        return x == v.x && y == v.y && z == v.z && w == v.w;
    }

    inline bool operator!=( const Vector4& v ) const
    {
        return x != v.x || y != v.y || z != v.z || w != v.w;
    }

    // We use L2 norm to determine which Vector is < or > the other.
    // Generally correct when finding min/pax points, e.g. for a bounding box.

    inline bool operator<( const Vector4& v ) const
    {
        return length() < v.length();
    }

    inline bool operator>( const Vector4& v ) const
    {
        return length() > v.length();
    }

    inline Vector4& operator*=( _TYPE s )
    {
        x *= s;
        y *= s;
        z *= s;
        w *= s;
        return *this;
    }

    inline Vector4& operator/=( _TYPE s )
    {
        x /= s;
        y /= s;
        z /= s;
        w /= s;
        return *this;
    }

    inline Vector4& operator+=( const Vector4& v )
    {
        x += v.x;
        y += v.y;
        z += v.z;
        w += v.w;
        return *this;
    }

    inline Vector4& operator-=( const Vector4& v )
    {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        w -= v.w;
        return *this;
    }

    inline Vector4& operator*=(const Vector4& v)
    {
        x *= v.x;
        y *= v.y;
        z *= v.z;
        w *= v.w;
        return *this;
    }

    inline Vector4& operator/=(const Vector4& v)
    {
        x /= v.x;
        y /= v.y;
        z /= v.z;
        w /= v.w;
        return *this;
    }

    inline float r() const { return x; }
    inline float g() const { return y; }
    inline float b() const { return z; }
    inline float a() const { return w; }

    _TYPE x;
    _TYPE y;
    _TYPE z;
    _TYPE w;
};


inline Vector4<float> operator*( float s, const Vector4<float>& v )
{
    return Vector4<float>( v.x * s, v.y * s, v.z * s, v.w * s );
}

inline Vector4<float> operator/( float s, const Vector4<float>& v )
{
    return Vector4<float>( v.x / s, v.y / s, v.z / s, v.w / s );
}

inline Vector4<int> operator*( int s, const Vector4<int>& v )
{
    return Vector4<int>( v.x * s, v.y * s, v.z * s, v.w * s );
}

inline Vector4<int> operator/( int s, const Vector4<int>& v )
{
    return Vector4<int>( v.x / s, v.y / s, v.z / s, v.w / s );
}

inline Vector4<float> operator*(const Vector4<float>& u, const Vector4<float>& v)
{
    return Vector4<float>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}

inline Vector4<int> operator*(const Vector4<int>& u, const Vector4<int>& v)
{
    return Vector4<int>(u.x * v.x, u.y * v.y, u.z * v.z, u.w * v.w);
}


typedef Vector2<int> ivec2;
typedef Vector3<int> ivec3;
typedef Vector4<int> ivec4;

typedef Vector2<float> vec2;
typedef Vector3<float> vec3;
typedef Vector4<float> vec4;

} // namespace pk
