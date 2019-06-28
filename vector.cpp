#include "vector.h"

namespace pk
{

Vector2<float> operator*( float s, const Vector2<float>& v )
{
    return Vector2<float>( v.x * s, v.y * s );
}

Vector2<float> operator/( float s, const Vector2<float>& v )
{
    return Vector2<float>( v.x / s, v.y / s );
}

Vector2<int> operator*( int s, const Vector2<int>& v )
{
    return Vector2<int>( v.x * s, v.y * s );
}

Vector2<int> operator/( int s, const Vector2<int>& v )
{
    return Vector2<int>( v.x / s, v.y / s );
}

Vector3<float> operator*( float s, const Vector3<float>& v )
{
    return Vector3<float>( v.x * s, v.y * s, v.z * s );
}

Vector3<float> operator/( float s, const Vector3<float>& v )
{
    return Vector3<float>( v.x / s, v.y / s, v.z / s );
}

Vector3<int> operator*( int s, const Vector3<int>& v )
{
    return Vector3<int>( v.x * s, v.y * s, v.z * s );
}

Vector3<int> operator/( int s, const Vector3<int>& v )
{
    return Vector3<int>( v.x / s, v.y / s, v.z / s );
}

Vector4<float> operator*( float s, const Vector4<float>& v )
{
    return Vector4<float>( v.x * s, v.y * s, v.z * s, v.w * s );
}

Vector4<float> operator/( float s, const Vector4<float>& v )
{
    return Vector4<float>( v.x / s, v.y / s, v.z / s, v.w / s );
}

Vector4<int> operator*( int s, const Vector4<int>& v )
{
    return Vector4<int>( v.x * s, v.y * s, v.z * s, v.w * s );
}

Vector4<int> operator/( int s, const Vector4<int>& v )
{
    return Vector4<int>( v.x / s, v.y / s, v.z / s, v.w / s );
}

} // namespace pk
