

export void simple(
    uniform float vin[],
    uniform float vout[],
    uniform int count
)
{
}


/************
export vector3 _color( const ray& r, const sphere_t* scene, uint32_t sceneSize, unsigned max_depth )
{
    hit_info hit;
    vector3  attenuation;
    ray      scattered = r;
    vector3  color( 1, 1, 1 );

    for ( unsigned i = 0; i < max_depth; i++ ) {
        if ( _sceneHit( scene, sceneSize, scattered, 0.001f, FLT_MAX, &hit ) ) {
#if defined( NORMAL_SHADE )
            vector3 normal = ( r.point( hit.distance ) - vector3( 0, 0, -1 ) ).normalized();
            return 0.5f * vector3( normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f );
#elif defined( DIFFUSE_SHADE )
            vector3 target = hit.point + hit.normal + randomInUnitSphereCUDA( rand );
            scattered      = ray( hit.point, target - hit.point );
            color *= 0.5f;
#else
            if ( hit.material && materialScatter( hit.material, scattered, hit, &attenuation, &scattered ) ) {
                color *= attenuation;
            } else {
                break;
            }
#endif
        } else {
            color *= _background( scattered );
            break;
        }
    }

    return color;
}


export vector3 _background( const ray& r )
{
    vector3 unitDirection = r.direction.normalized();
    float   t             = 0.5f * ( unitDirection.y + 1.0f );

    return ( 1.0f - t ) * vector3( 1.0f, 1.0f, 1.0f ) + t * vector3( 0.5f, 0.7f, 1.0f );
}


export bool _sphereHit( const sphere_t& sphere, const ray& r, float min, float max, hit_info* p_hit )
{
    assert( p_hit );

    vector3 oc = r.origin - sphere.center;
    float   a  = r.direction.dot( r.direction );
    float   b  = oc.dot( r.direction );
    float   c  = oc.dot( oc ) - ( sphere.radius * sphere.radius );

    float discriminant = b * b - a * c;

    if ( discriminant > 0 ) {
        float t = ( -b - sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - sphere.center ) / sphere.radius;
            p_hit->material = &sphere.material;
            return true;
        }

        t = ( -b + sqrt( discriminant ) ) / a;
        if ( t < max && t > min ) {
            p_hit->distance = t;
            p_hit->point    = r.point( t );
            p_hit->normal   = ( p_hit->point - sphere.center ) / sphere.radius;
            p_hit->material = &sphere.material;
            return true;
        }
    }

    return false;
}

export bool _sceneHit( const sphere_t* scene, uint32_t sceneSize, const ray& r, float min, float max, hit_info* p_hit )
{
    bool     rval         = false;
    float    closestSoFar = max;
    hit_info hit;

    for ( int i = 0; i < sceneSize; i++ ) {
        const sphere_t& sphere = scene[ i ];

        hit_info tmp;
        if ( _sphereHit( sphere, r, min, closestSoFar, &tmp ) ) {
            rval         = true;
            closestSoFar = tmp.distance;
            hit          = tmp;
        }
    }

    *p_hit = hit;
    return rval;
}
*****/
