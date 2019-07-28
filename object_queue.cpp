#include "object_queue.h"


namespace pk
{

bool testQueueFunction( void* context, uint32_t tid )
{
    const char* s = (const char*)context;

    printf( "[0x%x]: %s\n", tid, s );

    return false;
}


void testObjectQueues()
{
    // Test that we can enqueue complex objects, and they survive enqueu/dequeue

    const int MAX_QUEUE_DEPTH = 5;

    class Object {
    public:
        int                                          id;
        std::string                                  str;
        std::function<bool( const char*, uint32_t )> method;
        std::function<bool( void*, uint32_t )>       function;

        bool doSomething( const char* str, uint32_t value )
        {
            printf( "0x%p [0x%x]: %s\n", this, value, str );
            return true;
        }

        Object() :
            id( -1 )
        {
        }

        ~Object() {}
    };

    Object* buffer = new Object[ MAX_QUEUE_DEPTH ];

    obj_queue_t queue = Queue<Object>::create( MAX_QUEUE_DEPTH, buffer );

    for ( int i = 0; i < MAX_QUEUE_DEPTH; i++ ) {
        Object obj;
        obj.id       = i;
        obj.str      = "a string";
        obj.method   = std::bind( &Object::doSomething, &obj, std::placeholders::_1, std::placeholders::_2 );
        obj.function = std::bind( &testQueueFunction, std::placeholders::_1, std::placeholders::_2 );

        if ( R_OK == Queue<Object>::send( queue, &obj ) ) {
            printf( "Send: %d %s\n", obj.id, obj.str.c_str() );
        } else {
            printf( "Send: %d %s FAILED\n", obj.id, obj.str.c_str() );
        }
    }

    for ( int i = 0; i < MAX_QUEUE_DEPTH; i++ ) {
        Object obj;

        Queue<Object>::receive( queue, &obj, sizeof( Object ) );
        printf( "Recv: %d %s\n", obj.id, obj.str.c_str() );

        obj.function( (void*)"a function call", 0xDEADBEEF );
        obj.method( "a method call", 0xDEADBEEF );
    }

    Queue<Object>::destroy( queue );
}

} // namespace pk
