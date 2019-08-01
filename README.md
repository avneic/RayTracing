# RayTracing
Implementation of Peter Shirley's Ray Tracing in One Weekend.  This version is multi-threaded with a job sysem, and also supports ISPC or CUDA.
The output is a .PPM file; I recommend Image Viewer Pro for viewing it.


Ray tracer uses CPU multi-threading by default.

Set output filename with -f \<filename\> (defaults to foo.ppm)

Set number of threads with -t \<n\> (defaults to logical CPU cores - 1).

Set block size with -b \<width\> (defaults to 64 x 64)

```
C:\> RayTracing.exe -t 4 -b 32
```

Ray tracer supports CUDA if you have a recent Nvidia GPU.
Enable CUDA mode with -c flag

```
C:\> RayTracing.exe -c
```

Ray tracer can mix threads and SIMD if you enable ISPC with -i.

```
C:\> RayTracing.exe -i 
```
