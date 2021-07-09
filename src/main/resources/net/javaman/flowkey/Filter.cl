__kernel void sampleKernel(
    __global const char *a,
    __global char *b
) {
    int gid = get_global_id(0);
    b[gid] = a[gid];
}