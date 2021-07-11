__kernel void initialComparisonKernel(
    __global const char *input,
    __global char *output,
    __global const char *colorKey,
    __global const char *replacementKey,
    __global const float *floatOptions,
    __global const int *intOptions
) {
    float percentTolerance = floatOptions[PERCENT_TOLERANCE];
    int colorSpace = intOptions[COLOR_SPACE];
    int gid = get_global_id(0);

    if (gid % 3 == 0) {
        float percentDiff = calcColorDiff(input, gid, colorKey, 0, colorSpace);
        if (percentDiff < percentTolerance) {
            writePixel(output, gid, replacementKey, 0);
        } else {
            writePixel(output, gid, input, gid);
        }
    }
}