__kernel void flowKeyKernel(
    __global const char *input,
    __global char *output,
    __global const char *template,
    __global const char *colorKey,
    __global const float *floatOptions,
    __global const int *intOptions
) {
    float gradientTolerance = floatOptions[GRADIENT_TOLERANCE];
    int colorSpace = intOptions[COLOR_SPACE];
    int width = intOptions[WIDTH];
    int height = intOptions[HEIGHT];
    int gid = get_global_id(0);

    if (gid % 3 == 0) {
        if (checkPixelEquality(input, gid, colorKey) == 0) {
            if (
                (gid / 3) % width != 0 &&
                checkPixelEquality(input, gid - 3, colorKey) == 1 &&
                calcColorDiff(input, gid, template, gid - 3, colorSpace) > gradientTolerance
            ) {
                writePixel(output, gid, colorKey, 0);
                return;
            }
            if (
                (gid / 3) % width != width - 1 &&
                checkPixelEquality(input, gid + 3, colorKey) == 1 &&
                calcColorDiff(input, gid, template, gid + 3, colorSpace) > gradientTolerance
            ) {
                writePixel(output, gid, colorKey, 0);
                return;
            }
            if (
                (gid / 3) / width != 0 &&
                checkPixelEquality(input, gid - (width * 3), colorKey) == 1 && 
                calcColorDiff(input, gid, template, gid - (width * 3), colorSpace) > gradientTolerance
            ) {
                writePixel(output, gid, colorKey, 0);
                return;
            }
            if (
                (gid / 3) / width != height - 1 &&
                checkPixelEquality(input, gid + (width * 3), colorKey) == 1 &&
                calcColorDiff(input, gid, template, gid + (width * 3), colorSpace) > gradientTolerance
            ) {
                writePixel(output, gid, colorKey, 0);
                return;
            }
            writePixel(output, gid, template, gid);
        } else {
            writePixel(output, gid, colorKey, 0);
        }
    } else {
        writePixel(output, gid, colorKey, 0);
    }
}