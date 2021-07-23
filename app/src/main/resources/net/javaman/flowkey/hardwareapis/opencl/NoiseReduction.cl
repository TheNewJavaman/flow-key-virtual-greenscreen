__kernel void noiseReductionKernel(
    __global const char *input,
    __global char *output,
    __global const char *template,
    __global const char *colorKey,
    __global const int *intOptions
) {
    int width = intOptions[WIDTH];
    int height = intOptions[HEIGHT];
    int gid = get_global_id(0);

    if (gid % 3 == 0) {
        int anchorEquality = checkPixelEquality(input, gid, colorKey);
        if (anchorEquality == 1) {
            int surroundingPixels = 0;
            if ((gid / 3) % width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, gid - 3, colorKey);
            }
            if ((gid / 3) % width == width - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, gid + 3, colorKey);
            }
            if ((gid / 3) / width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, gid - (width * 3), colorKey);
            }
            if ((gid / 3) / width == height - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, gid + (width * 3), colorKey);
            }
            if (surroundingPixels < 3) {
                writePixel(output, gid, template, gid);
            } else {
                writePixel(output, gid, colorKey, 0);
            }
        } else {
            writePixel(output, gid, template, gid);
        }
    }
}