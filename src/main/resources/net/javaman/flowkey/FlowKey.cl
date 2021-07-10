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
                checkPixelEquality(input, gid - 3, colorKey) == 1
            ) {
                float colorDiff = calcColorDiff(input, gid, template, gid - 3, colorSpace);
                if (colorDiff < gradientTolerance) {
                    for (int i = 0; i < 3; i++) {
                        output[gid + i] = colorKey[i];
                    }
                    return;
                }
            }
            if (
                (gid / 3) % width != width - 1 &&
                checkPixelEquality(input, gid + 3, colorKey) == 1
            ) {
                float colorDiff = calcColorDiff(input, gid, template, gid + 3, colorSpace);
                if (colorDiff < gradientTolerance) {
                    for (int i = 0; i < 3; i++) {
                        output[gid + i] = colorKey[i];
                    }
                    return;
                }
            }
            if (
                (gid / 3) / width != 0 &&
                checkPixelEquality(input, gid - (width * 3), colorKey) == 1
            ) {
                float colorDiff = calcColorDiff(input, gid, template, gid - (width * 3), colorSpace);
                if (colorDiff < gradientTolerance) {
                    for (int i = 0; i < 3; i++) {
                        output[gid + i] = colorKey[i];
                    }
                    return;
                }
            }
            if (
                (gid / 3) / width != height - 1 &&
                checkPixelEquality(input, gid + (width * 3), colorKey) == 1
            ) {
                float colorDiff = calcColorDiff(input, gid, template, gid + (width * 3), colorSpace);
                if (colorDiff < gradientTolerance) {
                    for (int i = 0; i < 3; i++) {
                        output[gid + i] = colorKey[i];
                    }
                    return;
                }
            }
            for (int i = 0; i < 3; i++) {
                output[gid + i] = template[gid + i];
            }
        } else {
            for (int i = 0; i < 3; i++) {
                output[gid + i] = colorKey[i];
            }
        }
    }
}