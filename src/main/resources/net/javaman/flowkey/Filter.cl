enum ColorSpace {
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    ALL = 3
};

enum FloatOptions {
    PERCENT_TOLERANCE = 0
};

enum IntOptions {
    COLOR_SPACE = 0,
    NOISE_REDUCTION = 1
};

__kernel void greenscreenKernel(
    __global const char *input,
    __global char *output,
    __global const char *colorKey,
    __global const char *replacementKey,
    __global const float *floatOptions,
    __global const int *intOptions
) {
    int gid = get_global_id(0);
    if (gid % 3 == 0) {
        float percentTolerance = floatOptions[0];
        int colorSpace = intOptions[0];
        int noiseReduction = intOptions[1];

        float colorDiff[3];
        for (int i = 0; i < 3; i++) {
            colorDiff[i] = abs(input[gid + i] - colorKey[i]);
        }
        float percentDiff = 0;
        if (colorSpace < 3) {
            percentDiff = colorDiff[colorSpace] / 255.0;
        } else {
            for (int i = 0; i < 3; i++) {
                percentDiff += colorDiff[i] / 765.0;
            }
        }
        if (percentDiff < percentTolerance) {
            for (int i = 0; i < 3; i++) {
                output[gid + i] = replacementKey[i];
            }
        } else {
            for (int i = 0; i < 3; i++) {
                output[gid + i] = input[gid + i];
            }
        }


    }
}