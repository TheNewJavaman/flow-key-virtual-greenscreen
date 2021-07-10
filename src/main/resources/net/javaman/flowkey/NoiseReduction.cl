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
                surroundingPixels += checkPixelEquality(input, gid + (width * 3), colorKey);
            }
            if (surroundingPixels < 3) {
                for (int i = 0; i < 3; i++) {
                    output[gid + i] = template[gid + i];
                }
            } else {
                for (int i = 0; i < 3; i++) {
                    output[gid + i] = colorKey[i];
                }
            }
        } else {
            for (int i = 0; i < 3; i++) {
                output[gid + i] = template[gid + i];
            }
        }
    }
}

int checkPixelEquality(const char *input, const int i, const char *colorKey) {
    int diffSum = 0;
    for (int j = 0; j < 3; j++) {
        diffSum += abs(input[i + j] - colorKey[j]);
    }
    if (diffSum == 0) {
        return 1;
    } else {
        return 0;
    }
}