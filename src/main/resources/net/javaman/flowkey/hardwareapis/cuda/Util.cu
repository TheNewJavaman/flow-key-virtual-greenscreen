enum ColorSpace {
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    ALL = 3
};

__device__ int checkPixelEquality(
    char *input,
    int i,
    char *colorKey
) {
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

__device__ float calcColorDiff(
    char *a,
    int i,
    char *b,
    int j,
    int colorSpace
) {
    float colorDiff[3];
    for (int k = 0; k < 3; k++) {
        colorDiff[k] = abs(a[i + k] - b[j + k]);
    }
    if (colorSpace < 3) {
        return colorDiff[colorSpace] / 255.0;
    } else {
        float percentDiff = 0.0;
        for (int i = 0; i < 3; i++) {
            percentDiff += colorDiff[i] / 765.0;
        }
        return percentDiff;
    }
}

__device__ void writePixel(
    char *canvas,
    int i,
    char *ink,
    int j
) {
    for (int k = 0; k < 3; k++) {
        canvas[i + k] = ink[j + k];
    }
}