enum ColorSpace {
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    ALL = 3
};

enum FloatOptions {
    PERCENT_TOLERANCE = 0,
    GRADIENT_TOLERANCE = 1
};

enum IntOptions {
    COLOR_SPACE = 0,
    WIDTH = 1,
    HEIGHT = 2,
    BLOCK_SIZE = 3
};

int checkPixelEquality(
    const char *input, 
    const int i, 
    const char *colorKey
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

float calcColorDiff(
    const char *a,
    const int i,
    const char *b,
    const int j,
    const int colorSpace
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

void writePixel(
    char *canvas,
    const int i,
    const char *ink,
    const int j
) {
    for (int k = 0; k < 3; k++) {
        canvas[i + k] = ink[j + k];
    }
}

float avgBlock(
    const unsigned char *canvas,
    const int canvasWidth,
    const int canvasHeight,
    const int blockSize,
    const int blockX,
    const int blockY
) {
    int blockSum = 0;
    int pixelCount = 0;
    for (int row = blockSize * blockY; row < blockY * blockSize + blockSize && row < canvasHeight; row++) {
        for (int col = blockSize * blockX; col < blockX * blockSize + blockSize && col < canvasWidth; col++) {
            for (int i = 0; i < 3; i++) {
                blockSum += canvas[(canvasWidth * row + col) * 3 + i];
            }
            pixelCount += 1;
        }
    }
    return (float) blockSum / pixelCount / 3.0;
}