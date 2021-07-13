extern "C" __global__ void flowKeyKernel(
    int n,
    char *input,
    char *output,
    char *original,
    char *colorKey,
    float gradientTolerance,
    int colorSpace,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelEquality(input, idx, colorKey) == 0) {
            if (
                (idx / 3) % width != 0 &&
                checkPixelEquality(input, idx - 3, colorKey) == 1 &&
                calcColorDiff(input, idx, original, idx - 3, colorSpace) > gradientTolerance
            ) {
                writePixel(output, idx, colorKey, 0);
                return;
            }
            if (
                (idx / 3) % width != width - 1 &&
                checkPixelEquality(input, idx + 3, colorKey) == 1 &&
                calcColorDiff(input, idx, original, idx + 3, colorSpace) > gradientTolerance
            ) {
                writePixel(output, idx, colorKey, 0);
                return;
            }
            if (
                (idx / 3) / width != 0 &&
                checkPixelEquality(input, idx - (width * 3), colorKey) == 1 && 
                calcColorDiff(input, idx, original, idx - (width * 3), colorSpace) > gradientTolerance
            ) {
                writePixel(output, idx, colorKey, 0);
                return;
            }
            if (
                (idx / 3) / width != height - 1 &&
                checkPixelEquality(input, idx + (width * 3), colorKey) == 1 &&
                calcColorDiff(input, idx, original, idx + (width * 3), colorSpace) > gradientTolerance
            ) {
                writePixel(output, idx, colorKey, 0);
                return;
            }
            writePixel(output, idx, original, idx);
        } else {
            writePixel(output, idx, colorKey, 0);
        }
    } else {
        writePixel(output, idx, colorKey, 0);
    }
}