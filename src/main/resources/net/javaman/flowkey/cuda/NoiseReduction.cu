extern "C" __global__ void noiseReductionKernel(
    int n,
    char *input,
    char *output,
    char *original,
    char *colorKey,
    int width,
    int height
) {
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        int anchorEquality = checkPixelEquality(input, idx, colorKey);
        if (anchorEquality == 1) {
            int surroundingPixels = 0;
            if ((idx / 3) % width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, idx - 3, colorKey);
            }
            if ((idx / 3) % width == width - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, idx + 3, colorKey);
            }
            if ((idx / 3) / width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, idx - (width * 3), colorKey);
            }
            if ((idx / 3) / width == height - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelEquality(input, idx + (width * 3), colorKey);
            }
            if (surroundingPixels < 3) {
                writePixel(output, idx, original, idx);
            } else {
                writePixel(output, idx, colorKey, 0);
            }
        } else {
            writePixel(output, idx, original, idx);
        }
    }
}