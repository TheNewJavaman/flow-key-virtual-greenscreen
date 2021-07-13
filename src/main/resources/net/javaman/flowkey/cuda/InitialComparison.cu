extern "C" __global__ void initialComparisonKernel(
    int n,
    char *input,
    char *output,
    char *colorKey,
    char *replacementKey,
    float percentTolerance,
    int colorSpace
) {
    int idx = blockIdx.x * blockDim.x * threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        float percentDiff = calcColorDiff(input, idx, colorKey, 0, colorSpace);
        if (percentDiff < percentTolerance) {
            writePixel(output, idx, replacementKey, 0);
        } else {
            writePixel(output, idx, input, idx);
        }
    }
}