__kernel void splashKernel(
    __global const char *input,
    __global char *output,
    __global const float *inputBlockAverages,
    __global float *outputBlockAverages,
    __global const char *replacementKey,
    __global const float *floatOptions,
    __global const int *intOptions
) {
    float percentTolerance = floatOptions[PERCENT_TOLERANCE];
    int width = intOptions[WIDTH];
    int height = intOptions[HEIGHT];
    int blockSize = intOptions[BLOCK_SIZE];
    int gid = get_global_id(0);
    if (gid % 3 == 0) {
        int n = gid / 3;
        int posX = n % width;
        int posY = n / width;
        if (posX % blockSize == 0 && posY % blockSize == 0) {
            int blockX = posX / blockSize;
            int blockY = posY / blockSize;
            int blocksPerRow = width / blockSize;
            if (width % blockSize != 0) {
                blocksPerRow += 1;
            }
            float blockAverage = avgBlock(input, width, height, blockSize, blockX, blockY);
            if (fabs(blockAverage - inputBlockAverages[blockY * blocksPerRow + blockX]) / 255.0 < percentTolerance) {
                for (int row = blockSize * blockY; row < blockY * blockSize + blockSize && row < height; row++) {
                    for (int col = blockSize * blockX; col < blockX * blockSize + blockSize && col < width; col++) {
                        writePixel(output, (row * width + col) * 3, replacementKey, 0);
                    }
                }
                outputBlockAverages[blockY * blocksPerRow + blockX] = blockAverage;
            } else {
                for (int row = blockSize * blockY; row < blockY * blockSize + blockSize && row < height; row++) {
                    for (int col = blockSize * blockX; col < blockX * blockSize + blockSize && col < width; col++) {
                        writePixel(output, (row * width + col) * 3, input, (row * width + col) * 3);
                    }
                }
                outputBlockAverages[blockY * blocksPerRow + blockX] = inputBlockAverages[blockY * blocksPerRow + blockX];
            }
        }
    }
}