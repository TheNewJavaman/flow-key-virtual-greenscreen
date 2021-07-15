__kernel void splashPrepKernel(
    __global const char *input,
    __global float *output,
    __global const int *intOptions
) {
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
            output[blockY * blocksPerRow + blockX] = avgBlock(input, width, height, blockSize, blockX, blockY);
        }
    }
}