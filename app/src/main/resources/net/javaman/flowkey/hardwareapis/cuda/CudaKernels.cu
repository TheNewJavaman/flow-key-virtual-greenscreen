enum ColorSpace {
    BLUE = 0,
    GREEN = 1,
    RED = 2,
    ALL = 3
};

__device__ int checkPixelColorEquality(
        unsigned char *input,
        int i,
        unsigned char *colorKey
) {
    int diffSum = 0;
    for (int j = 0; j < 3; j++) {
        unsigned char a = input[i + j];
        unsigned char b = colorKey[j];
        diffSum += (a > b) ? a - b : b - a;
    }
    if (diffSum == 0) {
        return 1;
    } else {
        return 0;
    }
}

__device__ float calcColorDiff(
        unsigned char *a,
        int i,
        unsigned char *b,
        int j,
        int colorSpace
) {
    int colorDiff[3];
    for (int k = 0; k < 3; k++) {
        unsigned char c = a[i + k];
        unsigned char d = b[j + k];
        colorDiff[k] = (c > d) ? c - d : d - c;
    }
    if (colorSpace < ALL) {
        return colorDiff[colorSpace];
    } else {
        int sum = 0;
        for (int k = 0; k < 3; k++) {
            sum += colorDiff[k];
        }
        return sum / 3;
    }
}

__device__ void writePixel(
        unsigned char *canvas,
        int i,
        unsigned char *ink,
        int j
) {
    for (int k = 0; k < 3; k++) {
        canvas[i + k] = ink[j + k];
    }
}

__device__ void swapCharArrays(
        unsigned char *&a,
        unsigned char *&b
) {
    unsigned char *tmp = a;
    a = b;
    b = tmp;
}

extern "C" __global__ void initialComparisonKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *colorKey,
        unsigned char *replacementKey,
        int tolerance,
        int colorSpace
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        int diff = calcColorDiff(input, idx, colorKey, 0, colorSpace);
        if (diff < tolerance) {
            writePixel(output, idx, replacementKey, 0);
        } else {
            writePixel(output, idx, input, idx);
        }
    }
}

extern "C" __global__ void noiseReductionWorkerKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *original,
        unsigned char *colorKey,
        int width,
        int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelColorEquality(input, idx, colorKey) == 1) {
            int surroundingPixels = 0;
            if ((idx / 3) % width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx - 3, colorKey);
            }
            if ((idx / 3) % width == width - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx + 3, colorKey);
            }
            if ((idx / 3) / width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx - (width * 3), colorKey);
            }
            if ((idx / 3) / width == height - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx + (width * 3), colorKey);
            }
            if (surroundingPixels < 4) {
                writePixel(output, idx, original, idx);
            } else {
                writePixel(output, idx, colorKey, 0);
            }
        } else {
            writePixel(output, idx, original, idx);
        }
    }
}

extern "C" __global__ void noiseReductionKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *original,
        unsigned char *colorKey,
        int width,
        int height,
        int iterations,
        int blockSize,
        int gridSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < iterations; i++) {
            noiseReductionWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    output,
                    original,
                    colorKey,
                    width,
                    height
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
    }
}

extern "C" __global__ void flowKeyWorkerKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *original,
        unsigned char *replacementKey,
        int tolerance,
        int colorSpace,
        int width,
        int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelColorEquality(input, idx, replacementKey) == 0) {
            if ((idx / 3) % width != 0 &&
                checkPixelColorEquality(input, idx - 3, replacementKey) &&
                calcColorDiff(input, idx, original, idx - 3, colorSpace) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) % width != width - 1 &&
                checkPixelColorEquality(input, idx + 3, replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx + 3, colorSpace) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) / width != 0 &&
                checkPixelColorEquality(input, idx - (width * 3), replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx - (width * 3), colorSpace) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) / width != height - 1 &&
                checkPixelColorEquality(input, idx + (width * 3), replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx + (width * 3), colorSpace) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            writePixel(output, idx, original, idx);
        } else {
            writePixel(output, idx, replacementKey, 0);
        }
    }
}

extern "C" __global__ void flowKeyKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *original,
        unsigned char *replacementKey,
        int tolerance,
        int colorSpace,
        int width,
        int height,
        int iterations,
        int blockSize,
        int gridSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < iterations; i++) {
            flowKeyWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    output,
                    original,
                    replacementKey,
                    tolerance,
                    colorSpace,
                    width,
                    height
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
    }
}