__device__ int checkPixelColorEquality(
        unsigned char *input,
        int i,
        unsigned char *colorKey
) {
    for (int j = 0; j < 3; j++) {
        if (input[i + j] != colorKey[j]) {
            return 0;
        }
    }
    return 1;
}

__device__ float calcColorDiff(
        unsigned char *a,
        int i,
        unsigned char *b,
        int j
) {
    int sum = 0;
    for (int k = 0; k < 3; k++) {
        sum += abs(a[i + k] - b[j + k]);
    }
    return sum / 3;
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
        int tolerance
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        int diff = calcColorDiff(input, idx, colorKey, 0);
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
        unsigned char *replacementKey,
        int width,
        int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelColorEquality(input, idx, replacementKey) == 1) {
            int surroundingPixels = 0;
            if ((idx / 3) % width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx - 3, replacementKey);
            }
            if ((idx / 3) % width == width - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx + 3, replacementKey);
            }
            if ((idx / 3) / width == 0) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx - (width * 3), replacementKey);
            }
            if ((idx / 3) / width == height - 1) {
                surroundingPixels += 1;
            } else {
                surroundingPixels += checkPixelColorEquality(input, idx + (width * 3), replacementKey);
            }
            if (surroundingPixels < 3) {
                writePixel(output, idx, original, idx);
            } else {
                writePixel(output, idx, replacementKey, 0);
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
        unsigned char *replacementKey,
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
                    replacementKey,
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
        int width,
        int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelColorEquality(input, idx, replacementKey) == 0) {
            if ((idx / 3) % width != 0 &&
                checkPixelColorEquality(input, idx - 3, replacementKey) &&
                calcColorDiff(input, idx, original, idx - 3) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) % width != width - 1 &&
                checkPixelColorEquality(input, idx + 3, replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx + 3) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) / width != 0 &&
                checkPixelColorEquality(input, idx - (width * 3), replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx - (width * 3)) < tolerance) {
                writePixel(output, idx, replacementKey, 0);
                return;
            }
            if ((idx / 3) / width != height - 1 &&
                checkPixelColorEquality(input, idx + (width * 3), replacementKey) == 1 &&
                calcColorDiff(input, idx, original, idx + (width * 3)) < tolerance) {
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
                    width,
                    height
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
    }
}

extern "C" __global__ void gapFillerWorkerKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *replacementKey,
        int width,
        int height
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && idx % 3 == 0) {
        if (checkPixelColorEquality(input, idx, replacementKey) == 0) {
            int surroundingPixels = 0;
            if ((idx / 3) % width != 0) {
                surroundingPixels += checkPixelColorEquality(input, idx - 3, replacementKey);
            }
            if ((idx / 3) % width == width - 1) {
                surroundingPixels += checkPixelColorEquality(input, idx + 3, replacementKey);
            }
            if ((idx / 3) / width != 0) {
                surroundingPixels += checkPixelColorEquality(input, idx - (width * 3), replacementKey);
            }
            if ((idx / 3) / width != height - 1) {
                surroundingPixels += checkPixelColorEquality(input, idx + (width * 3), replacementKey);
            }
            if (surroundingPixels > 1) {
                writePixel(output, idx, replacementKey, 0);
            } else {
                writePixel(output, idx, input, idx);
            }
        } else {
            writePixel(output, idx, replacementKey, 0);
        }
    }
}

extern "C" __global__ void gapFillerKernel(
        int n,
        unsigned char *input,
        unsigned char *output,
        unsigned char *replacementKey,
        int width,
        int height,
        int iterations,
        int blockSize,
        int gridSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        for (int i = 0; i < iterations; i++) {
            gapFillerWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    output,
                    replacementKey,
                    width,
                    height
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
        swapCharArrays(input, output);
    }
}
