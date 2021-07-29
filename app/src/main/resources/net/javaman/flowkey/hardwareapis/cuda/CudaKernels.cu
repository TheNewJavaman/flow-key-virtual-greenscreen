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
        unsigned char *original,
        unsigned char *colorKey,
        int tolerance,
        unsigned char *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx] || calcColorDiff(original, idx * 3, colorKey, 0) < tolerance) {
            output[idx] = 0x01;
        }
    }
}

extern "C" __global__ void noiseReductionWorkerKernel(
        int n,
        unsigned char *input,
        int width,
        int height,
        unsigned char *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx]) {
            int surroundingPixels = 0;
            if (!(idx % width) || input[idx - 1]) {
                surroundingPixels += 1;
            }
            if (idx % width == width - 1 || input[idx + 1]) {
                surroundingPixels += 1;
            }
            if (!(idx / width) || input[idx - width]) {
                surroundingPixels += 1;
            }
            if (idx / width == height - 1 || input[idx + width]) {
                surroundingPixels += 1;
            }
            if (surroundingPixels > 2) {
                output[idx] = 0x01;
            } else {
                output[idx] = 0x00;
            }
        } else {
            output[idx] = 0x00;
        }
    }
}

extern "C" __global__ void noiseReductionKernel(
        int n,
        unsigned char *input,
        int width,
        int height,
        unsigned char *output,
        int iterations,
        int gridSize,
        int blockSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!idx) {
        for (int i = 0; i < iterations; i++) {
            noiseReductionWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    width,
                    height,
                    output
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
        swapCharArrays(input, output);
    }
}

extern "C" __global__ void flowKeyWorkerKernel(
        int n,
        unsigned char *input,
        unsigned char *original,
        int tolerance,
        int width,
        int height,
        unsigned char *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!input[idx]) {
            if (idx % width &&
                input[idx - 1] &&
                calcColorDiff(original, idx * 3, original, 3 * (idx - 1)) < tolerance) {
                output[idx] = 0x01;
                return;
            }
            if (idx % width != width - 1 &&
                input[idx + 1] &&
                calcColorDiff(original, idx * 3, original, 3 * (idx + 1)) < tolerance) {
                output[idx] = 0x01;
                return;
            }
            if (idx / width &&
                input[idx - width] &&
                calcColorDiff(original, idx * 3, original, 3 * (idx - width)) < tolerance) {
                output[idx] = 0x01;
                return;
            }
            if (idx / width != height - 1 &&
                input[idx + width] &&
                calcColorDiff(original, idx * 3, original, 3 * (idx + width)) < tolerance) {
                output[idx] = 0x01;
                return;
            }
        } else {
            output[idx] = 0x01;
        }
    }
}

extern "C" __global__ void flowKeyKernel(
        int n,
        unsigned char *input,
        unsigned char *original,
        int tolerance,
        int width,
        int height,
        unsigned char *output,
        int iterations,
        int gridSize,
        int blockSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!idx) {
        for (int i = 0; i < iterations; i++) {
            flowKeyWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    original,
                    tolerance,
                    width,
                    height,
                    output
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
        swapCharArrays(input, output);
    }
}

extern "C" __global__ void gapFillerWorkerKernel(
        int n,
        unsigned char *input,
        int width,
        int height,
        unsigned char *output
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (!input[idx]) {
            int surroundingPixels = 0;
            if (idx % width && input[idx - 1]) {
                surroundingPixels += 1;
            }
            if (idx % width != width - 1 && input[idx + 1]) {
                surroundingPixels += 1;
            }
            if (idx / width && input[idx - width]) {
                surroundingPixels += 1;
            }
            if (idx / width != height - 1 && input[idx + width]) {
                surroundingPixels += 1;
            }
            if (surroundingPixels > 1) {
                output[idx] = 0x01;
            }
        } else {
            output[idx] = 0x01;
        }
    }
}

extern "C" __global__ void gapFillerKernel(
        int n,
        unsigned char *input,
        int width,
        int height,
        unsigned char *output,
        int iterations,
        int gridSize,
        int blockSize
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (!idx) {
        for (int i = 0; i < iterations; i++) {
            gapFillerWorkerKernel<<<gridSize, blockSize>>>(
                    n,
                    input,
                    width,
                    height,
                    output
            );
            cudaDeviceSynchronize();
            swapCharArrays(input, output);
        }
        swapCharArrays(input, output);
    }
}

extern "C" __global__ void applyBitmapKernel(
        int n,
        unsigned char *input,
        unsigned char *original,
        unsigned char *replacementKey,
        unsigned char *modified
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (input[idx]) {
            memcpy(modified + 3 * idx, replacementKey, 3);
        } else {
            memcpy(modified + 3 * idx, original + 3 * idx, 3);
        }
    }
}
