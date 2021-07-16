// Adapted from https://github.com/jcuda/jcuda-samples/blob/master/JCudaSamples/src/main/java/jcuda/driver/samples/JCudaVectorAdd.java

@file:Suppress("WildcardImport")

package net.javaman.flowkey.filters

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.*
import jcuda.driver.JCudaDriver.*
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import kotlin.math.ceil


class CudaFilter @Suppress("LongParameterList") constructor(
    private val percentTolerance: Float = 0.01f,
    private val gradientTolerance: Float = 0.03f,
    private val colorKey: ByteArray = byteArrayOf(0, 0, 0),
    private val replacementKey: ByteArray = byteArrayOf(255.toByte(), 255.toByte(), 0),
    private val colorSpace: ColorSpace = ColorSpace.ALL,
    private val noiseReduction: Int = 10,
    private val flowDepth: Int = 1,
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    private val blockSize: Int = 256
) {
    companion object {
        @Suppress("Unused", "MagicNumber")
        enum class ColorSpace(val i: Int) {
            BLUE(0),
            RED(1),
            GREEN(2),
            ALL(3)
        }

        @Suppress("Unused")
        enum class FloatOption(val i: Int) {
            PERCENT_TOLERANCE(0),
            GRADIENT_TOLERANCE(1)
        }

        @Suppress("Unused")
        enum class IntOption(val i: Int) {
            COLOR_SPACE(0),
            WIDTH(1),
            HEIGHT(2)
        }
    }

    private var initialComparisonProgram: CUfunction

    private var noiseReductionProgram: CUfunction

    private var flowKeyProgram: CUfunction

    init {
        setExceptionsEnabled(true)

        cuInit(0)
        val device = CUdevice()
        cuDeviceGet(device, 0)

        val context = CUcontext()
        cuCtxCreate(context, 0, device)

        val module = CUmodule()
        cuModuleLoad(module, "ptx_out/Source.ptx")

        initialComparisonProgram = CUfunction()
        cuModuleGetFunction(initialComparisonProgram, module, "initialComparisonKernel")
        noiseReductionProgram = CUfunction()
        cuModuleGetFunction(noiseReductionProgram, module, "noiseReductionKernel")
        flowKeyProgram = CUfunction()
        cuModuleGetFunction(flowKeyProgram, module, "flowKeyKernel")
    }

    fun apply(originalImage: ByteArray): ByteArray {
        val initialComparisonOutput = applyInitialComparison(originalImage)
        val noiseReduction1Output = applyNoiseReductionHandler(initialComparisonOutput, originalImage)
        val flowKeyOutput = applyFlowKeyHandler(noiseReduction1Output, originalImage)
        return applyNoiseReductionHandler(flowKeyOutput, originalImage)
    }

    private fun applyInitialComparison(inputBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size

        val inputPtr = allocMem(Sizeof.CHAR * size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = allocMem(Sizeof.CHAR * size.toLong())
        val colorKeyPtr = allocMem(Sizeof.CHAR * COLOR_DEPTH.toLong(), Pointer.to(colorKey))
        val replacementKeyPtr = allocMem(Sizeof.CHAR * COLOR_DEPTH.toLong(), Pointer.to(replacementKey))

        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(colorKeyPtr),
            Pointer.to(replacementKeyPtr),
            Pointer.to(floatArrayOf(percentTolerance)),
            Pointer.to(intArrayOf(colorSpace.i))
        )

        val gridSize = ceil(size / blockSize.toDouble()).toInt()
        cuLaunchKernel(
            initialComparisonProgram,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val hostOutput = ByteArray(size = size)
        cuMemcpyDtoH(Pointer.to(hostOutput), outputPtr, Sizeof.FLOAT * size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(outputPtr)
        cuMemFree(colorKeyPtr)
        cuMemFree(replacementKeyPtr)

        return hostOutput
    }

    private fun applyNoiseReductionHandler(
        inputBuffer: ByteArray,
        templateBuffer: ByteArray,
        repeat: Int = noiseReduction
    ): ByteArray {
        var outputBuffer = inputBuffer.copyOf()
        for (i in 1..repeat) {
            outputBuffer = applyNoiseReduction(outputBuffer, templateBuffer)
        }
        return outputBuffer
    }

    private fun applyNoiseReduction(inputBuffer: ByteArray, templateBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size

        val inputPtr = allocMem(Sizeof.CHAR * size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = allocMem(Sizeof.CHAR * size.toLong())
        val templatePtr = allocMem(Sizeof.CHAR * size.toLong(), Pointer.to(templateBuffer))
        val colorKeyPtr = allocMem(Sizeof.CHAR * COLOR_DEPTH.toLong(), Pointer.to(replacementKey))

        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(templatePtr),
            Pointer.to(colorKeyPtr),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height))
        )

        val gridSize = ceil(size / blockSize.toDouble()).toInt()
        cuLaunchKernel(
            noiseReductionProgram,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val hostOutput = ByteArray(size = size)
        cuMemcpyDtoH(Pointer.to(hostOutput), outputPtr, Sizeof.FLOAT * size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(outputPtr)
        cuMemFree(templatePtr)
        cuMemFree(colorKeyPtr)

        return hostOutput
    }

    private fun applyFlowKeyHandler(
        inputBuffer: ByteArray,
        templateBuffer: ByteArray,
        repeat: Int = flowDepth
    ): ByteArray {
        var outputBuffer = inputBuffer.copyOf()
        for (i in 1..repeat) {
            outputBuffer = applyFlowKey(outputBuffer, templateBuffer)
        }
        return outputBuffer
    }

    private fun applyFlowKey(inputBuffer: ByteArray, templateBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size

        val inputPtr = allocMem(Sizeof.CHAR * size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = allocMem(Sizeof.CHAR * size.toLong())
        val templatePtr = allocMem(Sizeof.CHAR * size.toLong(), Pointer.to(templateBuffer))
        val colorKeyPtr = allocMem(Sizeof.CHAR * COLOR_DEPTH.toLong(), Pointer.to(replacementKey))

        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(templatePtr),
            Pointer.to(colorKeyPtr),
            Pointer.to(floatArrayOf(gradientTolerance)),
            Pointer.to(intArrayOf(colorSpace.i)),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height))
        )

        val gridSize = ceil(size / blockSize.toDouble()).toInt()
        cuLaunchKernel(
            flowKeyProgram,
            gridSize, 1, 1,
            blockSize, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val hostOutput = ByteArray(size = size)
        cuMemcpyDtoH(Pointer.to(hostOutput), outputPtr, Sizeof.FLOAT * size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(outputPtr)
        cuMemFree(templatePtr)
        cuMemFree(colorKeyPtr)

        return hostOutput
    }

    private fun allocMem(size: Long, hostPtr: Pointer? = null): CUdeviceptr {
        val devicePtr = CUdeviceptr()
        cuMemAlloc(devicePtr, size);
        hostPtr?.let {
            cuMemcpyHtoD(devicePtr, Pointer.to(hostPtr), size);
        }
        return devicePtr
    }
}
