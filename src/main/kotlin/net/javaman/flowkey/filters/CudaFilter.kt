package net.javaman.flowkey.filters

import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver.*
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

class CudaFilter constructor(
    private val percentTolerance: Float = 0.01f,
    private val gradientTolerance: Float = 0.03f,
    private val colorKey: ByteArray = byteArrayOf(0, 0, 0),
    private val replacementKey: ByteArray = byteArrayOf(255.toByte(), 255.toByte(), 0),
    private val colorSpace: ColorSpace = ColorSpace.ALL,
    private val noiseReduction: Int = 10,
    private val flowDepth: Int = 1,
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
) {
    companion object {
        @Suppress("Unused")
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

    init {
        setExceptionsEnabled(true)
        cuInit(0)
        val device = CUdevice()
        cuDeviceGet(device, 0)
        val context = CUcontext()
        cuCtxCreate(context, 0, device)
        val module = CUmodule()
        cuModuleLoad(module, "")
    }

    fun apply(input: Mat): BufferedImage {
        val size = input.cols() * input.rows() * COLOR_DEPTH
        val originalImage = ByteArray(size = size)
        input.get(0, 0, originalImage)

        val initialComparisonOutput = applyInitialComparison(originalImage)
        val noiseReduction1Output = applyNoiseReductionHandler(initialComparisonOutput, originalImage)
        val flowKeyOutput = applyFlowKeyHandler(noiseReduction1Output, originalImage)
        val noiseReduction2Output = applyNoiseReductionHandler(flowKeyOutput, originalImage)

        val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val targetPixels = (bufferedImage.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(noiseReduction2Output, 0, targetPixels, 0, size)
        return bufferedImage
    }

    private fun applyInitialComparison(inputBuffer: ByteArray): ByteArray {
        return inputBuffer
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
        return inputBuffer
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
        return inputBuffer
    }

    fun close() {

    }
}