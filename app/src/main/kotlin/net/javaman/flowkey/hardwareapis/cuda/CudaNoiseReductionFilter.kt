@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver.*
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.util.DEFAULT_COLOR
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_ITERATIONS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import kotlin.math.ceil

class CudaNoiseReductionFilter constructor(
    private val api: CudaApi
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Noise Reduction"
    }

    lateinit var templateBuffer: ByteArray

    private var replacementKey = DEFAULT_COLOR

    private var iterations = DEFAULT_ITERATIONS

    var width = DEFAULT_WIDTH_PIXELS

    var height = DEFAULT_HEIGHT_PIXELS

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.ITERATIONS to iterations,
        FilterProperty.REPLACEMENT_KEY to replacementKey
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.REPLACEMENT_KEY.listName -> replacementKey = newValue as ByteArray
        FilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        if (iterations == 0) {
            return inputBuffer
        }

        cuCtxSetCurrent(api.context)

        val inputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong())
        val templatePtr = api.allocMem(Sizeof.BYTE * templateBuffer.size.toLong(), Pointer.to(templateBuffer))
        val replacementKeyPtr = api.allocMem(Sizeof.BYTE * replacementKey.size.toLong(), Pointer.to(replacementKey))

        val gridSize = ceil(inputBuffer.size / CudaApi.BLOCK_SIZE.toDouble()).toInt()
        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(inputBuffer.size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(templatePtr),
            Pointer.to(replacementKeyPtr),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height)),
            Pointer.to(intArrayOf(iterations)),
            Pointer.to(intArrayOf(CudaApi.BLOCK_SIZE)),
            Pointer.to(intArrayOf(gridSize))
        )

        cuLaunchKernel(
            api.noiseReductionProgram,
            1, 1, 1,
            CudaApi.BLOCK_SIZE, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val outputBuffer = ByteArray(inputBuffer.size)
        cuMemcpyDtoH(Pointer.to(outputBuffer), outputPtr, Sizeof.BYTE * outputBuffer.size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(outputPtr)
        cuMemFree(templatePtr)
        cuMemFree(replacementKeyPtr)

        return outputBuffer
    }
}
