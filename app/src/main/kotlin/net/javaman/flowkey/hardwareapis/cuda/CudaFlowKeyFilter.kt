@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver
import jcuda.driver.JCudaDriver.cuCtxSetCurrent
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.util.*
import kotlin.math.ceil

class CudaFlowKeyFilter constructor(
    private val api: CudaApi
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Flow Key"
    }

    lateinit var templateBuffer: ByteArray

    private var iterations = DEFAULT_ITERATIONS

    private var replacementKey = DEFAULT_COLOR

    private var tolerance = DEFAULT_TOLERANCE

    var width = DEFAULT_WIDTH_PIXELS

    var height = DEFAULT_HEIGHT_PIXELS

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.TOLERANCE to tolerance,
        FilterProperty.ITERATIONS to iterations,
        FilterProperty.REPLACEMENT_KEY to replacementKey
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        FilterProperty.REPLACEMENT_KEY.listName -> replacementKey = newValue as ByteArray
        FilterProperty.TOLERANCE.listName -> tolerance = newValue as Float
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
            Pointer.to(intArrayOf((tolerance * PIXEL_MULTIPLIER).toInt())),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height)),
            Pointer.to(intArrayOf(iterations)),
            Pointer.to(intArrayOf(CudaApi.BLOCK_SIZE)),
            Pointer.to(intArrayOf(gridSize))
        )

        JCudaDriver.cuLaunchKernel(
            api.flowKeyProgram,
            1, 1, 1,
            CudaApi.BLOCK_SIZE, 1, 1,
            0, null,
            kernelParams, null
        )
        JCudaDriver.cuCtxSynchronize()

        val outputBuffer = ByteArray(inputBuffer.size)
        JCudaDriver.cuMemcpyDtoH(Pointer.to(outputBuffer), outputPtr, Sizeof.BYTE * outputBuffer.size.toLong())

        JCudaDriver.cuMemFree(inputPtr)
        JCudaDriver.cuMemFree(outputPtr)
        JCudaDriver.cuMemFree(templatePtr)
        JCudaDriver.cuMemFree(replacementKeyPtr)

        return outputBuffer
    }
}
