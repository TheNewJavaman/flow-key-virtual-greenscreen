@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver.*
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.util.DEFAULT_COLOR
import net.javaman.flowkey.util.DEFAULT_TOLERANCE
import net.javaman.flowkey.util.PIXEL_MULTIPLIER
import kotlin.math.ceil

class CudaInitialComparisonFilter constructor(
    private val api: CudaApi
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Initial Comparison"
    }

    private var colorKey = DEFAULT_COLOR

    private var replacementKey = DEFAULT_COLOR

    private var percentTolerance = DEFAULT_TOLERANCE

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.TOLERANCE to percentTolerance,
        FilterProperty.COLOR_KEY to colorKey,
        FilterProperty.REPLACEMENT_KEY to replacementKey
    )

    override fun setProperty(listName: String, newValue: Any)= when (listName) {
        FilterProperty.TOLERANCE.listName -> percentTolerance = newValue as Float
        FilterProperty.COLOR_KEY.listName -> colorKey = newValue as ByteArray
        FilterProperty.REPLACEMENT_KEY.listName -> replacementKey = newValue as ByteArray
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        cuCtxSetCurrent(api.context)

        val inputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong())
        val colorKeyPtr = api.allocMem(Sizeof.BYTE * colorKey.size.toLong(), Pointer.to(colorKey))
        val replacementKeyPtr = api.allocMem(Sizeof.BYTE * replacementKey.size.toLong(), Pointer.to(replacementKey))

        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(inputBuffer.size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(colorKeyPtr),
            Pointer.to(replacementKeyPtr),
            Pointer.to(intArrayOf((percentTolerance * PIXEL_MULTIPLIER).toInt()))
        )

        val gridSize = ceil(inputBuffer.size / CudaApi.BLOCK_SIZE.toDouble()).toInt()
        cuLaunchKernel(
            api.initialComparisonProgram,
            gridSize, 1, 1,
            CudaApi.BLOCK_SIZE, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val outputBuffer = ByteArray(inputBuffer.size)
        cuMemcpyDtoH(Pointer.to(outputBuffer), outputPtr, Sizeof.BYTE * outputBuffer.size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(outputPtr)
        cuMemFree(colorKeyPtr)
        cuMemFree(replacementKeyPtr)

        return outputBuffer
    }
}
