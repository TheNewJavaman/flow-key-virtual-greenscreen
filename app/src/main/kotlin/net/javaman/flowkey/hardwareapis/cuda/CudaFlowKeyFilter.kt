package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver.cuCtxSetCurrent
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.driver.JCudaDriver.cuLaunchKernel
import jcuda.driver.JCudaDriver.cuMemFree
import jcuda.driver.JCudaDriver.cuMemcpyDtoH
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.hardwareapis.cuda.CudaApi.Companion.BLOCK_SIZE
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_ITERATIONS
import net.javaman.flowkey.util.DEFAULT_TOLERANCE
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import net.javaman.flowkey.util.PIXEL_MULTIPLIER
import kotlin.math.ceil

class CudaFlowKeyFilter constructor(
    private val api: CudaApi
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Flow Key"
    }

    lateinit var originalBuffer: ByteArray

    private var tolerance = DEFAULT_TOLERANCE

    private var iterations = DEFAULT_ITERATIONS

    var width = DEFAULT_WIDTH_PIXELS

    var height = DEFAULT_HEIGHT_PIXELS

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.TOLERANCE to tolerance,
        FilterProperty.ITERATIONS to iterations
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        FilterProperty.TOLERANCE.listName -> tolerance = newValue as Float
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        if (iterations == 0) {
            return inputBuffer
        }

        cuCtxSetCurrent(api.context)

        val inputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong(), Pointer.to(inputBuffer))
        val originalPtr = api.allocMem(Sizeof.BYTE * originalBuffer.size.toLong(), Pointer.to(originalBuffer))
        val outputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong())

        val gridSize = ceil(inputBuffer.size / BLOCK_SIZE.toDouble()).toInt()
        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(inputBuffer.size)),
            Pointer.to(inputPtr),
            Pointer.to(originalPtr),
            Pointer.to(intArrayOf((tolerance * PIXEL_MULTIPLIER).toInt())),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height)),
            Pointer.to(outputPtr),
            Pointer.to(intArrayOf(iterations)),
            Pointer.to(intArrayOf(gridSize)),
            Pointer.to(intArrayOf(BLOCK_SIZE))
        )

        cuLaunchKernel(
            api.flowKeyProgram,
            1, 1, 1,
            BLOCK_SIZE, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val outputBuffer = ByteArray(inputBuffer.size)
        cuMemcpyDtoH(Pointer.to(outputBuffer), outputPtr, Sizeof.BYTE * outputBuffer.size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(originalPtr)
        cuMemFree(outputPtr)

        return outputBuffer
    }
}
