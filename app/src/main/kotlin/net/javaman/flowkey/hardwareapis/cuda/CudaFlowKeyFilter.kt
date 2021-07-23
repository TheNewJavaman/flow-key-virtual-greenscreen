package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver
import jcuda.driver.JCudaDriver.cuCtxSetCurrent
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty
import net.javaman.flowkey.hardwareapis.common.ColorSpace
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

    private var colorSpace = ColorSpace.ALL

    private var width = DEFAULT_WIDTH_PIXELS

    private var height = DEFAULT_HEIGHT_PIXELS

    override fun getProperties(): Map<AbstractFilterProperty, Any> = mapOf(
        AbstractFilterProperty.TOLERANCE to tolerance,
        AbstractFilterProperty.ITERATIONS to iterations,
        AbstractFilterProperty.REPLACEMENT_KEY to replacementKey,
        AbstractFilterProperty.COLOR_SPACE to colorSpace
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        AbstractFilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        AbstractFilterProperty.REPLACEMENT_KEY.listName -> replacementKey = newValue as ByteArray
        AbstractFilterProperty.TOLERANCE.listName -> tolerance = newValue as Float
        AbstractFilterProperty.COLOR_SPACE.listName -> colorSpace = newValue as ColorSpace
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        cuCtxSetCurrent(api.context)

        val inputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong(), Pointer.to(inputBuffer))
        val outputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong())
        val templatePtr = api.allocMem(Sizeof.BYTE * templateBuffer.size.toLong(), Pointer.to(templateBuffer))
        val replacementKeyPtr = api.allocMem(Sizeof.BYTE * replacementKey.size.toLong(), Pointer.to(replacementKey))

        val gridSize = ceil(inputBuffer.size / api.blockSize.toDouble()).toInt()
        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(inputBuffer.size)),
            Pointer.to(inputPtr),
            Pointer.to(outputPtr),
            Pointer.to(templatePtr),
            Pointer.to(replacementKeyPtr),
            Pointer.to(floatArrayOf(tolerance)),
            Pointer.to(intArrayOf(colorSpace.i)),
            Pointer.to(intArrayOf(width)),
            Pointer.to(intArrayOf(height)),
            Pointer.to(intArrayOf(iterations)),
            Pointer.to(intArrayOf(api.blockSize)),
            Pointer.to(intArrayOf(gridSize))
        )

        JCudaDriver.cuLaunchKernel(
            api.flowKeyProgram,
            1, 1, 1,
            api.blockSize, 1, 1,
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