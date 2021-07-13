package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.ColorSpace
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi.Companion.ClMemOperation
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof
import kotlin.math.ceil

class InitialComparisonFilter constructor(
    private val api: OpenClApi,
    private val colorKey: ByteArray = byteArrayOf(0, 255.toByte(), 0),
    private val replacementKey: ByteArray = byteArrayOf(0, 255.toByte(), 0),
    private val percentTolerance: Float = 0.025f,
    private val colorSpace: ColorSpace = ColorSpace.ALL,
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS
) : AbstractFilter{
    companion object : AbstractFilter.AbstractFilterConsts {
        override val LIST_NAME = "Initial Comparison"

        private const val KERNEL_NAME = "initialComparisonKernel"
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        val outputBuffer = ByteArray(size = inputBuffer.size)
        val floatOptionsBuffer = floatArrayOf(percentTolerance)
        val intOptionsBuffer = intArrayOf(colorSpace.i, width, height)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val colorKeyPtr = Pointer.to(colorKey)
        val replacementKeyPtr = Pointer.to(replacementKey)
        val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = api.allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * inputBuffer.size)
        val outputMem = api.allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * outputBuffer.size)
        val colorKeyMem = api.allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * colorKey.size)
        val replacementKeyMem = api.allocMem(
            replacementKeyPtr,
            ClMemOperation.READ,
            Sizeof.cl_char * replacementKey.size
        )
        val floatOptionsMem = api.allocMem(
            floatOptionsPtr,
            ClMemOperation.READ,
            Sizeof.cl_float * floatOptionsBuffer.size
        )
        val intOptionsMem = api.allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(api.program, KERNEL_NAME, null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(replacementKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = api.localWorkSize?.let {
            longArrayOf(ceil(inputBuffer.size / it.toFloat()).toLong() * it)
        } ?: longArrayOf(inputBuffer.size.toLong())
        val localWorkSizeBuffer = api.localWorkSize?.let { longArrayOf(api.localWorkSize) }

        clEnqueueNDRangeKernel(
            api.commandQueue,
            kernel,
            1,
            null,
            globalWorkSizeBuffer,
            localWorkSizeBuffer,
            0,
            null,
            null
        )
        clEnqueueReadBuffer(
            api.commandQueue,
            outputMem,
            CL_TRUE,
            0,
            (inputBuffer.size * Sizeof.cl_char).toLong(),
            outputPtr,
            0,
            null,
            null
        )

        clReleaseMemObject(inputMem)
        clReleaseMemObject(outputMem)
        clReleaseMemObject(colorKeyMem)
        clReleaseMemObject(replacementKeyMem)
        clReleaseMemObject(floatOptionsMem)
        clReleaseMemObject(intOptionsMem)
        clReleaseKernel(kernel)

        return outputBuffer
    }
}