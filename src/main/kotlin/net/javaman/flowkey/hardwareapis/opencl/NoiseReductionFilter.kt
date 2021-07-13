package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi.Companion.ClMemOperation
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof
import kotlin.math.ceil

class NoiseReductionFilter constructor(
    private val api: OpenClApi = OpenClApi(),
    private val iterations: Int = 5,
    private val colorKey: ByteArray = byteArrayOf(0, 255.toByte(), 0),
    private val replacementKey: ByteArray = byteArrayOf(0, 255.toByte(), 0),
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    private val templateBuffer: ByteArray = ByteArray(size = width * height * COLOR_DEPTH)
) : AbstractFilter {
    companion object : AbstractFilter.AbstractFilterConsts {
        override val LIST_NAME = "Noise Reduction"

        private const val KERNEL_NAME = "noiseReductionKernel"
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        var mutableInputBuffer = inputBuffer.clone()

        for (i in 0 until iterations) {
            val outputBuffer = ByteArray(size = mutableInputBuffer.size)
            val intOptionsBuffer = intArrayOf(0, width, height)

            val inputPtr = Pointer.to(mutableInputBuffer)
            val outputPtr = Pointer.to(outputBuffer)
            val templatePtr = Pointer.to(templateBuffer)
            val colorKeyPtr = Pointer.to(replacementKey)
            val intOptionsPtr = Pointer.to(intOptionsBuffer)

            val inputMem = api.allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * mutableInputBuffer.size)
            val outputMem = api.allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * outputBuffer.size)
            val templateMem = api.allocMem(templatePtr, ClMemOperation.READ, Sizeof.cl_char * templateBuffer.size)
            val colorKeyMem = api.allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * colorKey.size)
            val intOptionsMem = api.allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

            val kernel = clCreateKernel(api.program, KERNEL_NAME, null)
            var a = 0
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(templateMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
            clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
            val globalWorkSizeBuffer = api.localWorkSize?.let {
                longArrayOf(ceil(mutableInputBuffer.size / it.toFloat()).toLong() * it)
            } ?: longArrayOf(mutableInputBuffer.size.toLong())
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
                (mutableInputBuffer.size * Sizeof.cl_char).toLong(),
                outputPtr,
                0,
                null,
                null
            )

            clReleaseMemObject(inputMem)
            clReleaseMemObject(outputMem)
            clReleaseMemObject(templateMem)
            clReleaseMemObject(colorKeyMem)
            clReleaseMemObject(intOptionsMem)
            clReleaseKernel(kernel)

            mutableInputBuffer = outputBuffer
        }

        return mutableInputBuffer
    }
}