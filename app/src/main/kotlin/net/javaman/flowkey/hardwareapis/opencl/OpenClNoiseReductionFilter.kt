@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi.Companion.ClMemOperation
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.DEFAULT_COLOR
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof
import kotlin.math.ceil

class OpenClNoiseReductionFilter @Suppress("LongParameterList") constructor(
    private val api: OpenClApi = OpenClApi(),
    var iterations: Int = 5,
    var colorKey: ByteArray = DEFAULT_COLOR,
    var width: Int = DEFAULT_WIDTH_PIXELS,
    var height: Int = DEFAULT_HEIGHT_PIXELS,
    var templateBuffer: ByteArray = ByteArray(size = width * height * COLOR_DEPTH)
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Noise Reduction"

        private const val KERNEL_NAME = "noiseReductionKernel"
    }

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.ITERATIONS to iterations,
        FilterProperty.COLOR_KEY to colorKey,
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        FilterProperty.COLOR_KEY.listName -> colorKey = newValue as ByteArray
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    override fun apply(inputBuffer: ByteArray): ByteArray {
        var mutableInputBuffer = inputBuffer.clone()
        repeat(iterations) {
            val outputBuffer = ByteArray(size = mutableInputBuffer.size)
            val intOptionsBuffer = intArrayOf(0, width, height)

            val inputPtr = Pointer.to(mutableInputBuffer)
            val outputPtr = Pointer.to(outputBuffer)
            val templatePtr = Pointer.to(templateBuffer)
            val colorKeyPtr = Pointer.to(colorKey)
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
