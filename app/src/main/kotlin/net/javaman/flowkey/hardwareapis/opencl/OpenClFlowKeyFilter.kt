@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi.Companion.ClMemOperation
import net.javaman.flowkey.util.*
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof
import kotlin.math.ceil

class OpenClFlowKeyFilter @Suppress("LongParameterList") constructor(
    private val api: OpenClApi = OpenClApi(),
    var iterations: Int = 3,
    var colorKey: ByteArray = DEFAULT_COLOR,
    var gradientTolerance: Float = DEFAULT_TOLERANCE,
    var width: Int = DEFAULT_WIDTH_PIXELS,
    var height: Int = DEFAULT_HEIGHT_PIXELS,
    var templateBuffer: ByteArray = ByteArray(size = width * height * COLOR_DEPTH)
) : AbstractFilter {
    companion object : AbstractFilterConsts {
        override val listName = "Flow Key"

        private const val KERNEL_NAME = "flowKeyKernel"
    }

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.TOLERANCE to gradientTolerance,
        FilterProperty.ITERATIONS to iterations,
        FilterProperty.REPLACEMENT_KEY to colorKey,
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.TOLERANCE.listName -> gradientTolerance = newValue as Float
        FilterProperty.ITERATIONS.listName -> iterations = newValue as Int
        FilterProperty.REPLACEMENT_KEY.listName -> colorKey = newValue as ByteArray
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    @Suppress("LongMethod")
    override fun apply(inputBuffer: ByteArray): ByteArray {
        var mutableInputBuffer = inputBuffer.clone()

        for (i in 0 until iterations) {
            val outputBuffer = ByteArray(size = mutableInputBuffer.size)
            val floatOptionsBuffer = floatArrayOf(0.0f, gradientTolerance)
            val intOptionsBuffer = intArrayOf(width, height)

            val inputPtr = Pointer.to(mutableInputBuffer)
            val outputPtr = Pointer.to(outputBuffer)
            val templatePtr = Pointer.to(templateBuffer)
            val colorKeyPtr = Pointer.to(colorKey)
            val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
            val intOptionsPtr = Pointer.to(intOptionsBuffer)

            val inputMem = api.allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * mutableInputBuffer.size)
            val outputMem = api.allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * outputBuffer.size)
            val templateMem = api.allocMem(templatePtr, ClMemOperation.READ, Sizeof.cl_char * templateBuffer.size)
            val colorKeyMem = api.allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * colorKey.size)
            val floatOptionsMem = api.allocMem(
                floatOptionsPtr,
                ClMemOperation.READ,
                Sizeof.cl_float * floatOptionsBuffer.size
            )
            val intOptionsMem = api.allocMem(
                intOptionsPtr,
                ClMemOperation.READ,
                Sizeof.cl_int * intOptionsBuffer.size
            )

            val kernel = clCreateKernel(api.program, KERNEL_NAME, null)
            var a = 0
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(templateMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
            clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
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
            clReleaseMemObject(floatOptionsMem)
            clReleaseMemObject(intOptionsMem)
            clReleaseKernel(kernel)

            mutableInputBuffer = outputBuffer
        }

        return mutableInputBuffer
    }
}
