@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.common.AbstractFilterConsts
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi.Companion.ClMemOperation
import net.javaman.flowkey.stages.FilterProperty
import net.javaman.flowkey.util.DEFAULT_COLOR
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_TOLERANCE
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.jocl.CL.*
import org.jocl.Pointer
import org.jocl.Sizeof
import kotlin.math.ceil

class OpenClInitialComparisonFilter @Suppress("LongParameterList") constructor(
    private val api: OpenClApi,
    var colorKey: ByteArray = DEFAULT_COLOR,
    var replacementKey: ByteArray = DEFAULT_COLOR,
    var percentTolerance: Float = DEFAULT_TOLERANCE,
    var width: Int = DEFAULT_WIDTH_PIXELS,
    var height: Int = DEFAULT_HEIGHT_PIXELS
) : AbstractFilter{
    companion object : AbstractFilterConsts {
        override val listName = "Initial Comparison"

        private const val KERNEL_NAME = "initialComparisonKernel"
    }

    override fun getProperties(): Map<FilterProperty, Any> = mapOf(
        FilterProperty.TOLERANCE to percentTolerance,
        FilterProperty.COLOR_KEY to colorKey,
        FilterProperty.REPLACEMENT_KEY to replacementKey,
    )

    override fun setProperty(listName: String, newValue: Any) = when (listName) {
        FilterProperty.TOLERANCE.listName -> percentTolerance = newValue as Float
        FilterProperty.COLOR_KEY.listName -> colorKey = newValue as ByteArray
        FilterProperty.REPLACEMENT_KEY.listName -> replacementKey = newValue as ByteArray
        else -> throw ArrayIndexOutOfBoundsException("Couldn't find property $listName")
    }

    @Suppress("LongMethod")
    override fun apply(inputBuffer: ByteArray): ByteArray {
        val outputBuffer = ByteArray(size = inputBuffer.size)
        val floatOptionsBuffer = floatArrayOf(percentTolerance)
        val intOptionsBuffer = intArrayOf(width, height)

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
