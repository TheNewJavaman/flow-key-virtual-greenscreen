// Adapted from https://github.com/gpu/JOCLSamples/blob/master/src/main/java/org/jocl/samples/JOCLSample.java
//              http://jocl.org/samples/JOCLDeviceQuery.java

package net.javaman.flowkey

import net.javaman.flowkey.Camera.Companion.COLOR_DEPTH
import net.javaman.flowkey.Camera.Companion.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.Camera.Companion.DEFAULT_WIDTH_PIXELS
import org.jocl.*
import org.jocl.CL.*
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.util.logging.Logger


class Filter constructor(
    percentTolerance: Float = 1.0f,
    private val originalKey: FloatArray = floatArrayOf(0.0f, 255.0f, 0.0f),
    private val replacementKey: FloatArray = floatArrayOf(0.0f, 255.0f, 0.0f),
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    platformIndex: Int = 0,
    deviceType: Long = CL_DEVICE_TYPE_ALL,
    deviceIndex: Int = 0
) {
    companion object {
        const val STRING_ALLOC_BYTES = 1024
    }

    private val keyTolerance = percentTolerance * 255.0

    private val programSource = this::class.java.getResource("Filter.cl")!!.readText()

    private var platform: cl_platform_id?

    private var contextProperties: cl_context_properties

    private var device: cl_device_id?

    private var context: cl_context

    private var commandQueue: cl_command_queue

    private var program: cl_program

    private val logger = Logger.getLogger(Filter::class.java.simpleName)

    init {
        setExceptionsEnabled(true)

        val numPlatformsArray = IntArray(1)
        clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        clGetPlatformIDs(platforms.size, platforms, null)
        platform = platforms[platformIndex]
        contextProperties = cl_context_properties()
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)
        logger.info { "Selected platform ${getPlatformInfo(CL_PLATFORM_VENDOR)} ${getPlatformInfo(CL_PLATFORM_NAME)}" }

        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        device = devices[deviceIndex]
        logger.info { "Selected device ${getDeviceInfo(CL_DEVICE_VENDOR)} ${getDeviceInfo(CL_DEVICE_NAME)}" }

        context = clCreateContext(contextProperties, 1, arrayOf(device), null, null, null)
        val properties = cl_queue_properties()
        commandQueue = clCreateCommandQueueWithProperties(context, device, properties, null)

        program = clCreateProgramWithSource(
            context,
            1,
            arrayOf(programSource),
            null,
            null
        )
        clBuildProgram(program, 0, null, null, null, null)
    }

    private fun getPlatformInfo(paramName: Int): String {
        val size = LongArray(1)
        clGetPlatformInfo(platform, paramName, 0, null, size)
        val buffer = ByteArray(size[0].toInt())
        clGetPlatformInfo(platform, paramName, buffer.size.toLong(), Pointer.to(buffer), null)
        return String(buffer, 0, buffer.size - 1)
    }

    private fun getDeviceInfo(paramName: Int): String {
        val size = LongArray(1)
        clGetDeviceInfo(device, paramName, 0, null, size)
        val buffer = ByteArray(size[0].toInt())
        clGetDeviceInfo(device, paramName, buffer.size.toLong(), Pointer.to(buffer), null)
        return String(buffer, 0, buffer.size - 1)
    }

    fun apply(input: Mat): BufferedImage {
        val size = input.cols() * input.rows() * COLOR_DEPTH
        val srcBuffer = ByteArray(size = size)
        val dstBuffer = ByteArray(size = size)
        val srcPtr = Pointer.to(srcBuffer)
        val dstPtr = Pointer.to(dstBuffer)
        input.get(0, 0, srcBuffer)

        val srcMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_char * size).toLong(),
            srcPtr,
            null
        )
        val dstMem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            (Sizeof.cl_char * size).toLong(),
            null,
            null
        )

        val kernel = clCreateKernel(program, "sampleKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(srcMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(dstMem))
        val globalWorkSize = longArrayOf(size.toLong())

        clEnqueueNDRangeKernel(
            commandQueue,
            kernel,
            1,
            null,
            globalWorkSize,
            null,
            0,
            null,
            null
        )
        clEnqueueReadBuffer(
            commandQueue,
            dstMem,
            CL_TRUE,
            0,
            (size * Sizeof.cl_char).toLong(),
            dstPtr,
            0,
            null,
            null
        )

        clReleaseMemObject(srcMem)
        clReleaseMemObject(dstMem)
        clReleaseKernel(kernel)

        val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val targetPixels = (bufferedImage.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(dstBuffer, 0, targetPixels, 0, input.cols() * input.rows() * COLOR_DEPTH)
        return bufferedImage
    }

    fun close() {
        clReleaseProgram(program)
        clReleaseCommandQueue(commandQueue)
        clReleaseContext(context)
    }
}