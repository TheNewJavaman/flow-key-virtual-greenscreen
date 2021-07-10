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
    private val percentTolerance: Float = 0.025f,
    private val colorKey: ByteArray = byteArrayOf(14, 44, 65),
    private val replacementKey: ByteArray = byteArrayOf(255.toByte(), 255.toByte(), 0),
    private val colorSpace: ColorSpace = ColorSpace.ALL,
    private val noiseReduction: Int = 10,
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    platformIndex: Int = 0,
    deviceType: Long = CL_DEVICE_TYPE_ALL,
    deviceIndex: Int = 0
) {
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
        logger.info { "Selected platform ${getPlatformName()}" }

        val numDevicesArray = IntArray(1)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        device = devices[deviceIndex]
        logger.info { "Selected device ${getDeviceName()}" }

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

    private fun getPlatformName(): String {
        val size = LongArray(1)
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, null, size)
        val buffer = ByteArray(size[0].toInt())
        clGetPlatformInfo(platform, CL_PLATFORM_NAME, buffer.size.toLong(), Pointer.to(buffer), null)
        return String(buffer, 0, buffer.size - 1)
    }

    private fun getDeviceName(): String {
        val size = LongArray(1)
        clGetDeviceInfo(device, CL_DEVICE_NAME, 0, null, size)
        val buffer = ByteArray(size[0].toInt())
        clGetDeviceInfo(device, CL_DEVICE_NAME, buffer.size.toLong(), Pointer.to(buffer), null)
        return String(buffer, 0, buffer.size - 1)
    }

    fun apply(input: Mat): BufferedImage {
        val size = input.cols() * input.rows() * COLOR_DEPTH

        val inputBuffer = ByteArray(size = size)
        val outputBuffer = ByteArray(size = size)
        val floatOptionsBuffer = floatArrayOf(percentTolerance)
        val intOptionsBuffer = intArrayOf(colorSpace.i, noiseReduction)

        input.get(0, 0, inputBuffer)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val colorKeyPtr = Pointer.to(colorKey)
        val replacementKeyPtr = Pointer.to(replacementKey)
        val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_char * size).toLong(),
            inputPtr,
            null
        )
        val outputMem = clCreateBuffer(
            context,
            CL_MEM_READ_WRITE,
            (Sizeof.cl_char * size).toLong(),
            null,
            null
        )
        val colorKeyMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_char * COLOR_DEPTH).toLong(),
            colorKeyPtr,
            null
        )
        val replacementKeyMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_char * COLOR_DEPTH).toLong(),
            replacementKeyPtr,
            null
        )
        val floatOptionsMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_float * floatOptionsBuffer.size).toLong(),
            floatOptionsPtr,
            null
        )
        val intOptionsMem = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR,
            (Sizeof.cl_int * intOptionsBuffer.size).toLong(),
            intOptionsPtr,
            null
        )

        val kernel = clCreateKernel(program, "greenscreenKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(replacementKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
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
            outputMem,
            CL_TRUE,
            0,
            (size * Sizeof.cl_char).toLong(),
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

        val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val targetPixels = (bufferedImage.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(outputBuffer, 0, targetPixels, 0, input.cols() * input.rows() * COLOR_DEPTH)
        return bufferedImage
    }

    fun close() {
        clReleaseProgram(program)
        clReleaseCommandQueue(commandQueue)
        clReleaseContext(context)
    }

    enum class ColorSpace(val i: Int) {
        BLUE(0),
        RED(1),
        GREEN(2),
        ALL(3)
    }
}