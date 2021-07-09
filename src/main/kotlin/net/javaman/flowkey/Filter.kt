// Adapted from https://github.com/gpu/JOCLSamples/blob/master/src/main/java/org/jocl/samples/JOCLSample.java

package net.javaman.flowkey

import net.javaman.flowkey.Camera.Companion.COLOR_DEPTH
import net.javaman.flowkey.Camera.Companion.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.Camera.Companion.DEFAULT_WIDTH_PIXELS
import org.jocl.*
import org.jocl.CL.*
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte


class Filter constructor(
    percentTolerance: Float = 1.0f,
    private val originalKey: FloatArray = floatArrayOf(0.0f, 255.0f, 0.0f),
    private val replacementKey: FloatArray = floatArrayOf(0.0f, 255.0f, 0.0f),
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    private val platformIndex: Int = 0,
    private val deviceType: Long = CL_DEVICE_TYPE_ALL,
    private val deviceIndex: Int = 0
) {
    private val keyTolerance = percentTolerance * 255.0

    fun apply(input: Mat): BufferedImage {
        val size = input.cols() * input.rows() * COLOR_DEPTH
        val srcBuffer = ByteArray(size = size)
        val dstBuffer = ByteArray(size = size)
        val srcPtr = Pointer.to(srcBuffer)
        val dstPtr = Pointer.to(dstBuffer)
        input.get(0, 0, srcBuffer)

        // The platform, device type and device number
        // that will be used
        val platformIndex = 0
        val deviceType = CL.CL_DEVICE_TYPE_ALL
        val deviceIndex = 0

        // Enable exceptions and subsequently omit error checks in this sample
        CL.setExceptionsEnabled(true)

        // Obtain the number of platforms
        val numPlatformsArray = IntArray(1)
        CL.clGetPlatformIDs(0, null, numPlatformsArray)
        val numPlatforms = numPlatformsArray[0]

        // Obtain a platform ID
        val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
        CL.clGetPlatformIDs(platforms.size, platforms, null)
        val platform = platforms[platformIndex]

        // Initialize the context properties
        val contextProperties = cl_context_properties()
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM.toLong(), platform)

        // Obtain the number of devices for the platform
        val numDevicesArray = IntArray(1)
        CL.clGetDeviceIDs(platform, deviceType, 0, null, numDevicesArray)
        val numDevices = numDevicesArray[0]

        // Obtain a device ID
        val devices = arrayOfNulls<cl_device_id>(numDevices)
        CL.clGetDeviceIDs(platform, deviceType, numDevices, devices, null)
        val device = devices[deviceIndex]

        // Create a context for the selected device
        val context = CL.clCreateContext(
            contextProperties, 1, arrayOf(device),
            null, null, null
        )

        // Create a command-queue for the selected device
        val properties = cl_queue_properties()
        val commandQueue = CL.clCreateCommandQueueWithProperties(
            context, device, properties, null
        )

        // Allocate the memory objects for the input- and output data
        val srcMemA = CL.clCreateBuffer(
            context,
            CL.CL_MEM_READ_ONLY or CL.CL_MEM_COPY_HOST_PTR, (
                Sizeof.cl_char * size).toLong(), srcPtr, null
        )
        val dstMem = CL.clCreateBuffer(
            context,
            CL.CL_MEM_READ_WRITE, (
                Sizeof.cl_char * size).toLong(), null, null
        )

        // Create the program from the source code
        val program = CL.clCreateProgramWithSource(
            context,
            1, arrayOf(this::class.java.getResource("Filter.cl")!!.readText()), null, null
        )

        // Build the program
        CL.clBuildProgram(program, 0, null, null, null, null)

        // Create the kernel
        val kernel = CL.clCreateKernel(program, "sampleKernel", null)

        // Set the arguments for the kernel
        var a = 0
        CL.clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(srcMemA))
        CL.clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(dstMem))

        // Set the work-item dimensions
        val global_work_size = longArrayOf(size.toLong())

        // Execute the kernel
        CL.clEnqueueNDRangeKernel(
            commandQueue, kernel, 1, null,
            global_work_size, null, 0, null, null
        )

        // Read the output data
        CL.clEnqueueReadBuffer(
            commandQueue, dstMem, CL.CL_TRUE, 0, (
                size * Sizeof.cl_char).toLong(), dstPtr, 0, null, null
        )

        // Release kernel, program, and memory objects
        CL.clReleaseMemObject(srcMemA)
        CL.clReleaseMemObject(dstMem)
        CL.clReleaseKernel(kernel)
        CL.clReleaseProgram(program)
        CL.clReleaseCommandQueue(commandQueue)
        CL.clReleaseContext(context)

        val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        val targetPixels = (bufferedImage.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(dstBuffer, 0, targetPixels, 0, input.cols() * input.rows() * COLOR_DEPTH)
        return bufferedImage
    }

    private fun allocByteArray(ptr: Pointer?, size: Int, flags: Long, context: cl_context) = clCreateBuffer(
        context,
        flags,
        Sizeof.cl_char.toLong() * size,
        ptr,
        null
    )

    fun close() {

    }
}