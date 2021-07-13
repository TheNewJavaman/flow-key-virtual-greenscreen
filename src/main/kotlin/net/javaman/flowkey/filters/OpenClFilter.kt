// Adapted from https://github.com/gpu/JOCLSamples/blob/master/src/main/java/org/jocl/samples/JOCLSample.java
//              http://jocl.org/samples/JOCLDeviceQuery.java

package net.javaman.flowkey.filters

import net.javaman.flowkey.filters.Filter.Companion.ColorSpace
import net.javaman.flowkey.logger
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.DEFAULT_HEIGHT_PIXELS
import net.javaman.flowkey.util.DEFAULT_WIDTH_PIXELS
import org.jocl.*
import org.jocl.CL.*
import kotlin.math.ceil

@Suppress("Unused")
class OpenClFilter constructor(
    private val percentTolerance: Float = 0.025f,
    private val gradientTolerance: Float = 0.03f,
    private val colorKey: ByteArray = byteArrayOf(28, 56, 75),
    private val replacementKey: ByteArray = byteArrayOf(255.toByte(), 255.toByte(), 0),
    private val colorSpace: ColorSpace = ColorSpace.ALL,
    private val noiseReduction: Int = 1,
    private val flowDepth: Int = 1,
    private val width: Int = DEFAULT_WIDTH_PIXELS,
    private val height: Int = DEFAULT_HEIGHT_PIXELS,
    platformIndex: Int = 0,
    deviceType: Long = CL_DEVICE_TYPE_ALL,
    deviceIndex: Int = 0,
    private val localWorkSize: Long? = null,
    private val blockSize: Int = 1
) : Filter {
    companion object {
        enum class ClMemOperation(val flags: Long) {
            READ(CL_MEM_READ_ONLY or CL_MEM_COPY_HOST_PTR),

            //READ(CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR),
            WRITE(CL_MEM_WRITE_ONLY)
        }
    }

    private var platform: cl_platform_id?

    private var contextProperties: cl_context_properties

    private var device: cl_device_id?

    private var context: cl_context

    private var commandQueue: cl_command_queue

    private var program: cl_program

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

        val utilSource = this::class.java.getResource("../opencl/Util.cl")!!.readText()
        val initialComparisonSource = this::class.java.getResource("../opencl/InitialComparison.cl")!!.readText()
        val noiseReductionSource = this::class.java.getResource("../opencl/NoiseReduction.cl")!!.readText()
        val flowKeySource = this::class.java.getResource("../opencl/FlowKey.cl")!!.readText()
        val splashSource = this::class.java.getResource("../opencl/Splash.cl")!!.readText()
        val splashPrepSource = this::class.java.getResource("../opencl/SplashPrep.cl")!!.readText()
        val sources = arrayOf(
            utilSource,
            initialComparisonSource,
            noiseReductionSource,
            flowKeySource,
            splashSource,
            splashPrepSource
        )
        program = clCreateProgramWithSource(
            context,
            sources.size,
            sources,
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

    fun apply(originalImage: ByteArray): ByteArray {
        /*val initialComparisonOutput = applyInitialComparison(originalImage)
        val noiseReduction1Output = applyNoiseReductionHandler(initialComparisonOutput, originalImage)
        val flowKeyOutput = applyFlowKeyHandler(noiseReduction1Output, originalImage)
        return applyNoiseReductionHandler(flowKeyOutput, originalImage)*/
        //val prepOutput = applySplashPrep(originalImage)
        //return applySplash(originalImage, prepOutput)
        return originalImage
    }

    private fun applyInitialComparison(inputBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size
        val outputBuffer = ByteArray(size = size)
        val floatOptionsBuffer = floatArrayOf(percentTolerance)
        val intOptionsBuffer = intArrayOf(colorSpace.i, width, height)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val colorKeyPtr = Pointer.to(colorKey)
        val replacementKeyPtr = Pointer.to(replacementKey)
        val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val outputMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * size)
        val colorKeyMem = allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * COLOR_DEPTH)
        val replacementKeyMem = allocMem(replacementKeyPtr, ClMemOperation.READ, Sizeof.cl_char * COLOR_DEPTH)
        val floatOptionsMem = allocMem(floatOptionsPtr, ClMemOperation.READ, Sizeof.cl_float * floatOptionsBuffer.size)
        val intOptionsMem = allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(program, "initialComparisonKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(replacementKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = localWorkSize?.let { longArrayOf(ceil(size / it.toFloat()).toLong() * it) }
            ?: longArrayOf(size.toLong())
        val localWorkSizeBuffer = localWorkSize?.let { longArrayOf(localWorkSize) }

        clEnqueueNDRangeKernel(
            commandQueue,
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

        return outputBuffer
    }

    private fun applyNoiseReductionHandler(
        inputBuffer: ByteArray,
        templateBuffer: ByteArray,
        repeat: Int = noiseReduction
    ): ByteArray {
        var outputBuffer = inputBuffer.copyOf()
        for (i in 1..repeat) {
            outputBuffer = applyNoiseReduction(outputBuffer, templateBuffer)
        }
        return outputBuffer
    }

    private fun applyNoiseReduction(inputBuffer: ByteArray, templateBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size
        val outputBuffer = ByteArray(size = size)
        val intOptionsBuffer = intArrayOf(0, width, height)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val templatePtr = Pointer.to(templateBuffer)
        val colorKeyPtr = Pointer.to(replacementKey)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val outputMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * size)
        val templateMem = allocMem(templatePtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val colorKeyMem = allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * COLOR_DEPTH)
        val intOptionsMem = allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(program, "noiseReductionKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(templateMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = localWorkSize?.let { longArrayOf(ceil(size / it.toFloat()).toLong() * it) }
            ?: longArrayOf(size.toLong())
        val localWorkSizeBuffer = localWorkSize?.let { longArrayOf(localWorkSize) }

        clEnqueueNDRangeKernel(
            commandQueue,
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
            commandQueue,
            outputMem,
            CL_TRUE,
            0,
            (Sizeof.cl_char * size).toLong(),
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

        return outputBuffer
    }

    private fun applyFlowKeyHandler(
        inputBuffer: ByteArray,
        templateBuffer: ByteArray,
        repeat: Int = flowDepth
    ): ByteArray {
        var outputBuffer = inputBuffer.copyOf()
        for (i in 1..repeat) {
            outputBuffer = applyFlowKey(outputBuffer, templateBuffer)
        }
        return outputBuffer
    }

    private fun applyFlowKey(inputBuffer: ByteArray, templateBuffer: ByteArray): ByteArray {
        val size = inputBuffer.size
        val outputBuffer = ByteArray(size = size)
        val floatOptionsBuffer = floatArrayOf(0.0f, gradientTolerance)
        val intOptionsBuffer = intArrayOf(colorSpace.i, width, height)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val templatePtr = Pointer.to(templateBuffer)
        val colorKeyPtr = Pointer.to(replacementKey)
        val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val outputMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * size)
        val templateMem = allocMem(templatePtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val colorKeyMem = allocMem(colorKeyPtr, ClMemOperation.READ, Sizeof.cl_char * COLOR_DEPTH)
        val floatOptionsMem = allocMem(floatOptionsPtr, ClMemOperation.READ, Sizeof.cl_float * floatOptionsBuffer.size)
        val intOptionsMem = allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(program, "flowKeyKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(templateMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(colorKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = localWorkSize?.let { longArrayOf(ceil(size / it.toFloat()).toLong() * it) }
            ?: longArrayOf(size.toLong())
        val localWorkSizeBuffer = localWorkSize?.let { longArrayOf(localWorkSize) }

        clEnqueueNDRangeKernel(
            commandQueue,
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
            commandQueue,
            outputMem,
            CL_TRUE,
            0,
            (Sizeof.cl_char * size).toLong(),
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

        return outputBuffer
    }

    fun applySplash(inputBuffer: ByteArray, inputBlockAverageBuffer: FloatArray): Pair<ByteArray, FloatArray> {
        val size = inputBuffer.size
        val outputBuffer = ByteArray(size = size)
        val outputBlockAverageBuffer = FloatArray(size = inputBlockAverageBuffer.size)
        val floatOptionsBuffer = floatArrayOf(percentTolerance)
        val intOptionsBuffer = intArrayOf(0, width, height, blockSize)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val inputBlockAveragePtr = Pointer.to(inputBlockAverageBuffer)
        val outputBlockAveragePtr = Pointer.to(outputBlockAverageBuffer)
        val replacementKeyPtr = Pointer.to(replacementKey)
        val floatOptionsPtr = Pointer.to(floatOptionsBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val outputMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_char * size)
        val inputBlockAverageMem = allocMem(inputBlockAveragePtr, ClMemOperation.READ, Sizeof.cl_float * inputBlockAverageBuffer.size)
        val outputBlockAverageMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_float * outputBlockAverageBuffer.size)
        val replacementKeyMem = allocMem(replacementKeyPtr, ClMemOperation.READ, Sizeof.cl_char * COLOR_DEPTH)
        val floatOptionsMem = allocMem(floatOptionsPtr, ClMemOperation.READ, Sizeof.cl_float * floatOptionsBuffer.size)
        val intOptionsMem = allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(program, "splashKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputBlockAverageMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputBlockAverageMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(replacementKeyMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(floatOptionsMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = localWorkSize?.let { longArrayOf(ceil(size / it.toFloat()).toLong() * it) }
            ?: longArrayOf(size.toLong())
        val localWorkSizeBuffer = localWorkSize?.let { longArrayOf(localWorkSize) }

        clEnqueueNDRangeKernel(
            commandQueue,
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
            commandQueue,
            outputMem,
            CL_TRUE,
            0,
            (Sizeof.cl_char * size).toLong(),
            outputPtr,
            0,
            null,
            null
        )
        clEnqueueReadBuffer(
            commandQueue,
            outputBlockAverageMem,
            CL_TRUE,
            0,
            (Sizeof.cl_float * outputBlockAverageBuffer.size).toLong(),
            outputBlockAveragePtr,
            0,
            null,
            null
        )

        clReleaseMemObject(inputMem)
        clReleaseMemObject(outputMem)
        clReleaseMemObject(inputBlockAverageMem)
        clReleaseMemObject(outputBlockAverageMem)
        clReleaseMemObject(replacementKeyMem)
        clReleaseMemObject(floatOptionsMem)
        clReleaseMemObject(intOptionsMem)
        clReleaseKernel(kernel)

        return Pair(outputBuffer, outputBlockAverageBuffer)
    }

    fun applySplashPrep(inputBuffer: ByteArray): FloatArray {
        val size = inputBuffer.size
        val blocksPerRow = if (width % blockSize == 0) {
            width / blockSize
        } else {
            width / blockSize + 1
        }
        val blocksPerColumn = if (height % blockSize == 0) {
            height / blockSize
        } else {
            height / blockSize + 1
        }
        println("blocks per row: $blocksPerRow; blocks per column: $blocksPerColumn")
        val outputBuffer = FloatArray(size = blocksPerRow * blocksPerColumn)
        val intOptionsBuffer = intArrayOf(0, width, height, blockSize)

        val inputPtr = Pointer.to(inputBuffer)
        val outputPtr = Pointer.to(outputBuffer)
        val intOptionsPtr = Pointer.to(intOptionsBuffer)

        val inputMem = allocMem(inputPtr, ClMemOperation.READ, Sizeof.cl_char * size)
        val outputMem = allocMem(null, ClMemOperation.WRITE, Sizeof.cl_float * blocksPerRow * blocksPerColumn)
        val intOptionsMem = allocMem(intOptionsPtr, ClMemOperation.READ, Sizeof.cl_int * intOptionsBuffer.size)

        val kernel = clCreateKernel(program, "splashPrepKernel", null)
        var a = 0
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(inputMem))
        clSetKernelArg(kernel, a++, Sizeof.cl_mem.toLong(), Pointer.to(outputMem))
        clSetKernelArg(kernel, a, Sizeof.cl_mem.toLong(), Pointer.to(intOptionsMem))
        val globalWorkSizeBuffer = localWorkSize?.let { longArrayOf(ceil(size / it.toFloat()).toLong() * it) }
            ?: longArrayOf(size.toLong())
        val localWorkSizeBuffer = localWorkSize?.let { longArrayOf(localWorkSize) }

        clEnqueueNDRangeKernel(
            commandQueue,
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
            commandQueue,
            outputMem,
            CL_TRUE,
            0,
            (Sizeof.cl_float * blocksPerRow * blocksPerColumn).toLong(),
            outputPtr,
            0,
            null,
            null
        )

        clReleaseMemObject(inputMem)
        clReleaseMemObject(outputMem)
        clReleaseMemObject(intOptionsMem)
        clReleaseKernel(kernel)

        return outputBuffer
    }


    private fun allocMem(ptr: Pointer?, op: ClMemOperation, size: Int) = clCreateBuffer(
        context,
        op.flags,
        size.toLong(),
        ptr,
        null
    )

    fun close() {
        clReleaseProgram(program)
        clReleaseCommandQueue(commandQueue)
        clReleaseContext(context)
    }
}