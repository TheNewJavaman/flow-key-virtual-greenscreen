@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.opencl

import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import org.jocl.*
import org.jocl.CL.*

class OpenClApi constructor(
    platformIndex: Int = 0,
    deviceIndex: Int = 0,
    val localWorkSize: Long? = null
) : AbstractApi {
    companion object : AbstractApi.AbstractApiConsts {
        override val listName = "OpenCl"

        enum class ClMemOperation(val flags: Long) {
            // CL_MEM_USE_HOST_PTR instead of CL_MEM_COPY_HOST_PTR speeds up most operations for realtime video
            READ(CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR),
            WRITE(CL_MEM_WRITE_ONLY)
        }

        private fun getPlatforms(): Array<cl_platform_id?> {
            val numPlatformsArray = IntArray(1)
            clGetPlatformIDs(0, null, numPlatformsArray)
            val numPlatforms = numPlatformsArray[0]
            val platforms = arrayOfNulls<cl_platform_id>(numPlatforms)
            clGetPlatformIDs(platforms.size, platforms, null)
            return platforms
        }

        private fun getPlatform(platformId: Int) = getPlatforms()[platformId]
            ?: throw ArrayIndexOutOfBoundsException("Couldn't find the specified platform")

        fun getPlatformsMap(): Map<Int, String> {
            val platforms = getPlatforms()
            val result = mutableMapOf<Int, String>()
            for (platformId in platforms.indices) {
                val platformFromList = platforms[platformId]
                val size = LongArray(1)
                clGetPlatformInfo(platformFromList, CL_PLATFORM_NAME, 0, null, size)
                val buffer = ByteArray(size[0].toInt())
                clGetPlatformInfo(platformFromList, CL_PLATFORM_NAME, buffer.size.toLong(), Pointer.to(buffer), null)
                result[platformId] = String(buffer, 0, buffer.size - 1)
            }
            return result
        }

        private fun getDevices(platformId: Int): Array<cl_device_id?> {
            val platform = getPlatform(platformId)
            val numDevicesArray = IntArray(1)
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray)
            val numDevices = numDevicesArray[0]
            val devices = arrayOfNulls<cl_device_id>(numDevices)
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numDevices, devices, null)
            return devices
        }

        private fun getDevice(platformId: Int, deviceId: Int) = getDevices(platformId)[deviceId]
                ?: throw ArrayIndexOutOfBoundsException("Couldn't find the specified platform or device")

        fun getDevicesMap(platformId: Int): Map<Int, String> {
            val devices = getDevices(platformId)
            val result = mutableMapOf<Int, String>()
            for (deviceId in devices.indices) {
                val deviceFromList = devices[deviceId]
                val size = LongArray(1)
                clGetDeviceInfo(deviceFromList, CL_DEVICE_NAME, 0, null, size)
                val buffer = ByteArray(size[0].toInt())
                clGetDeviceInfo(deviceFromList, CL_DEVICE_NAME, buffer.size.toLong(), Pointer.to(buffer), null)
                result[deviceId] = String(buffer, 0, buffer.size - 1)
            }
            return result
        }
    }

    private val platform: cl_platform_id = getPlatform(platformIndex)

    private val contextProperties: cl_context_properties = cl_context_properties()

    private val device: cl_device_id = getDevice(platformIndex, deviceIndex)

    private val context: cl_context = clCreateContext(contextProperties, 1, arrayOf(device), null, null, null)

    val commandQueue: cl_command_queue

    val program: cl_program

    init {
        setExceptionsEnabled(true)
        contextProperties.addProperty(CL_CONTEXT_PLATFORM.toLong(), platform)
        val properties = cl_queue_properties()
        commandQueue = clCreateCommandQueueWithProperties(context, device, properties, null)
        val sources = arrayOf(
            "Util",
            "InitialComparison",
            "NoiseReduction",
            "FlowKey",
            "Splash",
            "SplashPrep"
        ).map {
            this::class.java.getResource("$it.cl")!!.readText()
        }.toTypedArray()
        program = clCreateProgramWithSource(context, sources.size, sources, null, null)
        clBuildProgram(program, 0, null, null, null, null)
    }

    override fun getFilters(): Map<String, AbstractFilter> = mapOf(
        OpenClInitialComparisonFilter.listName to OpenClInitialComparisonFilter(api = this),
        OpenClNoiseReductionFilter.listName to OpenClNoiseReductionFilter(api = this),
        OpenClFlowKeyFilter.listName to OpenClFlowKeyFilter(api = this),
        OpenClSplashFilter.listName to OpenClSplashFilter(api = this),
    )

    override fun close() {
        clReleaseProgram(program)
        clReleaseCommandQueue(commandQueue)
        clReleaseContext(context)
    }

    fun allocMem(ptr: Pointer?, op: ClMemOperation, size: Int): cl_mem = clCreateBuffer(
        context,
        op.flags,
        size.toLong(),
        ptr,
        null
    )
}
