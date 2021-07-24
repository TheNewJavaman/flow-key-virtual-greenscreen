@file:Suppress("WildcardImport")

package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.driver.*
import jcuda.driver.CUjitInputType.CU_JIT_INPUT_LIBRARY
import jcuda.driver.CUjitInputType.CU_JIT_INPUT_PTX
import jcuda.driver.JCudaDriver.*
import jcuda.nvrtc.JNvrtc
import jcuda.nvrtc.JNvrtc.*
import jcuda.nvrtc.nvrtcProgram
import mu.KotlinLogging
import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractApiConsts
import net.javaman.flowkey.hardwareapis.common.AbstractFilter

val logger = KotlinLogging.logger {}

class CudaApi : AbstractApi {
    companion object : AbstractApiConsts {
        override val listName = "Cuda"

        const val KERNELS_FILE = "CudaKernels.cu"

        const val BLOCK_SIZE = 256

        const val STDOUT_BUFFER_SIZE = 4096L
    }

    val initialComparisonProgram: CUfunction

    val noiseReductionProgram: CUfunction

    val flowKeyProgram: CUfunction

    val gapFillerProgram: CUfunction

    val context: CUcontext

    init {
        JCudaDriver.setExceptionsEnabled(true)
        JNvrtc.setExceptionsEnabled(true)

        cuInit(0)
        val device = CUdevice()
        cuDeviceGet(device, 0)

        context = CUcontext()
        cuCtxCreate(context, 0, device)

        val programSource = this::class.java.getResource(KERNELS_FILE)!!.readText()
        val program = nvrtcProgram()
        nvrtcCreateProgram(program, programSource, KERNELS_FILE, 0, null, null)
        val options = arrayOf("-lineinfo")
        nvrtcCompileProgram(program, options.size, options)

        val programLog = arrayOfNulls<String>(1)
        nvrtcGetProgramLog(program, programLog)
        when (val output = programLog[0]) {
            "" -> logger.info { "Cuda program compiled without output" }
            else -> logger.info { "Cuda program compiled with the following output:\n${output}" }
        }

        val ptx = arrayOfNulls<String>(1)
        nvrtcGetPTX(program, ptx)
        nvrtcDestroyProgram(program)

        val libPath = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.4\\lib\\x64"
        val libName = "$libPath\\cudadevrt.lib"
        val ptxData = ptx[0]!!.encodeToByteArray()
        val jitOptions = JITOptions()

        val state = CUlinkState()
        cuLinkCreate(jitOptions, state)
        cuLinkAddFile(state, CU_JIT_INPUT_LIBRARY, libName, jitOptions)
        cuLinkAddData(state, CU_JIT_INPUT_PTX, Pointer.to(ptxData), ptxData.size.toLong(), "input.ptx", jitOptions)

        val image = Pointer()
        cuLinkComplete(state, image, longArrayOf(0))
        val module = CUmodule()
        cuModuleLoadDataEx(module, image, 0, IntArray(0), Pointer.to(IntArray(0)))
        cuLinkDestroy(state)

        initialComparisonProgram = CUfunction()
        cuModuleGetFunction(initialComparisonProgram, module, "initialComparisonKernel")
        noiseReductionProgram = CUfunction()
        cuModuleGetFunction(noiseReductionProgram, module, "noiseReductionKernel")
        flowKeyProgram = CUfunction()
        cuModuleGetFunction(flowKeyProgram, module, "flowKeyKernel")
        gapFillerProgram = CUfunction()
        cuModuleGetFunction(gapFillerProgram, module, "gapFillerKernel")

        cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, STDOUT_BUFFER_SIZE)
    }

    override fun getFilters(): Map<String, AbstractFilter> = mapOf(
        CudaInitialComparisonFilter.listName to CudaInitialComparisonFilter(api = this),
        CudaNoiseReductionFilter.listName to CudaNoiseReductionFilter(api = this),
        CudaFlowKeyFilter.listName to CudaFlowKeyFilter(api = this),
        CudaGapFillerFilter.listName to CudaGapFillerFilter(api = this)
    )

    @Suppress("EmptyFunctionBlock")
    override fun close() {
    }

    fun allocMem(size: Long, hostPtr: Pointer? = null): CUdeviceptr {
        val devicePtr = CUdeviceptr()
        cuMemAlloc(devicePtr, size)
        if (hostPtr != null) {
            cuMemcpyHtoD(devicePtr, hostPtr, size)
        }
        return devicePtr
    }
}
