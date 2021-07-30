package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.driver.CUcontext
import jcuda.driver.CUdevice
import jcuda.driver.CUdeviceptr
import jcuda.driver.CUfunction
import jcuda.driver.CUlimit
import jcuda.driver.CUmodule
import jcuda.driver.JCudaDriver
import jcuda.driver.JCudaDriver.cuCtxCreate
import jcuda.driver.JCudaDriver.cuCtxSetLimit
import jcuda.driver.JCudaDriver.cuDeviceGet
import jcuda.driver.JCudaDriver.cuInit
import jcuda.driver.JCudaDriver.cuMemAlloc
import jcuda.driver.JCudaDriver.cuMemcpyHtoD
import jcuda.driver.JCudaDriver.cuModuleGetFunction
import jcuda.driver.JCudaDriver.cuModuleLoadData
import jcuda.nvrtc.JNvrtc
import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractApiConsts
import net.javaman.flowkey.hardwareapis.common.AbstractApplyBitmap
import net.javaman.flowkey.hardwareapis.common.AbstractFilter

class CudaApi : AbstractApi {
    companion object : AbstractApiConsts {
        override val listName = "Cuda"

        const val BLOCK_SIZE = 256

        const val STDOUT_BUFFER_SIZE = 4096L
    }

    val initialComparisonProgram: CUfunction

    val noiseReductionProgram: CUfunction

    val flowKeyProgram: CUfunction

    val gapFillerProgram: CUfunction

    val applyBitmapProgram: CUfunction

    val context: CUcontext

    init {
        JCudaDriver.setExceptionsEnabled(true)
        JNvrtc.setExceptionsEnabled(true)

        cuInit(0)
        val device = CUdevice()
        cuDeviceGet(device, 0)

        context = CUcontext()
        cuCtxCreate(context, 0, device)

        val module = CUmodule()
        val fatbin = this::class.java.getResource("CudaKernels.fatbin")!!.readBytes()
        cuModuleLoadData(module, fatbin)

        initialComparisonProgram = CUfunction()
        cuModuleGetFunction(initialComparisonProgram, module, "initialComparisonKernel")
        noiseReductionProgram = CUfunction()
        cuModuleGetFunction(noiseReductionProgram, module, "noiseReductionKernel")
        flowKeyProgram = CUfunction()
        cuModuleGetFunction(flowKeyProgram, module, "flowKeyKernel")
        gapFillerProgram = CUfunction()
        cuModuleGetFunction(gapFillerProgram, module, "gapFillerKernel")
        applyBitmapProgram = CUfunction()
        cuModuleGetFunction(applyBitmapProgram, module, "applyBitmapKernel")

        cuCtxSetLimit(CUlimit.CU_LIMIT_PRINTF_FIFO_SIZE, STDOUT_BUFFER_SIZE)
    }

    override fun getFilters(): Map<String, AbstractFilter> = mapOf(
        CudaInitialComparisonFilter.listName to CudaInitialComparisonFilter(api = this),
        CudaNoiseReductionFilter.listName to CudaNoiseReductionFilter(api = this),
        CudaFlowKeyFilter.listName to CudaFlowKeyFilter(api = this),
        CudaGapFillerFilter.listName to CudaGapFillerFilter(api = this)
    )

    override fun getApplyBitmap(): AbstractApplyBitmap = CudaApplyBitmap(api = this)

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
