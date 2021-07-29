package net.javaman.flowkey.hardwareapis.cuda

import jcuda.Pointer
import jcuda.Sizeof
import jcuda.driver.JCudaDriver.cuCtxSetCurrent
import jcuda.driver.JCudaDriver.cuCtxSynchronize
import jcuda.driver.JCudaDriver.cuLaunchKernel
import jcuda.driver.JCudaDriver.cuMemFree
import jcuda.driver.JCudaDriver.cuMemcpyDtoH
import net.javaman.flowkey.hardwareapis.common.AbstractApplyBitmap
import net.javaman.flowkey.hardwareapis.cuda.CudaApi.Companion.BLOCK_SIZE
import kotlin.math.ceil

class CudaApplyBitmap constructor(
    private val api: CudaApi
) : AbstractApplyBitmap {
    override lateinit var originalBuffer: ByteArray

    override lateinit var replacementKey: ByteArray

    override fun apply(inputBuffer: ByteArray): ByteArray {
        cuCtxSetCurrent(api.context)

        val inputPtr = api.allocMem(Sizeof.BYTE * inputBuffer.size.toLong(), Pointer.to(inputBuffer))
        val originalPtr = api.allocMem(Sizeof.BYTE * originalBuffer.size.toLong(), Pointer.to(originalBuffer))
        val replacementKeyPtr = api.allocMem(Sizeof.BYTE * replacementKey.size.toLong(), Pointer.to(replacementKey))
        val outputPtr = api.allocMem(Sizeof.BYTE * originalBuffer.size.toLong())

        val kernelParams = Pointer.to(
            Pointer.to(intArrayOf(inputBuffer.size)),
            Pointer.to(inputPtr),
            Pointer.to(originalPtr),
            Pointer.to(replacementKeyPtr),
            Pointer.to(outputPtr)
        )

        val gridSize = ceil(inputBuffer.size / BLOCK_SIZE.toDouble()).toInt()
        cuLaunchKernel(
            api.applyBitmapProgram,
            gridSize, 1, 1,
            BLOCK_SIZE, 1, 1,
            0, null,
            kernelParams, null
        )
        cuCtxSynchronize()

        val outputBuffer = ByteArray(originalBuffer.size)
        cuMemcpyDtoH(Pointer.to(outputBuffer), outputPtr, Sizeof.BYTE * outputBuffer.size.toLong())

        cuMemFree(inputPtr)
        cuMemFree(originalPtr)
        cuMemFree(replacementKeyPtr)
        cuMemFree(outputPtr)

        return outputBuffer
    }
}