package net.javaman.flowkey.util

import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

fun Mat.toByteArray(): ByteArray {
    val size = this.cols() * this.rows() * COLOR_DEPTH
    val originalImage = ByteArray(size = size)
    this.get(0, 0, originalImage)
    return originalImage
}

fun ByteArray.toBufferedImage(width: Int, height: Int): BufferedImage {
    val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    val targetPixels = (bufferedImage.raster.dataBuffer as DataBufferByte).data
    System.arraycopy(this, 0, targetPixels, 0, size)
    return bufferedImage
}