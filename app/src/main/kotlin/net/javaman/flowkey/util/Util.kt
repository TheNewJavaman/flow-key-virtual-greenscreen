package net.javaman.flowkey.util

import java.awt.Dimension
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

fun ByteArray.toBufferedImage(width: Int, height: Int): BufferedImage {
    val bufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    System.arraycopy(this, 0, (bufferedImage.raster.dataBuffer as DataBufferByte).data, 0, size)
    return bufferedImage
}

fun BufferedImage.toByteArray(): ByteArray {
    val byteBufferedImage = BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
    byteBufferedImage.createGraphics().drawImage(this, 0, 0, width, height, null)
    return (byteBufferedImage.data.dataBuffer as DataBufferByte).data
}

fun Double.format(n: Int) = "%.${n}f".format(this)

fun Dimension.toListString() = "${this.width}x${this.height}"

fun String.toDimension(): Dimension {
    val split = this.split("x")
    return Dimension(split[0].toInt(), split[1].toInt())
}
