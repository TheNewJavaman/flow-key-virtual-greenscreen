package net.javaman.flowkey

import javafx.embed.swing.SwingFXUtils
import javafx.scene.image.Image
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

object Util {
    const val ONE_SECOND_MS = 1000L

    private fun matToBufferedImage(original: Mat): BufferedImage {
        val width = original.width()
        val height = original.height()
        val channels = original.channels()
        val sourcePixels = ByteArray(width * height * channels)
        original[0, 0, sourcePixels]
        val image = if (original.channels() > 1) {
            BufferedImage(width, height, BufferedImage.TYPE_3BYTE_BGR)
        } else {
            BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY)
        }
        val targetPixels = (image.raster.dataBuffer as DataBufferByte).data
        System.arraycopy(sourcePixels, 0, targetPixels, 0, sourcePixels.size)
        return image
    }

    fun mat2Image(frame: Mat): Image = SwingFXUtils.toFXImage(matToBufferedImage(frame), null)
}