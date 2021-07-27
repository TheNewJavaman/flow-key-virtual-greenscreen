package net.javaman.flowkey.util

import com.github.sarxos.webcam.Webcam
import java.awt.Dimension
import java.awt.image.BufferedImage
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

class Camera constructor(
    private val onFrame: (BufferedImage, Int, Int) -> Unit,
    private val onCameraStart: () -> Unit,
    private val onCameraStop: () -> Unit,
    cameraId: String,
    framesPerSecond: Long = 30L
) {
    private val frameLatencyMs = ONE_SECOND_MS / framesPerSecond

    private var timer: Timer? = null

    private val camera: Webcam = Webcam.getWebcamByName(cameraId)

    var isActive: Boolean = false

    fun start() {
        camera.open()
        timer = Timer()
        timer!!.scheduleAtFixedRate(0L, frameLatencyMs) {
            with (camera.viewSize) {
                camera.image?.let {
                    onFrame(it, this.width, this.height)
                }
            }
        }
        isActive = true
        onCameraStart()
    }

    fun stop() {
        isActive = false
        timer?.cancel()
        timer?.purge()
        camera.close()
        onCameraStop()
    }

    fun getResolutions(): Array<Dimension> = camera.viewSizes

    fun getResolution(): Dimension = camera.viewSize ?: camera.viewSizes.first()

    fun setResolution(resolution: Dimension) {
        stop()
        camera.viewSize = resolution
        start()
    }
}
