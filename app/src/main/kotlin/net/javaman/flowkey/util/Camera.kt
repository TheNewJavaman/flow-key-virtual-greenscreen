package net.javaman.flowkey.util

import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.util.*
import kotlin.concurrent.scheduleAtFixedRate

@Suppress("LongParameterList")
class Camera constructor(
    private val onFrame: (Mat) -> Unit,
    private val onCameraStart: () -> Unit,
    private val onCameraStop: () -> Unit,
    framesPerSecond: Long = 30L,
    private val cameraId: Int = 0,
    val maxWidth: Int = DEFAULT_WIDTH_PIXELS,
    val maxHeight: Int = DEFAULT_HEIGHT_PIXELS
) {
    private var timer: Timer? = null

    private val capture = VideoCapture()

    var cameraActive = false

    private var frameLatencyMs = ONE_SECOND_MS / framesPerSecond

    fun toggle() {
        if (!cameraActive) {
            capture.open(cameraId)
            if (capture.isOpened) {
                cameraActive = true
                capture.set(Videoio.CAP_PROP_FRAME_WIDTH, maxWidth.toDouble())
                capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, maxHeight.toDouble())
                timer = Timer()
                timer!!.scheduleAtFixedRate(0L, frameLatencyMs) { onFrame(grabFrame()) }
                onCameraStart()
            }
        } else {
            cameraActive = false
            onCameraStop()
            stopAcquisition()
        }
    }

    private fun grabFrame(): Mat {
        val frame = Mat()
        if (capture.isOpened) {
            capture.read(frame)
        }
        return frame
    }

    fun close() = stopAcquisition()

    private fun stopAcquisition() {
        if (timer != null) {
            timer!!.cancel()
            timer!!.purge()
        }
        if (capture.isOpened) {
            capture.release()
        }
    }
}
