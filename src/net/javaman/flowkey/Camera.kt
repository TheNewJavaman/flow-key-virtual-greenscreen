package net.javaman.flowkey

import net.javaman.flowkey.Util.ONE_SECOND_MS
import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

class Camera constructor(
    private val onFrame: (Mat) -> Unit,
    private val onCameraStart: () -> Unit,
    private val onCameraStop: () -> Unit,
    val framesPerSecond: Long = 60L,
    private val cameraId: Int = 0
) {
    private var timer: ScheduledExecutorService? = null

    private val capture = VideoCapture()

    private var cameraActive = false

    private var frameLatencyMs = ONE_SECOND_MS / framesPerSecond

    fun toggle() {
        if (!cameraActive) {
            capture.open(cameraId)
            if (capture.isOpened) {
                cameraActive = true
                val frameGrabber = Runnable {
                    onFrame(grabFrame())
                }
                timer = Executors.newSingleThreadScheduledExecutor()
                timer!!.scheduleAtFixedRate(frameGrabber, 0, frameLatencyMs, TimeUnit.MILLISECONDS)
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
        if (timer != null && !timer!!.isShutdown) {
            timer!!.shutdown()
            timer!!.awaitTermination(frameLatencyMs, TimeUnit.MILLISECONDS)
        }
        if (capture.isOpened) {
            capture.release()
        }
    }
}