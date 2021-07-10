package net.javaman.flowkey

import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.util.concurrent.Executors
import java.util.concurrent.ScheduledExecutorService
import java.util.concurrent.TimeUnit

class Camera constructor(
    private val onFrame: (Mat) -> Unit,
    private val onCameraStart: () -> Unit,
    private val onCameraStop: () -> Unit,
    framesPerSecond: Long = 30L,
    private val cameraId: Int = 0,
    private val maxWidth: Int = DEFAULT_WIDTH_PIXELS,
    private val maxHeight: Int = DEFAULT_HEIGHT_PIXELS
) {
    companion object {
        const val DEFAULT_WIDTH_PIXELS = 1280
        const val DEFAULT_HEIGHT_PIXELS = 720
        const val COLOR_DEPTH = 3
        const val ONE_SECOND_MS = 1000
    }

    private var timer: ScheduledExecutorService? = null

    private val capture = VideoCapture()

    private var cameraActive = false

    private var frameLatencyMs = ONE_SECOND_MS / framesPerSecond

    fun toggle() {
        if (!cameraActive) {
            capture.open(cameraId)
            if (capture.isOpened) {
                cameraActive = true
                capture.set(Videoio.CAP_PROP_FRAME_WIDTH, maxWidth.toDouble())
                capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, maxHeight.toDouble())
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
            try {
                capture.read(frame)
            } catch (e: Exception) {
                logger.warn(e) { "Couldn't grab frame" }
            }
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