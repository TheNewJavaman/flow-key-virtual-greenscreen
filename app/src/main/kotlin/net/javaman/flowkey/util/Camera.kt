package net.javaman.flowkey.util

import org.opencv.core.Mat
import org.opencv.videoio.VideoCapture
import org.opencv.videoio.Videoio
import java.util.concurrent.*

@Suppress("LongParameterList")
class Camera constructor(
    private val onFrame: (Mat) -> Unit,
    private val onCameraStart: () -> Unit,
    private val onCameraStop: () -> Unit,
    framesPerSecond: Long = 30L,
    private val cameraId: Int = 0,
    val maxWidth: Int = DEFAULT_WIDTH_PIXELS,
    val maxHeight: Int = DEFAULT_HEIGHT_PIXELS,
    private val threads: Int = 2
) {
    private var timer: ScheduledExecutorService? = null

    private val capture = VideoCapture()

    var cameraActive = false

    private var frameLatencyMs = ONE_SECOND_MS / framesPerSecond

    private val threadPool = ThreadPoolExecutor(threads, threads, 0L, TimeUnit.SECONDS, LinkedBlockingQueue())

    fun toggle() {
        if (!cameraActive) {
            capture.open(cameraId)
            if (capture.isOpened) {
                cameraActive = true
                capture.set(Videoio.CAP_PROP_FRAME_WIDTH, maxWidth.toDouble())
                capture.set(Videoio.CAP_PROP_FRAME_HEIGHT, maxHeight.toDouble())
                val frameGrabber = Runnable {
                    threadPool.submit {
                        onFrame(grabFrame())
                    }
                }
                timer = Executors.newScheduledThreadPool(threads)
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
