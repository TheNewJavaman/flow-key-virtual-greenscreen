// Adapted from https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

package net.javaman.flowkey

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.embed.swing.SwingFXUtils
import javafx.event.ActionEvent
import javafx.fxml.FXML
import javafx.scene.control.Button
import javafx.scene.image.Image
import javafx.scene.image.ImageView
import org.opencv.core.Mat
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte

class StageController {
    @FXML
    private lateinit var startCameraButton: Button

    @FXML
    lateinit var currentFrame: ImageView

    private lateinit var camera: Camera

    private val filter = Filter()

    @FXML
    fun startCamera(
        @Suppress("UNUSED_PARAMETER")
        actionEvent: ActionEvent
    ) {
        camera = Camera(
            onFrame = ::onFrame,
            onCameraStart = { startCameraButton.text = "Stop Camera" },
            onCameraStop = { startCameraButton.text = "Start Camera" },
        )
        camera.toggle()
    }

    private fun onFrame(frame: Mat) {
        val processedFrame = filter.apply(frame)
        onFXThread(currentFrame.imageProperty(), SwingFXUtils.toFXImage(processedFrame, null))
    }

    fun setClosed() {
        camera.close()
        filter.close()
    }

    private fun <T> onFXThread(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }
}