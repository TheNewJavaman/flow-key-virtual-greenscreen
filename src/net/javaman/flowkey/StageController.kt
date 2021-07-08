package net.javaman.flowkey

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.event.ActionEvent
import javafx.fxml.FXML
import javafx.scene.control.Button
import javafx.scene.image.Image
import javafx.scene.image.ImageView


class StageController {
    @FXML
    private lateinit var startCameraButton: Button

    @FXML
    lateinit var currentFrame: ImageView

    private lateinit var camera: Camera

    @FXML
    fun startCamera(
        @Suppress("UNUSED_PARAMETER")
        actionEvent: ActionEvent
    ) {
        camera = Camera(
            onFrame = { image: Image -> onFXThread(currentFrame.imageProperty(), image) },
            onCameraStart = { startCameraButton.text = "Stop Camera" },
            onCameraStop = { startCameraButton.text = "Start Camera" },
            framesPerSecond = 60L,
            cameraId = 0
        )
        camera.toggle()
    }

    fun setClosed() = camera.close()

    private fun <T> onFXThread(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }
}