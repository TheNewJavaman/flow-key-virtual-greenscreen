// Adapted from https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

package net.javaman.flowkey.stages

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.embed.swing.SwingFXUtils
import javafx.event.ActionEvent
import javafx.fxml.FXML
import javafx.scene.control.Button
import javafx.scene.image.ImageView
import javafx.scene.layout.Pane
import net.javaman.flowkey.filters.OpenClFilter
import net.javaman.flowkey.util.Camera
import net.javaman.flowkey.util.toBufferedImage
import net.javaman.flowkey.util.toByteArray
import org.opencv.core.Mat

class StageController {
    @FXML
    private lateinit var startCameraButton: Button

    @FXML
    lateinit var originalPane: Pane

    @FXML
    lateinit var originalFrame: ImageView

    @FXML
    lateinit var modifiedPane: Pane

    @FXML
    lateinit var modifiedFrame: ImageView

    private lateinit var camera: Camera

    private val filter = OpenClFilter()

    private var initialBlockAvg: FloatArray? = null

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
        val originalFrameData = frame.toByteArray()
        if (initialBlockAvg == null) {
            initialBlockAvg = filter.applySplashPrep(originalFrameData)
        }
        val originalImage = originalFrameData.toBufferedImage(camera.maxWidth, camera.maxHeight)
        onFXThread(originalFrame.imageProperty(), SwingFXUtils.toFXImage(originalImage, null))
        filter.applySplash(originalFrameData, initialBlockAvg!!).run {
            val modifiedImage = this.first.toBufferedImage(camera.maxWidth, camera.maxHeight)
            onFXThread(modifiedFrame.imageProperty(), SwingFXUtils.toFXImage(modifiedImage, null))
            initialBlockAvg = this.second
        }
    }

    fun setClosed() {
        camera.close()
        filter.close()
    }

    private fun <T> onFXThread(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }
}