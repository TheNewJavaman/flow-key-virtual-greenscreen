// Adapted from https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

package net.javaman.flowkey.stages

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.embed.swing.SwingFXUtils
import javafx.event.ActionEvent
import javafx.fxml.FXML
import javafx.scene.control.*
import javafx.scene.image.ImageView
import javafx.scene.layout.GridPane
import javafx.scene.layout.HBox
import javafx.scene.layout.Pane
import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi
import net.javaman.flowkey.hardwareapis.opencl.SplashPrepFilter
import net.javaman.flowkey.util.Camera
import net.javaman.flowkey.util.toBufferedImage
import net.javaman.flowkey.util.toByteArray
import org.opencv.core.Mat

class StageController {
    @FXML
    lateinit var originalPane: Pane

    @FXML
    lateinit var originalHBox: HBox

    @FXML
    lateinit var originalFrame: ImageView

    @FXML
    lateinit var modifiedPane: Pane

    @FXML
    lateinit var modifiedHBox: HBox

    @FXML
    lateinit var modifiedFrame: ImageView

    @FXML
    lateinit var filtersPane: TitledPane

    @FXML
    lateinit var filtersHeader: GridPane

    @FXML
    lateinit var filterAdd: MenuButton

    @FXML
    lateinit var filterDelete: Button

    @FXML
    lateinit var filterPropertiesPane: TitledPane

    @FXML
    lateinit var filterPropertiesHeader: GridPane

    @FXML
    lateinit var generalSettingsPane: TitledPane

    @FXML
    lateinit var generalSettingsHeader: GridPane

    @FXML
    lateinit var playButton: Button

    @FXML
    lateinit var refreshButton: Button

    @FXML
    lateinit var filtersListPane: Pane

    @FXML
    lateinit var filtersListView: ListView<ListCell<String>>

    @FXML
    lateinit var filterPropertiesListPane: Pane

    @FXML
    lateinit var filterPropertiesListView: ListView<ListCell<String>>

    private var camera: Camera? = null

    val api: AbstractApi = OpenClApi()

    private var initialBlockAvg: FloatArray? = null

    private val filters: MutableList<AbstractFilter> = mutableListOf()

    @FXML
    fun startCamera(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        camera ?: run {
            camera = Camera(
                onFrame = ::onFrame,
                onCameraStart = { playButton.text = "⏸" },
                onCameraStop = { playButton.text = "▶" },
                cameraId = 1
            )
        }
        camera!!.toggle()
    }

    private fun onFrame(frame: Mat) {
        val originalFrameData = frame.toByteArray()
        initialBlockAvg ?: run {
            initialBlockAvg = SplashPrepFilter(api = api as OpenClApi).apply(originalFrameData)
        }
        var workingFrame = originalFrameData.clone()
        filters.forEach { filter ->
            workingFrame = filter.apply(workingFrame)
        }
        val initialImage = originalFrameData.toBufferedImage(camera!!.maxWidth, camera!!.maxHeight)
        val modifiedImage = workingFrame.toBufferedImage(camera!!.maxWidth, camera!!.maxHeight)
        onFXThread(originalFrame.imageProperty(), SwingFXUtils.toFXImage(initialImage, null))
        onFXThread(modifiedFrame.imageProperty(), SwingFXUtils.toFXImage(modifiedImage, null))
    }

    private fun <T> onFXThread(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }

    fun setClosed() {
        camera?.close()
        api.close()
    }
}