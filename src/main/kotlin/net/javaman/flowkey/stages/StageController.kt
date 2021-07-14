// Adapted from https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

package net.javaman.flowkey.stages

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.embed.swing.SwingFXUtils
import javafx.event.ActionEvent
import javafx.fxml.FXML
import javafx.geometry.Side
import javafx.scene.control.*
import javafx.scene.image.Image
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
    lateinit var filterAdd: Button

    @FXML
    lateinit var filterAddIcon: ImageView

    @FXML
    lateinit var filterAddMenu: ContextMenu

    @FXML
    lateinit var filterDelete: Button

    @FXML
    lateinit var filterDeleteIcon: ImageView

    @FXML
    lateinit var filterUp: Button

    @FXML
    lateinit var filterUpIcon: ImageView

    @FXML
    lateinit var filterDown: Button

    @FXML
    lateinit var filterDownIcon: ImageView

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
    lateinit var playButtonIcon: ImageView

    @FXML
    lateinit var refreshButton: Button

    @FXML
    lateinit var refreshButtonIcon: ImageView

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
                onCameraStart = { playButtonIcon.image = getImage("pause") },
                onCameraStop = { playButtonIcon.image = getImage("play") },
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

    private fun getImage(name: String) = Image(this::class.java.getResourceAsStream("../icons/$name.png"))

    @FXML
    fun onRefreshAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        if (camera?.cameraActive == true) {
            repeat(2) { startCamera(actionEvent) }
        }
    }

    @FXML
    fun onFilterAddAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) = filterAddMenu.show(filterAdd, Side.BOTTOM, 0.0, 0.0)

    fun onFilterAddItem(name: String) {
        api.getFilters()[name]?.let { filters.add(it) }
        val listCell = ListCell<String>()
        listCell.text = name
        listCell.id = (filters.size - 1).toString()
        listCell.maxWidth = Double.MAX_VALUE
        filtersListView.items.add(listCell)
    }

    @FXML
    fun onFilterDeleteAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        filtersListView.selectionModel.selectedItem?.id?.let { id ->
            filtersListView.items.removeIf { it.id == id }
            filtersListView.items
                .filter { it.id.toInt() > id.toInt() }
                .forEach { it.id = (it.id.toInt() - 1).toString() }
            filters.removeAt(id.toInt())
        }
    }

    @FXML
    fun onFilterUpAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        filtersListView.selectionModel.selectedItem?.id?.let { id ->
            if (id.toInt() != 0) {
                val listCell = ListCell<String>()
                listCell.text = filtersListView.items.first { it.id == id }.text
                listCell.id = (id.toInt() - 1).toString()
                listCell.maxWidth = Double.MAX_VALUE
                filtersListView.items.removeIf { it.id == id }
                filtersListView.items
                    .first { it.id == (id.toInt() - 1).toString() }
                    .run { this.id = id }
                filtersListView.items.add(id.toInt() - 1, listCell)
                val filter = filters[id.toInt()]
                filters.removeAt(id.toInt())
                filters.add(id.toInt() - 1, filter)
                filtersListView.selectionModel.select(id.toInt() - 1)
            }
        }
    }

    @FXML
    fun onFilterDownAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        filtersListView.selectionModel.selectedItem?.id?.let { id ->
            if (id.toInt() != filters.size - 1) {
                val listCell = ListCell<String>()
                listCell.text = filtersListView.items.first { it.id == id }.text
                listCell.id = (id.toInt() + 1).toString()
                listCell.maxWidth = Double.MAX_VALUE
                filtersListView.items.removeIf { it.id == id }
                filtersListView.items.first { it.id == (id.toInt() + 1).toString() }.run { this.id = id }
                filtersListView.items.add(id.toInt() + 1, listCell)
                val filter = filters[id.toInt()]
                filters.removeAt(id.toInt())
                filters.add(id.toInt() + 1, filter)
                filtersListView.selectionModel.select(id.toInt() + 1)
            }
        }
    }

    fun setClosed() {
        camera?.close()
        api.close()
    }
}