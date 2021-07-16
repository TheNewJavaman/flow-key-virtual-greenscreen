// Adapted from https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

@file:Suppress("WildcardImport")

package net.javaman.flowkey.stages

import javafx.application.Platform
import javafx.beans.property.ObjectProperty
import javafx.beans.property.StringProperty
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
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty
import net.javaman.flowkey.hardwareapis.opencl.*
import net.javaman.flowkey.util.Camera
import net.javaman.flowkey.util.ONE_SECOND_MS
import net.javaman.flowkey.util.toBufferedImage
import net.javaman.flowkey.util.toByteArray
import org.opencv.core.Mat
import java.time.Instant

@Suppress("TooManyFunctions")
class StageController {
    companion object {
        const val LATENCY_COUNTER_DELAY = 250L
    }

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
    lateinit var filterPropertiesTablePane: Pane

    @FXML
    lateinit var filterPropertiesTableView: TableView<Pair<AbstractFilterProperty, Any>>

    @FXML
    lateinit var filterPropertiesName: TableColumn<Pair<AbstractFilterProperty, Any>, String>

    @FXML
    lateinit var filterPropertiesValue: TableColumn<Pair<AbstractFilterProperty, Any>, Number>

    @FXML
    lateinit var bottomBarPane: TitledPane

    @FXML
    lateinit var bottomBarGrid: GridPane

    @FXML
    lateinit var versionLabel: Label

    @FXML
    lateinit var latencyCounter: Label

    @FXML
    lateinit var fpsCounter: Label

    private var camera: Camera? = null

    var api: AbstractApi = OpenClApi()

    private var initialBlockAvg: FloatArray? = null

    private val filters: MutableList<AbstractFilter> = mutableListOf()

    private var lastTEndMilli = 0L

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
        val initialImage = originalFrameData.toBufferedImage(camera!!.maxWidth, camera!!.maxHeight)
        onFXThreadImage(originalFrame.imageProperty(), SwingFXUtils.toFXImage(initialImage, null))

        initialBlockAvg ?: run {
            initialBlockAvg = OpenClSplashPrepFilter(api = api as OpenClApi).apply(originalFrameData)
        }
        val tStart = Instant.now()
        var workingFrame = originalFrameData.clone()
        filters.forEach { filter ->
            when (filter) {
                is OpenClNoiseReductionFilter -> {
                    filter.apply {
                        templateBuffer = originalFrameData
                    }
                }
                is OpenClFlowKeyFilter -> {
                    filter.apply {
                        templateBuffer = originalFrameData
                    }
                }
                is OpenClSplashFilter -> {
                    filter.apply {
                        inputBlockAverageBuffer = initialBlockAvg!!
                    }
                }
            }
            workingFrame = filter.apply(workingFrame)
        }
        val tEnd = Instant.now()
        val modifiedImage = workingFrame.toBufferedImage(camera!!.maxWidth, camera!!.maxHeight)
        onFXThreadImage(modifiedFrame.imageProperty(), SwingFXUtils.toFXImage(modifiedImage, null))

        if (tEnd.toEpochMilli() - lastTEndMilli > LATENCY_COUNTER_DELAY) {
            val tDelta = tEnd.toEpochMilli() - tStart.toEpochMilli()
            onFXThreadText(latencyCounter.textProperty(), "${tDelta}ms Filter Latency")
            val fps = ONE_SECOND_MS / tDelta
            onFXThreadText(fpsCounter.textProperty(), "$fps FPS")
            lastTEndMilli = tEnd.toEpochMilli()
        }
    }

    private fun <T> onFXThreadImage(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }

    private fun onFXThreadText(property: StringProperty, value: String) = Platform.runLater { property.set(value) }

    private fun getImage(name: String) = Image(this::class.java.getResourceAsStream("../icons/$name.png"))

    @FXML
    fun onRefreshAction(
        actionEvent: ActionEvent
    ) {
        if (camera?.cameraActive == true) {
            repeat(2) { startCamera(actionEvent) }
        }
        api = OpenClApi()
    }

    @FXML
    fun onFilterAddAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) = filterAddMenu.show(filterAdd, Side.BOTTOM, 0.0, 0.0)

    fun onFilterAddItem(name: String) {
        val listCell = ListCell<String>()
        listCell.text = name
        listCell.id = filters.size.toString()
        listCell.maxWidth = Double.MAX_VALUE
        filtersListView.items.add(listCell)
        filtersListView.selectionModel.select(filters.size)
        api.getFilters()[name]?.let {
            filters.add(it)
        }
        updateFilterProperties()
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
            val selectionId = if (id.toInt() == filters.size - 1) {
                filters.size - 2
            } else if (id.toInt() == 0 && filters.size == 1){
                null
            } else {
                id.toInt()
            }
            selectionId?.let { filtersListView.selectionModel.select(it) }
            filters.removeAt(id.toInt())
            selectionId?.let { updateFilterProperties() }
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
                filtersListView.selectionModel.select(id.toInt() - 1)
                val filter = filters[id.toInt()]
                filters.removeAt(id.toInt())
                filters.add(id.toInt() - 1, filter)
                updateFilterProperties()
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
                filtersListView.selectionModel.select(id.toInt() + 1)
                val filter = filters[id.toInt()]
                filters.removeAt(id.toInt())
                filters.add(id.toInt() + 1, filter)
                updateFilterProperties()
            }
        }
    }

    fun updateFilterProperties() {
        filtersListView.selectionModel.selectedItem?.id?.let { id ->
            val filter = filters[id.toInt()]
            val properties = filter.getProperties()
            filterPropertiesTableView.items.setAll(properties.toList())
        } ?: filterPropertiesTableView.items.setAll()
    }

    fun setClosed() {
        camera?.close()
        api.close()
    }
}
