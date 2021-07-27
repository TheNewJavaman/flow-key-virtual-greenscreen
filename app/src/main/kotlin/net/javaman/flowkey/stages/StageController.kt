@file:Suppress("WildcardImport")

package net.javaman.flowkey.stages

import com.github.sarxos.webcam.Webcam
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
import mu.KotlinLogging
import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractFilter
import net.javaman.flowkey.hardwareapis.cuda.CudaFlowKeyFilter
import net.javaman.flowkey.hardwareapis.cuda.CudaGapFillerFilter
import net.javaman.flowkey.hardwareapis.cuda.CudaNoiseReductionFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi
import net.javaman.flowkey.hardwareapis.opencl.OpenClFlowKeyFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClNoiseReductionFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClSplashFilter
import net.javaman.flowkey.util.*
import java.awt.image.BufferedImage

val logger = KotlinLogging.logger {}

@Suppress("TooManyFunctions")
class StageController {
    companion object {
        const val LATENCY_COUNTER_DELAY_NS = 250_000_000L
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
    lateinit var filtersListPane: Pane

    @FXML
    lateinit var filtersListView: ListView<ListCell<String>>

    @FXML
    lateinit var filterPropertiesTablePane: Pane

    @FXML
    lateinit var filterPropertiesTableView: TableView<Pair<FilterProperty, Any>>

    @FXML
    lateinit var filterPropertiesName: TableColumn<Pair<FilterProperty, Any>, String>

    @FXML
    lateinit var filterPropertiesValue: TableColumn<Pair<FilterProperty, Any>, Any>

    @FXML
    lateinit var generalSettingsTablePane: Pane

    @FXML
    lateinit var generalSettingsTableView: TableView<GeneralSettingsProperty>

    @FXML
    lateinit var generalSettingsName: TableColumn<GeneralSettingsProperty, String>

    @FXML
    lateinit var generalSettingsValue: TableColumn<GeneralSettingsProperty, Any>

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

    var api: AbstractApi = OpenClApi()

    var filters: MutableList<AbstractFilter> = mutableListOf()

    var camera: Camera

    var cameraId: String = Webcam.getDefault().name

    private var initialBlockAvg: FloatArray? = null

    private var lastInstant = System.nanoTime()

    init {
        camera = Camera(
            onFrame = ::onFrame,
            onCameraStart = ::showPauseIcon,
            onCameraStop = ::showPlayIcon,
            cameraId = cameraId
        )
    }

    fun setUpFiltersMenu() {
        filterAddMenu.items.setAll(api.getFilters().keys.toList().map { name ->
            val menuItem = MenuItem(name)
            menuItem.setOnAction { onFilterAddItem(name) }
            menuItem
        })
        filters = mutableListOf()
        filtersListView.items.setAll(emptyList())
    }

    @FXML
    fun startCamera(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) {
        if (camera.isActive) {
            camera.stop()
        } else {
            camera.start()
        }
    }

    fun refreshCamera() {
        camera.stop()
        camera = Camera(
            onFrame = ::onFrame,
            onCameraStart = ::showPauseIcon,
            onCameraStop = ::showPlayIcon,
            cameraId = cameraId
        )
    }

    private fun showPauseIcon() {
        playButtonIcon.image = getImage("pause")
    }

    private fun showPlayIcon() {
        playButtonIcon.image = getImage("play")
    }

    private fun onFrame(originalImage: BufferedImage, frameWidth: Int, frameHeight: Int) {
        val originalFrame = originalImage.toByteArray()
        val tStart = System.nanoTime()
        var workingFrame = originalFrame.clone()
        try {
            filters.forEach { filter ->
                when (filter) {
                    is OpenClNoiseReductionFilter -> filter.apply {
                        templateBuffer = originalFrame
                    }
                    is OpenClFlowKeyFilter -> filter.apply {
                        templateBuffer = originalFrame
                    }
                    is OpenClSplashFilter -> filter.apply {
                        inputBlockAverageBuffer = initialBlockAvg!!
                    }
                    is CudaNoiseReductionFilter -> filter.apply {
                        templateBuffer = originalFrame
                        width = frameWidth
                        height = frameHeight
                    }
                    is CudaFlowKeyFilter -> filter.apply {
                        templateBuffer = originalFrame
                        width = frameWidth
                        height = frameHeight
                    }
                    is CudaGapFillerFilter -> filter.apply {
                        width = frameWidth
                        height = frameHeight
                    }
                }
                workingFrame = filter.apply(workingFrame)
            }
        } catch (e: ConcurrentModificationException) {
            logger.warn { "A filter was likely changed, resulting in an out-of-sync filter list" }
        }
        val tEnd = System.nanoTime()
        if (tEnd - lastInstant > LATENCY_COUNTER_DELAY_NS) {
            val tDelta = tEnd - tStart
            onFXThreadText(
                latencyCounter.textProperty(),
                "${(tDelta / NS_PER_MS.toDouble()).format(2)}ms Filter Latency"
            )
            val fps = ONE_SECOND_NS / tDelta.toDouble()
            onFXThreadText(fpsCounter.textProperty(), "${fps.format(2)} FPS")
            lastInstant = tEnd
        }
        Thread {
            val modifiedImage = workingFrame.toBufferedImage(frameWidth, frameHeight)
            onFXThreadImage(this.originalFrame.imageProperty(), SwingFXUtils.toFXImage(originalImage, null))
            onFXThreadImage(modifiedFrame.imageProperty(), SwingFXUtils.toFXImage(modifiedImage, null))
        }.start()
    }

    private fun <T> onFXThreadImage(property: ObjectProperty<T>, value: T) = Platform.runLater { property.set(value) }

    private fun onFXThreadText(property: StringProperty, value: String) = Platform.runLater { property.set(value) }

    private fun getImage(name: String) = Image(this::class.java.getResourceAsStream("../icons/$name.png"))

    @FXML
    fun onFilterAddAction(
        @Suppress("UNUSED_PARAMETER") actionEvent: ActionEvent
    ) = filterAddMenu.show(filterAdd, Side.BOTTOM, 0.0, 0.0)

    private fun onFilterAddItem(name: String) {
        api.getFilters()[name]?.let { filters.add(it) }
        val listCell = ListCell<String>()
        listCell.text = name
        listCell.id = (filters.size - 1).toString()
        listCell.maxWidth = Double.MAX_VALUE
        filtersListView.items.add(listCell)
        filtersListView.selectionModel.select(listCell)
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
            } else if (id.toInt() == 0 && filters.size == 1) {
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
        camera.stop()
        api.close()
    }
}
