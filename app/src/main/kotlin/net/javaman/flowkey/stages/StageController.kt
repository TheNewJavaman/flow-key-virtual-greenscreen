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
import javafx.scene.control.Button
import javafx.scene.control.ContextMenu
import javafx.scene.control.Label
import javafx.scene.control.ListCell
import javafx.scene.control.ListView
import javafx.scene.control.MenuItem
import javafx.scene.control.TableColumn
import javafx.scene.control.TableView
import javafx.scene.control.TitledPane
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
import net.javaman.flowkey.hardwareapis.cuda.CudaInitialComparisonFilter
import net.javaman.flowkey.hardwareapis.cuda.CudaNoiseReductionFilter
import net.javaman.flowkey.hardwareapis.opencl.OpenClApi
import net.javaman.flowkey.util.COLOR_DEPTH
import net.javaman.flowkey.util.Camera
import net.javaman.flowkey.util.DEFAULT_COLOR
import net.javaman.flowkey.util.NS_PER_MS
import net.javaman.flowkey.util.ONE_SECOND_NS
import net.javaman.flowkey.util.format
import net.javaman.flowkey.util.toBufferedImage
import net.javaman.flowkey.util.toByteArray
import java.awt.image.BufferedImage
import java.util.LinkedList

val logger = KotlinLogging.logger {}

@Suppress("TooManyFunctions")
class StageController {
    companion object {
        const val LATENCY_COUNTER_DELAY_NS = 250_000_000L

        const val FRAME_TIME_QUEUE_MAX_SIZE = 20
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

    var controllerReplacementKey: ByteArray = DEFAULT_COLOR

    private var lastInstant = System.nanoTime()

    private var lastFpsShownInstant = System.nanoTime()

    private var frameTimeQueue = LinkedList<Long>()

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
        val originalFrameBytes = originalImage.toByteArray()
        var workingBitmap = ByteArray(originalFrameBytes.size / COLOR_DEPTH)
        try {
            filters.forEach { filter ->
                when (filter) {
                    /*is OpenClNoiseReductionFilter -> filter.apply {
                        templateBuffer = originalFrame
                    }
                    is OpenClFlowKeyFilter -> filter.apply {
                        templateBuffer = originalFrame
                    }
                    is OpenClSplashFilter -> filter.apply {
                        inputBlockAverageBuffer = initialBlockAvg!!
                    }*/
                    is CudaInitialComparisonFilter -> filter.apply {
                        originalBuffer = originalFrameBytes
                    }
                    is CudaNoiseReductionFilter -> filter.apply {
                        width = frameWidth
                        height = frameHeight
                    }
                    is CudaFlowKeyFilter -> filter.apply {
                        originalBuffer = originalFrameBytes
                        width = frameWidth
                        height = frameHeight
                    }
                    is CudaGapFillerFilter -> filter.apply {
                        width = frameWidth
                        height = frameHeight
                    }
                }
                workingBitmap = filter.apply(workingBitmap)
            }
        } catch (_: ConcurrentModificationException) {
            logger.warn { "A filter was likely changed, resulting in an out-of-sync filter list" }
        }
        val applyBitmap = api.getApplyBitmap().apply {
            originalBuffer = originalFrameBytes
            replacementKey = controllerReplacementKey
        }
        val finalFrameBytes = applyBitmap.apply(workingBitmap)
        val tEnd = System.nanoTime()
        val tDelta = tEnd - lastInstant
        lastInstant = tEnd
        frameTimeQueue.add(tDelta)
        if (frameTimeQueue.size > FRAME_TIME_QUEUE_MAX_SIZE) {
            frameTimeQueue.remove()
        }
        if (tEnd - lastFpsShownInstant > LATENCY_COUNTER_DELAY_NS) {
            onFXThreadText(
                latencyCounter.textProperty(),
                "${(tDelta / NS_PER_MS.toDouble()).format(2)}ms Frame Latency"
            )
            val fps = ONE_SECOND_NS / frameTimeQueue.average()
            onFXThreadText(
                fpsCounter.textProperty(),
                "${fps.format(2)} FPS"
            )
            lastFpsShownInstant = tEnd
        }
        val modifiedImage = finalFrameBytes.toBufferedImage(frameWidth, frameHeight)
        onFXThreadImage(originalFrame.imageProperty(), SwingFXUtils.toFXImage(originalImage, null))
        onFXThreadImage(modifiedFrame.imageProperty(), SwingFXUtils.toFXImage(modifiedImage, null))
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
