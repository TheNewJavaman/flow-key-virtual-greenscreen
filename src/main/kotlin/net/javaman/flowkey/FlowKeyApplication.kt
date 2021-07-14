package net.javaman.flowkey

import javafx.application.Application
import javafx.event.EventHandler
import javafx.fxml.FXMLLoader
import javafx.scene.Scene
import javafx.scene.control.MenuItem
import javafx.scene.layout.GridPane
import javafx.stage.Stage
import mu.KotlinLogging
import net.javaman.flowkey.stages.StageController
import org.opencv.core.Core

val logger = KotlinLogging.logger {}

class FlowKeyApplication : Application() {
    companion object {
        lateinit var rootElement: GridPane

        @JvmStatic
        fun main(args: Array<String>) {
            logger.info { "Starting Flow Key application..." }
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
            launch(FlowKeyApplication::class.java)
        }
    }

    override fun start(primaryStage: Stage) {
        val loader = FXMLLoader(FlowKeyApplication::class.java.getResource("stages/Stage.fxml"))
        rootElement = loader.load()
        val scene = Scene(rootElement, 1280.0, 800.0)
        primaryStage.title = "Flow Key Virtual Greenscreen"
        primaryStage.scene = scene
        primaryStage.show()
        val controller: StageController = loader.getController()
        controller.originalHBox.minWidthProperty().bind(controller.originalPane.widthProperty())
        controller.originalHBox.minHeightProperty().bind(controller.originalPane.heightProperty())
        controller.originalFrame.fitWidthProperty().bind(controller.originalHBox.widthProperty())
        controller.originalFrame.fitHeightProperty().bind(controller.originalHBox.heightProperty())
        controller.modifiedHBox.minWidthProperty().bind(controller.originalPane.widthProperty())
        controller.modifiedHBox.minHeightProperty().bind(controller.originalPane.heightProperty())
        controller.modifiedFrame.fitWidthProperty().bind(controller.modifiedHBox.widthProperty())
        controller.modifiedFrame.fitHeightProperty().bind(controller.modifiedHBox.heightProperty())
        controller.filtersHeader.minWidthProperty().bind(controller.filtersPane.widthProperty())
        controller.filterPropertiesHeader.minWidthProperty().bind(controller.filterPropertiesPane.widthProperty())
        controller.generalSettingsHeader.minWidthProperty().bind(controller.generalSettingsPane.widthProperty())
        controller.filtersListView.minWidthProperty().bind(controller.filtersListPane.widthProperty())
        controller.filtersListView.minHeightProperty().bind(controller.filtersListPane.heightProperty())
        controller.filterAddMenu.items.setAll(controller.api.getFilters().keys.toList().map { name ->
            val menuItem = MenuItem(name)
            menuItem.setOnAction { controller.onFilterAddItem(name) }
            menuItem
        })
        primaryStage.onCloseRequest = EventHandler { controller.setClosed() }
    }
}