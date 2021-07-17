package net.javaman.flowkey

import javafx.application.Application
import javafx.event.EventHandler
import javafx.fxml.FXMLLoader
import javafx.scene.Scene
import javafx.scene.control.Label
import javafx.scene.control.MenuItem
import javafx.scene.layout.GridPane
import javafx.scene.text.TextAlignment
import javafx.stage.Stage
import net.javaman.flowkey.stages.FilterPropertyNameTableCell
import net.javaman.flowkey.stages.FilterPropertyValueTableCell
import net.javaman.flowkey.stages.StageController
import org.opencv.core.Core
import java.util.*
import kotlin.system.exitProcess


class FlowKeyApplication : Application() {
    companion object {
        const val DEFAULT_WIDTH = 1280.0

        const val DEFAULT_HEIGHT = 800.0

        lateinit var rootElement: GridPane

        lateinit var version: String

        lateinit var controller: StageController

        @JvmStatic
        fun main(args: Array<String>) {
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
            val properties = Properties()
            properties.load(this::class.java.getResourceAsStream("application.properties"))
            version = properties.getProperty("version")
            launch(FlowKeyApplication::class.java)
        }
    }

    override fun start(primaryStage: Stage) {
        val loader = FXMLLoader(this::class.java.getResource("stages/Stage.fxml"))
        rootElement = loader.load()
        val scene = Scene(rootElement, DEFAULT_WIDTH, DEFAULT_HEIGHT)
        scene.stylesheets.add(this::class.java.getResource("stages/Style.css")!!.toExternalForm())
        primaryStage.title = "Flow Key Virtual Greenscreen"
        primaryStage.scene = scene
        primaryStage.show()
        controller = loader.getController()
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
        controller.filtersListView.selectionModel.selectedItemProperty().addListener { _, _, _ ->
            controller.updateFilterProperties()
        }
        val filtersListViewPlaceholder = Label("Nothing here yet.\nAdd a filter above!")
        filtersListViewPlaceholder.textAlignment = TextAlignment.CENTER
        controller.filtersListView.placeholder = filtersListViewPlaceholder
        controller.filterAddMenu.items.setAll(controller.api.getFilters().keys.toList().map { name ->
            val menuItem = MenuItem(name)
            menuItem.setOnAction { controller.onFilterAddItem(name) }
            menuItem
        })
        controller.filterPropertiesTableView.minWidthProperty()
            .bind(controller.filterPropertiesTablePane.widthProperty())
        controller.filterPropertiesTableView.minHeightProperty()
            .bind(controller.filterPropertiesTablePane.heightProperty())
        controller.filterPropertiesTableView.styleClass.add("noheader")
        controller.filterPropertiesName.setCellFactory { FilterPropertyNameTableCell() }
        controller.filterPropertiesValue.setCellFactory { FilterPropertyValueTableCell() }
        controller.bottomBarGrid.minWidthProperty().bind(controller.bottomBarPane.widthProperty())
        controller.versionLabel.text = "Version $version"
        primaryStage.onCloseRequest = EventHandler {
            controller.setClosed()
            exitProcess(0)
        }
    }
}
