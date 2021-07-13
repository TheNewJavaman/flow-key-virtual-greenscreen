package net.javaman.flowkey

import javafx.application.Application
import javafx.event.EventHandler
import javafx.fxml.FXMLLoader
import javafx.scene.Scene
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
        val scene = Scene(rootElement, 800.0, 600.0)
        primaryStage.title = "Flow Key Virtual Greenscreen"
        primaryStage.scene = scene
        primaryStage.show()
        val controller: StageController = loader.getController()
        controller.originalFrame.fitWidthProperty().bind(controller.originalPane.widthProperty())
        controller.originalFrame.fitHeightProperty().bind(controller.originalPane.heightProperty())
        controller.modifiedFrame.fitWidthProperty().bind(controller.modifiedPane.widthProperty())
        controller.modifiedFrame.fitHeightProperty().bind(controller.modifiedPane.widthProperty())
        primaryStage.onCloseRequest = EventHandler { controller.setClosed() }
    }
}