package net.javaman.flowkey

import javafx.application.Application
import javafx.event.EventHandler
import javafx.fxml.FXMLLoader
import javafx.scene.Scene
import javafx.scene.layout.BorderPane
import javafx.stage.Stage
import mu.KotlinLogging
import net.javaman.flowkey.stages.StageController
import org.opencv.core.Core

val logger = KotlinLogging.logger {}

class FlowKeyApplication : Application() {
    companion object {
        lateinit var rootElement: BorderPane

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
        primaryStage.title = "Flow Key"
        primaryStage.scene = scene
        primaryStage.show()
        val controller: StageController = loader.getController()
        controller.currentFrame.fitWidthProperty().bind(rootElement.widthProperty())
        controller.currentFrame.fitHeightProperty().bind(rootElement.heightProperty())
        primaryStage.onCloseRequest = EventHandler { controller.setClosed() }
    }
}