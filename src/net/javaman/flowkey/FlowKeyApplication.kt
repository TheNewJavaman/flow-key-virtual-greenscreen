// Followed this tutorial: https://opencv-java-tutorials.readthedocs.io/en/latest/_images/03-08.png

package net.javaman.flowkey

import javafx.application.Application
import javafx.event.EventHandler
import javafx.fxml.FXMLLoader
import javafx.scene.Scene
import javafx.scene.layout.BorderPane
import javafx.stage.Stage
import org.opencv.core.Core
import java.util.logging.Logger

class FlowKeyApplication : Application() {
    companion object {
        private val logger: Logger = Logger.getLogger(FlowKeyApplication::class.java.name)

        lateinit var rootElement: BorderPane

        @JvmStatic
        fun main(args: Array<String>) {
            logger.info { "Starting Flow Key application..." }
            System.loadLibrary(Core.NATIVE_LIBRARY_NAME)
            launch(FlowKeyApplication::class.java)
        }
    }

    override fun start(primaryStage: Stage) {
        val loader = FXMLLoader(FlowKeyApplication::class.java.getResource("Stage.fxml"))
        rootElement = loader.load<BorderPane>()
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