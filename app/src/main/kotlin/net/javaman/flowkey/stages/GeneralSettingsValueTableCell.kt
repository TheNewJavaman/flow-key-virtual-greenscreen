package net.javaman.flowkey.stages

import com.github.sarxos.webcam.Webcam
import javafx.geometry.Pos
import javafx.scene.control.ColorPicker
import javafx.scene.control.ComboBox
import javafx.scene.control.TableCell
import javafx.scene.paint.Color
import net.javaman.flowkey.FlowKeyApplication
import net.javaman.flowkey.FlowKeyApplication.Companion.MAIN_PACKAGE
import net.javaman.flowkey.hardwareapis.common.AbstractApi
import net.javaman.flowkey.hardwareapis.common.AbstractApiConsts
import net.javaman.flowkey.util.PIXEL_MULTIPLIER
import net.javaman.flowkey.util.toDimension
import net.javaman.flowkey.util.toListString
import org.reflections.Reflections
import kotlin.reflect.full.companionObjectInstance
import kotlin.reflect.full.createInstance

class GeneralSettingsValueTableCell<T> : TableCell<GeneralSettingsProperty, T>() {
    companion object {
        const val WIDTH_PIXELS = 200.0
    }

    init {
        this.alignment = Pos.CENTER_RIGHT
        text = null
    }

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        if (empty) {
            graphic = null
        } else {
            val data = tableView.items[tableRow.index] as GeneralSettingsProperty
            when (data.dataType) {
                GeneralSettingsPropertyType.API -> setApiProperty()
                GeneralSettingsPropertyType.CAMERA -> setCameraProperty()
                GeneralSettingsPropertyType.RESOLUTION -> setResolutionProperty()
                GeneralSettingsPropertyType.COLOR -> setColorProperty()
            }
        }
    }

    private fun setApiProperty() {
        val apiOption = ComboBox<String>()
        val apis = Reflections(MAIN_PACKAGE).getSubTypesOf(AbstractApi::class.java)
        apiOption.items.setAll(apis.map {
            (it.kotlin.companionObjectInstance as AbstractApiConsts).listName
        })
        val controller = FlowKeyApplication.controller
        val listName = (controller.api::class.companionObjectInstance as AbstractApiConsts).listName
        apiOption.selectionModel.select(listName)
        apiOption.setOnAction {
            controller.api.close()
            controller.api = apis.first { apiClass ->
                val apiClassName = (apiClass.kotlin.companionObjectInstance as AbstractApiConsts).listName
                apiOption.selectionModel.selectedItem == apiClassName
            }.kotlin.createInstance()
            controller.setUpFiltersMenu()
        }
        apiOption.prefWidth = WIDTH_PIXELS
        graphic = apiOption
        controller.setUpFiltersMenu()
    }

    private fun setCameraProperty() {
        val cameraOption = ComboBox<String>()
        val cameras = Webcam.getWebcams()
        cameraOption.items.setAll(cameras.map { it.name })
        val controller = FlowKeyApplication.controller
        cameraOption.selectionModel.select(controller.cameraId)
        cameraOption.setOnAction {
            controller.camera.stop()
            controller.cameraId = cameraOption.selectionModel.selectedItem
            controller.refreshCamera()
            controller.generalSettingsTableView.refresh()
        }
        cameraOption.prefWidth = WIDTH_PIXELS
        graphic = cameraOption
    }

    private fun setResolutionProperty() {
        val resolutionOption = ComboBox<String>()
        val controller = FlowKeyApplication.controller
        val resolutions = controller.camera.getResolutions()
        resolutionOption.items.setAll(resolutions.map { it.toListString() })
        val resolution = controller.camera.getResolution()
        resolutionOption.selectionModel.select(resolution.toListString())
        resolutionOption.setOnAction {
            controller.camera.setResolution(resolutionOption.selectionModel.selectedItem.toDimension())
        }
        resolutionOption.prefWidth = WIDTH_PIXELS
        graphic = resolutionOption
    }

    private fun setColorProperty() {
        val controller = FlowKeyApplication.controller
        val originalColor = controller.controllerReplacementKey.map {it.toUByte().toInt() / PIXEL_MULTIPLIER.toDouble()}
        val colorPicker = ColorPicker(Color.color(originalColor[2], originalColor[1], originalColor[0]))
        colorPicker.setOnAction {
            val newColor = colorPicker.value
            controller.controllerReplacementKey = listOf(newColor.blue, newColor.green, newColor.red).map {
                (it * PIXEL_MULTIPLIER.toDouble()).toInt().toByte()
            }.toByteArray()
        }
        colorPicker.prefWidth = WIDTH_PIXELS
        graphic = colorPicker
    }
}