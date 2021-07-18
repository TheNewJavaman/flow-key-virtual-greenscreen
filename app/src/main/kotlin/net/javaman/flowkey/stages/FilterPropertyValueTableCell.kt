// Adapted from https://stackoverflow.com/questions/56029250/how-can-i-set-decimal-places-in-javafx-spinner

package net.javaman.flowkey.stages

import javafx.geometry.Pos
import javafx.scene.control.ColorPicker
import javafx.scene.control.ComboBox
import javafx.scene.control.Spinner
import javafx.scene.control.TableCell
import javafx.scene.paint.Color
import javafx.util.StringConverter
import net.javaman.flowkey.FlowKeyApplication
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty
import net.javaman.flowkey.hardwareapis.common.AbstractFilterPropertyType
import net.javaman.flowkey.hardwareapis.common.ColorSpace
import net.javaman.flowkey.util.PIXEL_MULTIPLIER
import java.text.DecimalFormat
import java.util.*
import kotlin.math.abs


class FilterPropertyValueTableCell<T> : TableCell<Pair<AbstractFilterProperty, Any>, T>() {
    companion object {
        const val INT_STEP = 1

        const val FLOAT_STEP = 0.005

        const val WIDTH_PIXELS = 100.0
    }

    init {
        this.alignment = Pos.CENTER_RIGHT
        text = null
    }

    class DoubleConverter : StringConverter<Double>() {
        private val decimalFormat = DecimalFormat("#.###")

        override fun toString(doubleValue: Double): String? = decimalFormat.format(doubleValue)

        override fun fromString(string: String?): Double? {
            var mutableString = string ?: return null
            mutableString = mutableString.trim { it <= ' ' }
            return if (mutableString.isEmpty()) null
            else decimalFormat.parse(mutableString).toDouble()
        }
    }

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        if (empty) {
            graphic = null
        } else {
            val data = (tableView.items[tableRow.index] as Pair<AbstractFilterProperty, Any>)
            when (data.first.dataType) {
                AbstractFilterPropertyType.INT -> setIntProperty(data)
                AbstractFilterPropertyType.FLOAT -> setFloatProperty(data)
                AbstractFilterPropertyType.COLOR_SPACE -> setColorSpaceProperty(data)
                AbstractFilterPropertyType.COLOR -> setColorProperty(data)
            }
        }
    }

    private fun setIntProperty(data: Pair<AbstractFilterProperty, Any>) {
        val intSpinner = Spinner<Int>(0, Int.MAX_VALUE, 0, INT_STEP)
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        intSpinner.valueFactory.value = data.second as Int
        intSpinner.editor.textProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
        }
        intSpinner.focusedProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
        }
        intSpinner.prefWidth = WIDTH_PIXELS
        graphic = intSpinner
    }

    private fun setFloatProperty(data: Pair<AbstractFilterProperty, Any>) {
        val doubleSpinner = Spinner<Double>(0.0, 1.0, 0.0, FLOAT_STEP)
        doubleSpinner.valueFactory.converter = DoubleConverter()
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        doubleSpinner.valueFactory.value = (data.second as Float).toDouble()
        doubleSpinner.editor.textProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(
                data.first.listName,
                doubleSpinner.valueFactory.value.toFloat()
            )
        }
        doubleSpinner.focusedProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(
                data.first.listName,
                doubleSpinner.valueFactory.value.toFloat()
            )
        }
        doubleSpinner.prefWidth = WIDTH_PIXELS
        graphic = doubleSpinner
    }

    private fun setColorProperty(data: Pair<AbstractFilterProperty, Any>) {
        val originalColor = (data.second as ByteArray).map { abs(it.toDouble()) }
        val colorPicker = ColorPicker(Color.color(originalColor[2], originalColor[1], originalColor[0]))
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        colorPicker.setOnAction { _ ->
            val newColor = colorPicker.value
            controller.filters[selectedId].setProperty(
                data.first.listName,
                listOf(newColor.blue, newColor.green, newColor.red).map {
                    (it * PIXEL_MULTIPLIER.toDouble()).toInt().toByte()
                }.toByteArray()
            )
        }
        colorPicker.prefWidth = WIDTH_PIXELS
        graphic = colorPicker
    }

    private fun setColorSpaceProperty(data: Pair<AbstractFilterProperty, Any>) {
        val colorOption = ComboBox<String>()
        colorOption.items.setAll(ColorSpace.values().toList().map { it.listName }.reversed())
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        colorOption.selectionModel.select((data.second as ColorSpace).listName)
        colorOption.setOnAction { _ ->
            controller.filters[selectedId].setProperty(
                data.first.listName,
                ColorSpace.valueOf(colorOption.selectionModel.selectedItem.uppercase(Locale.getDefault()))
            )
        }
        graphic = colorOption
        colorOption.prefWidth = WIDTH_PIXELS
    }
}
