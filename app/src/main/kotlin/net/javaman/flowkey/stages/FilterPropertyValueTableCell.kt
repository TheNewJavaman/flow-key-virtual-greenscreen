// Adapted from https://stackoverflow.com/questions/56029250/how-can-i-set-decimal-places-in-javafx-spinner

package net.javaman.flowkey.stages

import javafx.geometry.Pos
import javafx.scene.control.ColorPicker
import javafx.scene.control.Spinner
import javafx.scene.control.TableCell
import javafx.scene.paint.Color
import javafx.util.StringConverter
import net.javaman.flowkey.FlowKeyApplication
import net.javaman.flowkey.util.PIXEL_MULTIPLIER
import java.text.DecimalFormat

class FilterPropertyValueTableCell<T> : TableCell<Pair<FilterProperty, Any>, T>() {
    companion object {
        const val INT_STEP = 1

        const val FLOAT_STEP = 0.001

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
            val data = tableView.items[tableRow.index] as Pair<FilterProperty, Any>
            when (data.first.dataType) {
                FilterPropertyType.INT -> setIntProperty(data)
                FilterPropertyType.FLOAT -> setFloatProperty(data)
                FilterPropertyType.COLOR -> setColorProperty(data)
            }
        }
    }

    private fun setIntProperty(data: Pair<FilterProperty, Any>) {
        val intSpinner = Spinner<Int>(0, Int.MAX_VALUE, INT_STEP, INT_STEP)
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        intSpinner.valueFactory.value = data.second as Int
        intSpinner.isEditable = true
        intSpinner.editor.textProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
        }
        intSpinner.focusedProperty().addListener { _, _, _ ->
            controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
        }
        intSpinner.prefWidth = WIDTH_PIXELS
        graphic = intSpinner
    }

    private fun setFloatProperty(data: Pair<FilterProperty, Any>) {
        val doubleSpinner = Spinner<Double>(0.0, 1.0, FLOAT_STEP, FLOAT_STEP)
        doubleSpinner.valueFactory.converter = DoubleConverter()
        val controller = FlowKeyApplication.controller
        val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
        doubleSpinner.valueFactory.value = (data.second as Float).toDouble()
        doubleSpinner.isEditable = true
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

    private fun setColorProperty(data: Pair<FilterProperty, Any>) {
        val originalColor = (data.second as ByteArray).map { it.toUByte().toInt() / PIXEL_MULTIPLIER.toDouble() }
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
}
