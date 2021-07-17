// Adapted from https://stackoverflow.com/questions/56029250/how-can-i-set-decimal-places-in-javafx-spinner

package net.javaman.flowkey.stages

import javafx.geometry.Pos
import javafx.scene.control.Spinner
import javafx.scene.control.TableCell
import javafx.util.StringConverter
import net.javaman.flowkey.FlowKeyApplication
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty
import net.javaman.flowkey.hardwareapis.common.AbstractFilterPropertyType
import java.text.DecimalFormat
import java.text.ParseException


class FilterPropertyValueTableCell<T> : TableCell<Pair<AbstractFilterProperty, Any>, T>() {
    companion object {
        const val INT_STEP = 1

        const val FLOAT_STEP = 0.005
    }

    private val intSpinner = Spinner<Int>(0, Int.MAX_VALUE, 0, INT_STEP)

    private val doubleSpinner = Spinner<Double>(0.0, 1.0, 0.0, FLOAT_STEP)

    init {
        this.alignment = Pos.CENTER_RIGHT
        text = null
        doubleSpinner.valueFactory.converter = DoubleConverter()
    }

    class DoubleConverter : StringConverter<Double>() {
        private val decimalFormat = DecimalFormat("#.###")

        override fun toString(doubleValue: Double): String? = decimalFormat.format(doubleValue)

        override fun fromString(string: String?): Double? {
            var mutableString = string
            return try {
                if (mutableString == null) return null
                mutableString = mutableString.trim { it <= ' ' }
                if (mutableString.isEmpty()) null
                else decimalFormat.parse(mutableString).toDouble()
            } catch (ex: ParseException) {
                throw RuntimeException(ex)
            }
        }
    }

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        if (empty) {
            graphic = null
        } else {
            val data = (tableView.items[tableRow.index] as Pair<AbstractFilterProperty, Any>)
            when (data.first.dataType) {
                AbstractFilterPropertyType.INT -> {
                    graphic = intSpinner
                    val controller = FlowKeyApplication.controller
                    val selectedId = controller.filtersListView.selectionModel.selectedItem.id.toInt()
                    intSpinner.valueFactory.value = data.second as Int
                    intSpinner.editor.textProperty().addListener { _, _, _ ->
                        controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
                    }
                    intSpinner.focusedProperty().addListener { _, _, _ ->
                        controller.filters[selectedId].setProperty(data.first.listName, intSpinner.valueFactory.value)
                    }
                }
                AbstractFilterPropertyType.FLOAT -> {
                    graphic = doubleSpinner
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
                }
                AbstractFilterPropertyType.COLOR_SPACE -> {
                    println("color space")
                }
                AbstractFilterPropertyType.COLOR -> {
                    println("color")
                }
            }
        }
    }
}
