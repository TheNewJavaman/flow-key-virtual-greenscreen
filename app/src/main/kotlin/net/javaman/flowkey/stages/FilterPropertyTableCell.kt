package net.javaman.flowkey.stages

import javafx.beans.property.Property
import javafx.beans.value.ObservableValue
import javafx.scene.control.Spinner
import javafx.scene.control.TableCell
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty
import net.javaman.flowkey.hardwareapis.common.AbstractFilterPropertyType

class FilterPropertyTableCell<T> : TableCell<Pair<AbstractFilterProperty, Any>, T>() {
    companion object {
        const val INT_STEP = 1

        const val FLOAT_STEP = 0.01
    }

    private var observableValue: ObservableValue<T>? = null

    private val intSpinner = Spinner<Int>(0, Int.MAX_VALUE, 0, INT_STEP)

    private val floatSpinner = Spinner<Float>(0.0, 1.0, 0.0, FLOAT_STEP)

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        if (empty) {
            graphic = null
        } else {
            observableValue?.let {
                intSpinner.valueFactory.valueProperty().unbindBidirectional(it as Property<Int>)
                floatSpinner.valueFactory.valueProperty().unbindBidirectional(it as Property<Float>)
            }
            observableValue = tableColumn.getCellObservableValue(index)
            observableValue?.let {
                val data = (tableView.items[tableRow.index] as Pair<AbstractFilterProperty, Any>)
                when (data.first.dataType) {
                    AbstractFilterPropertyType.INT -> {
                        graphic = intSpinner
                        intSpinner.valueFactory.valueProperty().bindBidirectional(observableValue as Property<Int>)
                    }
                    AbstractFilterPropertyType.FLOAT -> {
                        graphic = floatSpinner
                        floatSpinner.valueFactory.valueProperty().bindBidirectional(observableValue as Property<Float>)
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
}
