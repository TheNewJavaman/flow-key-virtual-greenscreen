package net.javaman.flowkey.stages

import javafx.geometry.Pos
import javafx.scene.control.Label
import javafx.scene.control.TableCell
import net.javaman.flowkey.hardwareapis.common.AbstractFilterProperty

class FilterPropertyNameTableCell<T> : TableCell<Pair<AbstractFilterProperty, Any>, T>() {
    private val label = Label()

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        text = null
        if (empty) {
            graphic = null
        } else {
            val data = (tableView.items[tableRow.index] as Pair<AbstractFilterProperty, Any>)
            graphic = label
            label.text = data.first.listName
            this.alignment = Pos.CENTER_LEFT
        }
    }
}