package net.javaman.flowkey.stages

import javafx.geometry.Pos
import javafx.scene.control.Label
import javafx.scene.control.TableCell

class GeneralSettingsNameTableCell<T> : TableCell<GeneralSettingsProperty, T>() {
    init {
        this.alignment = Pos.CENTER_LEFT
        text = null
    }

    override fun updateItem(item: T, empty: Boolean) {
        super.updateItem(item, empty)
        graphic = if (empty) {
            null
        } else {
            val data = tableView.items[tableRow.index] as GeneralSettingsProperty
            val label = Label(data.listName)
            label
        }
    }
}