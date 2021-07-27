package net.javaman.flowkey.hardwareapis.common

import net.javaman.flowkey.stages.FilterProperty

interface AbstractFilter {
    fun getProperties(): Map<FilterProperty, Any>

    fun setProperty(listName: String, newValue: Any)

    fun apply(inputBuffer: ByteArray): ByteArray
}

interface AbstractFilterConsts {
    val listName: String
}
