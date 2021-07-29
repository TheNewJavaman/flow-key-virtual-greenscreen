package net.javaman.flowkey.hardwareapis.common

interface AbstractApi {
    fun getFilters(): Map<String, AbstractFilter>

    fun getApplyBitmap(): AbstractApplyBitmap

    fun close()
}

interface AbstractApiConsts {
    val listName: String
}
