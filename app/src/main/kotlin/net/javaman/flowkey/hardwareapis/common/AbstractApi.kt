package net.javaman.flowkey.hardwareapis.common

interface AbstractApi {
    interface AbstractApiConsts {
        val listName: String
    }

    fun getFilters(): Map<String, AbstractFilter>

    fun close()
}
