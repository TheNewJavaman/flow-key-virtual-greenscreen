package net.javaman.flowkey.hardwareapis.common

interface AbstractApi {
    interface AbstractApiConsts {
        val LIST_NAME: String
    }

    fun getFilters(): Map<String, AbstractFilter>

    fun close()
}