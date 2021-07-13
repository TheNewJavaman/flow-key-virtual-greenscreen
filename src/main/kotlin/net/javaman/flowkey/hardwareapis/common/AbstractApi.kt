package net.javaman.flowkey.hardwareapis.common

interface AbstractApi {
    interface AbstractApiConsts {
        val LIST_NAME: String

        fun getFilters(api: AbstractApi): Map<String, AbstractFilter>
    }

    fun close()
}