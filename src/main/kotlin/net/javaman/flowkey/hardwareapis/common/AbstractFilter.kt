package net.javaman.flowkey.hardwareapis.common

interface AbstractFilter {
    interface AbstractFilterConsts {
        val LIST_NAME: String
    }

    fun apply(inputBuffer: ByteArray): ByteArray
}