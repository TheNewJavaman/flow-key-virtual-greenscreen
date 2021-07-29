package net.javaman.flowkey.hardwareapis.common

interface AbstractApplyBitmap {
    var originalBuffer: ByteArray

    var replacementKey: ByteArray

    fun apply(inputBuffer: ByteArray): ByteArray
}
