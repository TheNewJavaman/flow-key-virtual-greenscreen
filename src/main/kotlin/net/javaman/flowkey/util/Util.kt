package net.javaman.flowkey.util

fun Float.format(digits: Int) = "%.${digits}f".format(this)