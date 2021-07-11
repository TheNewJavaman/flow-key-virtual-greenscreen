package net.javaman.flowkey

const val DEFAULT_WIDTH_PIXELS = 1280
const val DEFAULT_HEIGHT_PIXELS = 720
const val COLOR_DEPTH = 3
const val ONE_SECOND_MS = 1000

fun Float.format(digits: Int) = "%.${digits}f".format(this)