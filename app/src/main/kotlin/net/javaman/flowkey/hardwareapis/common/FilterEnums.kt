package net.javaman.flowkey.hardwareapis.common

@Suppress("Unused")
enum class ColorSpace(val i: Int) {
    BLUE(0),
    GREEN(1),
    RED(2),
    ALL(3)
}

@Suppress("Unused")
enum class FloatOption(val i: Int) {
    PERCENT_TOLERANCE(0),
    GRADIENT_TOLERANCE(1)
}

@Suppress("Unused")
enum class IntOption(val i: Int) {
    COLOR_SPACE(0),
    WIDTH(1),
    HEIGHT(2)
}