package net.javaman.flowkey.hardwareapis.common

interface AbstractFilter {
    fun getProperties(): Map<AbstractFilterProperty, Any>

    fun setProperty(listName: String, newValue: Any)

    fun apply(inputBuffer: ByteArray): ByteArray
}

interface AbstractFilterConsts {
    val listName: String
}

enum class AbstractFilterProperty(
    val listName: String,
    val dataType: AbstractFilterPropertyType
) {
    TOLERANCE("Tolerance", AbstractFilterPropertyType.FLOAT),
    ITERATIONS("Iterations", AbstractFilterPropertyType.INT),
    COLOR_KEY("Color Key", AbstractFilterPropertyType.COLOR),
    REPLACEMENT_KEY("Replacement Key", AbstractFilterPropertyType.COLOR),
    COLOR_SPACE("Color Space", AbstractFilterPropertyType.COLOR_SPACE),
    BLOCK_SIZE("Block Size", AbstractFilterPropertyType.INT)
}

enum class AbstractFilterPropertyType {
    COLOR,
    FLOAT,
    INT,
    COLOR_SPACE
}
