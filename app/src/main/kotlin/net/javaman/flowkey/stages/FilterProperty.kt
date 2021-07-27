package net.javaman.flowkey.stages

enum class FilterProperty(
    val listName: String,
    val dataType: FilterPropertyType
) {
    TOLERANCE("Tolerance", FilterPropertyType.FLOAT),
    ITERATIONS("Iterations", FilterPropertyType.INT),
    COLOR_KEY("Color Key", FilterPropertyType.COLOR),
    REPLACEMENT_KEY("Replacement Key", FilterPropertyType.COLOR),
    BLOCK_SIZE("Block Size", FilterPropertyType.INT)
}

enum class FilterPropertyType {
    COLOR,
    FLOAT,
    INT
}
