package net.javaman.flowkey.stages

enum class GeneralSettingsProperty(
    val listName: String,
    val dataType: GeneralSettingsPropertyType
) {
    API("Hardware API", GeneralSettingsPropertyType.API),
    CAMERA("Camera Device", GeneralSettingsPropertyType.CAMERA),
    RESOLUTION("Resolution", GeneralSettingsPropertyType.RESOLUTION)
}

enum class GeneralSettingsPropertyType {
    API,
    CAMERA,
    RESOLUTION
}
