import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
    application
    kotlin("jvm") version "1.5.21"
    id("org.openjfx.javafxplugin") version "0.0.10"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jcuda:jcuda:11.2.0") { isTransitive = false }
    implementation("org.jocl:jocl:2.0.4")
    implementation("org.openjfx:javafx-controls:16")
    implementation("org.openjfx:javafx-fxml:16")
    implementation("org.openjfx:javafx-swing:16")

    // Download OpenCV jar from the official site; not available on Maven
    implementation(files("libs/opencv-452.jar"))
}

javafx {
    modules = listOf("javafx.controls", "javafx.fxml", "javafx.swing")
}

application {
    mainClass.set("net.javaman.flowkey.FlowKeyApplication")
    // Download OpenCV dll from the official site; platform-specific
    applicationDefaultJvmArgs = listOf("-Djava.library.path=libs")
}

val compileKotlin: KotlinCompile by tasks
compileKotlin.kotlinOptions {
    jvmTarget = "16"
}