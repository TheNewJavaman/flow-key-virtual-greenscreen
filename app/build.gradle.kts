import org.jetbrains.kotlin.gradle.tasks.KotlinCompile
import java.util.*

fun getOsString(): String {
    val vendor = System.getProperty("java.vendor")
    return if ("The Android Project" == vendor) {
        "android"
    } else {
        var osName = System.getProperty("os.name")
        osName = osName.toLowerCase(Locale.ENGLISH)
        when {
            osName.startsWith("windows") -> "windows"
            osName.startsWith("mac os") -> "apple"
            osName.startsWith("linux") -> "linux"
            osName.startsWith("sun") -> "sun"
            else -> "unknown"
        }
    }
}

fun getArchString(): String {
    var osArch = System.getProperty("os.arch")
    osArch = osArch.toLowerCase(Locale.ENGLISH)
    return when {
        "i386" == osArch || "x86" == osArch || "i686" == osArch -> "x86"
        osArch.startsWith("amd64") || osArch.startsWith("x86_64") -> "x86_64"
        osArch.startsWith("arm64") -> "arm64"
        osArch.startsWith("arm") -> "arm"
        "ppc" == osArch || "powerpc" == osArch -> "ppc"
        osArch.startsWith("ppc") -> "ppc_64"
        osArch.startsWith("sparc") -> "sparc"
        osArch.startsWith("mips64") -> "mips64"
        osArch.startsWith("mips") -> "mips"
        osArch.contains("risc") -> "risc"
        else -> "unknown"
    }
}

plugins {
    application
    kotlin("jvm") version "1.5.21"
    id("org.openjfx.javafxplugin") version "0.0.10"
    id("io.gitlab.arturbosch.detekt") version "1.18.0-RC1"
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("org.jcuda:jcuda:11.2.0") { isTransitive = false }
    implementation("org.jcuda:jcuda-natives:11.2.0:${getOsString() + "-" + getArchString()}")
    implementation("org.jocl:jocl:2.0.4")
    implementation("org.openjfx:javafx-controls:16")
    implementation("org.openjfx:javafx-fxml:16")
    implementation("org.openjfx:javafx-swing:16")
    implementation("org.slf4j:slf4j-simple:1.7.29")
    implementation("io.github.microutils:kotlin-logging-jvm:2.0.10")
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