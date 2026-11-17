import java.net.URL
import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val sdlVersion = "3.2.8"
val sdlDir = layout.projectDirectory.dir("third_party/SDL3-$sdlVersion")

// Downloads SDL3 (C sources for the NDK build + its Java glue) once.
val fetchSdl by tasks.registering {
    val tgz = layout.buildDirectory.file("sdl-$sdlVersion.tgz")
    outputs.dir(sdlDir)
    doLast {
        if (!sdlDir.asFile.resolve("CMakeLists.txt").exists()) {
            val f = tgz.get().asFile
            f.parentFile.mkdirs()
            URL("https://github.com/libsdl-org/SDL/releases/download/release-$sdlVersion/SDL3-$sdlVersion.tar.gz")
                .openStream().use { i -> f.outputStream().use { o -> i.copyTo(o) } }
            copy { from(tarTree(resources.gzip(f))); into(layout.projectDirectory.dir("third_party")) }
        }
    }
}
tasks.matching { it.name.startsWith("preBuild") }.configureEach { dependsOn(fetchSdl) }
tasks.matching { it.name.contains("ExternalNativeBuild") || it.name.startsWith("configureCMake") }
    .configureEach { dependsOn(fetchSdl) }

android {
    namespace = "com.jall.ea.nfsmw"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.jall.ea.nfsmw"
        minSdk = 26            // Vulkan 1.1 + 64-bit only
        targetSdk = 35
        versionCode = 126111662
        versionName = "0.9.0"
        setProperty("archivesBaseName", "NFSMW-$versionName")
        ndk { abiFilters += listOf("arm64-v8a") }
        externalNativeBuild {
            cmake { arguments += listOf("-DANDROID_STL=c++_static", "-DANDROID_ARM_NEON=ON") }
        }
    }
    externalNativeBuild {
        cmake { path = file("../CMakeLists.txt"); version = "3.22.1" }
    }
    sourceSets["main"].java.srcDir(sdlDir.dir("android-project/app/src/main/java"))
    // release signing from keystore/keystore.properties (see that file: the key must stay the same for updates)
    val keyProps = Properties().apply {
        val f = rootProject.file("keystore/keystore.properties")
        if (f.exists()) f.inputStream().use { load(it) }
    }
    signingConfigs {
        if (keyProps.getProperty("storeFile") != null) create("release") {
            storeFile = file(keyProps.getProperty("storeFile"))
            storePassword = keyProps.getProperty("storePassword")
            keyAlias = keyProps.getProperty("keyAlias")
            keyPassword = keyProps.getProperty("keyPassword")
        }
    }
    buildTypes {
        release {
            // the Kotlin library's unused classes out (SDL's Java glue is kept whole: its native side calls it by name)
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            isDebuggable = false
            isJniDebuggable = false
            signingConfig = signingConfigs.findByName("release")
        }
    }
    // native libraries compressed in the APK (about half the download; Android unpacks them when installing)
    packaging { jniLibs { useLegacyPackaging = true } }
    compileOptions { sourceCompatibility = JavaVersion.VERSION_17; targetCompatibility = JavaVersion.VERSION_17 }
    kotlinOptions { jvmTarget = "17" }
}
