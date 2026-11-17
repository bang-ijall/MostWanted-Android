# The game as a native library: Android (libnfsmw.so, loaded by SDL's activity) or a Linux executable for testing.
# Needs generated/game/ (run tools: xbe_symbols + xbe_recomp on your default.xbe first).
cmake_minimum_required(VERSION 3.22)
project(nfsmw LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SDK ${CMAKE_CURRENT_LIST_DIR}/sdk)
set(GEN ${CMAKE_CURRENT_LIST_DIR}/generated/game)
if(NOT EXISTS ${GEN}/x86_func_mapping.cpp)
    message(FATAL_ERROR "generated/game is missing: run xbe_symbols and xbe_recomp on your default.xbe (see README)")
endif()

# SDL3 (source tree fetched by Gradle on Android, or -DSDL_SRC=... on Linux)
if(NOT SDL_SRC)
    set(SDL_SRC ${CMAKE_CURRENT_LIST_DIR}/app/third_party/SDL3-3.2.8)
endif()
if(NOT EXISTS ${SDL_SRC}/CMakeLists.txt)
    # Android Studio's sync runs CMake before Gradle's download task, so fetch SDL here as well
    set(SDL_TGZ ${CMAKE_BINARY_DIR}/SDL3-3.2.8.tar.gz)
    message(STATUS "downloading SDL3 3.2.8")
    file(DOWNLOAD https://github.com/libsdl-org/SDL/releases/download/release-3.2.8/SDL3-3.2.8.tar.gz ${SDL_TGZ}
         STATUS SDL_DL_STATUS TLS_VERIFY ON)
    list(GET SDL_DL_STATUS 0 SDL_DL_CODE)
    if(NOT SDL_DL_CODE EQUAL 0)
        message(FATAL_ERROR "could not download SDL3 (${SDL_DL_STATUS}); download it manually and extract to ${SDL_SRC}")
    endif()
    get_filename_component(SDL_PARENT ${SDL_SRC} DIRECTORY)
    file(ARCHIVE_EXTRACT INPUT ${SDL_TGZ} DESTINATION ${SDL_PARENT})
endif()
set(SDL_STATIC OFF CACHE BOOL "" FORCE)
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
add_subdirectory(${SDL_SRC} SDL3-build EXCLUDE_FROM_ALL)

file(GLOB RUNTIME_SRC CONFIGURE_DEPENDS ${SDK}/runtime/*/*.cpp)
list(FILTER RUNTIME_SRC EXCLUDE REGEX "/runtime/app/|/gpu/d3d12\\.cpp|/audio/xaudio2\\.cpp|/gpu/texturedumper\\.cpp")
file(GLOB PLATFORM_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/platform/*.cpp ${CMAKE_CURRENT_LIST_DIR}/platform/ref/*.cpp)
file(GLOB GAME_SRC CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/src/*/*.cpp)
file(GLOB GEN_SRC CONFIGURE_DEPENDS ${GEN}/*.cpp)
set_source_files_properties(${GEN_SRC} PROPERTIES COMPILE_OPTIONS "-O2;-g0;-w")

set(APP_SOURCES ${RUNTIME_SRC} ${PLATFORM_SRC} ${GAME_SRC} ${GEN_SRC})

# glslang from the NDK's sources: the Vulkan renderer turns its generated GLSL into SPIR-V on the device
if(ANDROID)
    set(GLSLANG_DIR ${CMAKE_ANDROID_NDK}/sources/third_party/shaderc/third_party/glslang)
    file(GLOB GLSLANG_SRC
        ${GLSLANG_DIR}/glslang/GenericCodeGen/*.cpp
        ${GLSLANG_DIR}/glslang/MachineIndependent/*.cpp
        ${GLSLANG_DIR}/glslang/MachineIndependent/preprocessor/*.cpp
        ${GLSLANG_DIR}/glslang/OSDependent/Unix/ossource.cpp
        ${GLSLANG_DIR}/OGLCompilersDLL/InitializeDll.cpp
        ${GLSLANG_DIR}/SPIRV/GlslangToSpv.cpp
        ${GLSLANG_DIR}/SPIRV/InReadableOrder.cpp
        ${GLSLANG_DIR}/SPIRV/Logger.cpp
        ${GLSLANG_DIR}/SPIRV/SpvBuilder.cpp
        ${GLSLANG_DIR}/SPIRV/SpvPostProcess.cpp
        ${GLSLANG_DIR}/SPIRV/doc.cpp)
    add_library(glslang_lite STATIC ${GLSLANG_SRC})
    target_include_directories(glslang_lite PUBLIC ${GLSLANG_DIR} ${CMAKE_CURRENT_LIST_DIR}/third_party/glslang_gen)
    target_compile_definitions(glslang_lite PUBLIC GLSLANG_OSINCLUDE_UNIX)
    target_compile_options(glslang_lite PRIVATE -O2 -w -fno-exceptions)
endif()
if(ANDROID)
    add_library(nfsmw SHARED ${APP_SOURCES})
    target_link_libraries(nfsmw PRIVATE log android nativewindow EGL GLESv3 vulkan glslang_lite)
    # calls between the library's own functions go straight, not through the PLT
    target_link_options(nfsmw PRIVATE -Wl,-Bsymbolic-functions)
    set_target_properties(nfsmw PROPERTIES OUTPUT_NAME Game)  # libGame.so
    # a smaller library: only SDL_main exported, unused code and data dropped, relocations packed (Android 6+)
    target_link_options(nfsmw PRIVATE -Wl,--version-script=${CMAKE_CURRENT_LIST_DIR}/platform/exports.map
                        -Wl,--gc-sections -Wl,--pack-dyn-relocs=android)
    target_compile_options(nfsmw PRIVATE -ffunction-sections -fdata-sections)
else()
    add_executable(nfsmw ${APP_SOURCES})
    find_library(NFS_EGL NAMES EGL libEGL.so.1 REQUIRED)
    find_library(NFS_GLES NAMES GLESv2 libGLESv2.so.2 REQUIRED)
    target_link_libraries(nfsmw PRIVATE pthread ${NFS_EGL} ${NFS_GLES})
    target_include_directories(nfsmw PRIVATE ${SDL_SRC}/src/video/khronos)
endif()
target_include_directories(nfsmw PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/src ${SDK}/runtime ${GEN} ${CMAKE_CURRENT_LIST_DIR}/platform
    ${CMAKE_CURRENT_LIST_DIR}/platform/ref ${CMAKE_CURRENT_LIST_DIR}/platform/compat)
target_compile_definitions(nfsmw PRIVATE XBE_RAW_VERTICES=0 XBE_RENDER_THREAD=1)
# always optimised: Gradle's debug variant would otherwise build the whole runtime at -O0
# ARMv8.2 (every arm64 phone since about 2018): atomics as single instructions, not calls to helpers that pick them
target_compile_options(nfsmw PRIVATE -fno-strict-aliasing -ffp-contract=off -O2 -g0 -march=armv8.2-a)
target_link_libraries(nfsmw PRIVATE SDL3::SDL3)
