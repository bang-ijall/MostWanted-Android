#!/bin/bash
# Local build (Linux/macOS/WSL): recompile your XBE, then build the APK.   usage: ./build.sh path/to/default.xbe
set -e
cd "$(dirname "$0")"
[ -n "$1" ] && cp "$1" assets/default.xbe
[ -f assets/default.xbe ] || { echo "usage: $0 path/to/default.xbe"; exit 2; }
cmake -S tools -B build/tools -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build/tools
build/tools/xbe_symbols assets/default.xbe sdk/xdk --library nfsmw.dll --out config/game.toml
build/tools/xbe_recomp config/game.toml --xdk sdk/xdk
gradle :app:assembleDebug
echo "APK: app/build/outputs/apk/debug/app-debug.apk"
