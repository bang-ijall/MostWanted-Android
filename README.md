# NFS Most Wanted (Xbox) – native Android port

Static recompilation of the original-Xbox *Need for Speed: Most Wanted* `default.xbe` to C++, built into an arm64
Android app. Built on the MIT-licensed [XboxXBERecomp](https://github.com/TGP482/XboxXBERecomp) SDK (in `sdk/`, XDK
5849 — the version this game uses). No game files or generated code are in this repo; you supply your own `default.xbe`.

## Build (pick one)
- **GitHub Actions:** push this repo, run *Build Android APK* manually with `xbe_url` (a download link to your
  `default.xbe`), or set the repo secret `XBE_URL`. The APK is the build artifact.
- **Local:** `./build.sh /path/to/default.xbe` (Linux/macOS/WSL; needs clang, cmake, ninja, JDK 17, Android SDK + NDK 27).
- **Android Studio / VS Code:** put `default.xbe` in `assets/`, run the host tools (VS Code tasks 1–2), then open the
  folder in Android Studio and Run, or use task 3/4.

## Run
Copy `default.xbe` **and the game's data folders** (or the extracted disc contents) to
`/sdcard/Android/media/com.xboxrecomp.game/game/`, then start the app. Touch controls are on screen; gamepads work.
Logs: `adb logcat -s nfsmw`.

## Status
| Part | State |
|---|---|
| x86 → C++ recompiler | done (28,848 functions, whole XBE) |
| Xbox kernel, XDK (D3D8/DSound/XAPI) HLE | from SDK; boots on Linux to first file access |
| Audio (SDL3), input (touch + gamepad), app shell | written, untested on a device |
| Renderer | **CPU software renderer (slow, for bring-up).** Vulkan renderer not written yet |
| Android build | untested (no NDK in the authoring environment) |
