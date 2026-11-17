# NFS: Most Wanted for Android

A native Android port of the original Xbox *Need for Speed: Most Wanted* (2005). The game's `default.xbe` is
statically recompiled from x86 to C++ and runs on arm64 phones with a Vulkan renderer, no emulator involved.

Built on the MIT-licensed [XboxXBERecomp](https://github.com/TGP482/XboxXBERecomp) SDK (`sdk/`, XDK 5849).

> **No game content is included.** This repository holds no game files and no recompiled game code. You need your
> own copy of the Xbox game.

## Features
- Vulkan 1.3 renderer at the screen's full resolution, with OpenGL ES 3.2 as a fallback
- Frame pacing locked to the display's refresh rate (up to 120 Hz)
- Import screen: pick the game's Xbox ISO (redump or XISO) and it is copied into the app
- On-screen Xbox-style controller, hidden while a physical gamepad is connected
- Saves in `Android/data/com.jall.ea.nfsmw/files/savedata`, easy to back up

## Requirements
- Android 8.0 or newer, arm64, with Vulkan 1.3 or OpenGL ES 3.2
- An ISO of NFS: Most Wanted for the original Xbox (the release whose `default.xbe` SHA-1 is
  `f636a16af04171089f879a4bed646e185c734233`)
- About 3 GB of free storage

## Install and play
1. Install the APK.
2. Open the app and tap **Pilih file ISO** ("pick ISO file"), then choose your game's ISO.
3. Wait for the import to finish; the game starts. The ISO can be deleted afterwards.

## Building from source
The recompiled code is not in the repository; it is generated from your own `default.xbe`.

```sh
./build.sh /path/to/default.xbe   # macOS / Linux / WSL: clang, cmake, ninja, JDK 17, Android SDK + NDK 27
```

`build.sh` builds the recompiler (`tools/`), writes the C++ into `generated/game/`, then builds the APK with Gradle.
The working build also relies on a few hand edits in the generated code (among them the `D3DDevice_BeginPush` /
`D3DDevice_EndPush` symbols, `00373E10` and `00373E30`, in `image_symbols.h`); running the generator again replaces
them, so keep a copy of `generated/` once it works.

Release signing reads `keystore/keystore.properties` (not in the repository). Without it Gradle builds an unsigned
release APK.

## Layout
| Path | What |
|---|---|
| `tools/` | x86 → C++ recompiler |
| `sdk/runtime/` | Xbox kernel and XDK (Direct3D 8, DirectSound, XAPI) reimplemented on the host |
| `platform/` | Android side: SDL3 window, Vulkan / GL ES renderers, audio, touch controller |
| `src/game/` | game-specific hooks |
| `app/` | Android app (Kotlin): ISO import screen and the game activity |

## Legal
Need for Speed: Most Wanted is a trademark of Electronic Arts. This project is not affiliated with or endorsed by
Electronic Arts or Microsoft. It contains no copyrighted game data; use it only with a copy of the game you own.

---

### Bahasa Indonesia
Port native NFS: Most Wanted (Xbox) untuk Android: kode game dikompilasi ulang dari x86 ke C++, dengan renderer
Vulkan di resolusi layar penuh dan hingga 120 fps. Repo ini **tidak berisi file game**; impor ISO Xbox milik Anda
sendiri lewat layar pertama aplikasi.
