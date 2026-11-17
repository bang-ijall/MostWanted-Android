// Hooks for default.xbe. A strong definition replaces the recompiled function at that address, for example:
//
// #include "xdk/hle.h"
//
// X86_EXTERN_FUNC(__imp__sub_00012345);
// X86_FUNC(sub_00012345) {
//     __imp__sub_00012345(ctx, base);  // the original
// }


#include <cstdint>
#include <cstring>

#include "core/log.h"
#include "frame.h"

// HAND PATCH in generated/game/x86_recomp.24.cpp (sub_000C7740 at 000C784B; redo it if the code is generated again):
// in widescreen mode (the console settings say widescreen) the camera takes height / width = 9/16. By default it
// takes the phone screen's own height / width instead (more view at the sides, full screen, true shapes).
//
// The HUD and menus are laid out on a 480 line screen whose width, in widescreen mode, is the float at 0x00427C00:
// 853 (480 * 16/9, read at 000D4897 and 000D8339 to build the 2D projection). With the camera at the screen's shape
// it takes 480 / (height / width), so gauges stay round instead of stretched across the wider screen.
uint32_t nfsmw_wide_aspect_bits(uint8_t* base) {
    const float hw = platform::ScreenAspect();
    const float v = hw > 0.2f && hw < 1.0f ? hw : 0.5625f;
    const float hud_width = v == 0.5625f ? 853.f : 480.f / v;
    std::memcpy(base + 0x00427C00u, &hud_width, 4);
    static bool logged = false;
    if (!logged) {
        logged = true;
        LOGI("camera: height / width %.4f (the game's widescreen is 0.5625), HUD width %.1f (853)", v, hud_width);
    }
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    return bits;
}
