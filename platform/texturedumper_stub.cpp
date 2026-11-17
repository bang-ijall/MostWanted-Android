// Texture packs/dumps are a Windows-only extra (WIC). Stand-in for Linux/Android: never replaces, never dumps.
#include "gpu/texturedumper.h"
namespace gpu::texturedumper {
bool Update() { return false; }
uint32_t Generation() { return 0; }
bool Replacing() { return false; }
uint64_t Key(const TextureDesc&, const FormatInfo&, const uint8_t*, const uint32_t*) { return 0; }
void Dump(uint64_t, const TextureDesc&, const uint8_t*, const uint32_t*) {}
Lookup Take(uint64_t, Image&) { return Lookup::None; }
}  // namespace gpu::texturedumper
