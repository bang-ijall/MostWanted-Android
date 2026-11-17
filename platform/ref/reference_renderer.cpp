// CPU reference renderer. Surfaces, textures, vertex programs, register combiners and fixed function stages are
// evaluated per the XDK's definitions, decoded the way gpu/shaders.cpp decodes them for HLSL.
#include "reference_renderer.h"

#include <chrono>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <unordered_map>
#include <set>
#include <memory>
#include <vector>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include "gpu/shaders.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"

namespace reference {
DrawDbg g_dbg;
std::string g_dump_dir;
std::atomic<uint64_t> g_prep_us{0}, g_pixels{0}, g_tris{0}, g_img_us{0}, g_vs_us{0}, g_nontri{0}, g_prepare_us{0};
static DrawPacket* g_packet = nullptr;
static bool g_gpu_mode = false;
static uint64_t g_image_ids = 0;
static TargetTest g_target_test = nullptr;
static bool g_allow_raw = false;

using namespace gpu;

namespace {

struct Vec {
    float c[4] = {};
    float& operator[](int i) { return c[i]; }
    float operator[](int i) const { return c[i]; }
};
Vec V(float x, float y, float z, float w) { return {{x, y, z, w}}; }
float Sat(float v) { return v < 0 ? 0 : v > 1 ? 1 : v; }
float AsFloat(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}
// D3DCOLOR to rgba
Vec Unpack(uint32_t argb) {
    return V(((argb >> 16) & 255) / 255.f, ((argb >> 8) & 255) / 255.f, (argb & 255) / 255.f, (argb >> 24) / 255.f);
}
Vec Rgb565(uint16_t c) { return V(((c >> 11) & 31) / 31.f, ((c >> 5) & 63) / 63.f, (c & 31) / 31.f, 1); }
bool Depth16Format(uint32_t f) { return f == 0x2C || f == 0x2D || f == 0x30 || f == 0x31; }

// swizzled texel order: x and y bits interleaved from bit 0, x first, while either axis has bits left
uint32_t SwizzleOffset(uint32_t x, uint32_t y, uint32_t w, uint32_t h) {
    uint32_t xbits = 0, ybits = 0;
    while ((1u << xbits) < w) xbits++;
    while ((1u << ybits) < h) ybits++;
    uint32_t out = 0, at = 0;
    for (uint32_t i = 0; i < xbits || i < ybits; i++) {
        if (i < xbits) out |= ((x >> i) & 1) << at++;
        if (i < ybits) out |= ((y >> i) & 1) << at++;
    }
    return out;
}

// a surface in guest memory

struct Plane {
    bool ok = false, swizzled = false;
    uint32_t data = 0, format = 0, w = 0, h = 0, pitch = 0, bytes = 4;

    uint8_t* At(uint32_t x, uint32_t y) const {
        uint32_t off = swizzled ? SwizzleOffset(x, y, w, h) * bytes : y * pitch + x * bytes;
        return static_cast<uint8_t*>(kernel::HostPtr(data)) + off;
    }
    uint32_t End() const { return data + pitch * h; }
    bool Depth16() const { return Depth16Format(format); }

    Vec Load(uint32_t x, uint32_t y) const {
        const uint8_t* p = At(x, y);
        if (bytes != 4) {
            uint16_t c;
            std::memcpy(&c, p, 2);
            return Rgb565(c);
        }
        uint32_t c;
        std::memcpy(&c, p, 4);
        Vec v = Unpack(c);
        if (format == 0x07 || format == 0x1E) v[3] = 1;  // X8R8G8B8
        return v;
    }
    // mask: 1 blue, 2 green, 4 red, 8 alpha
    void Store(uint32_t x, uint32_t y, Vec c, uint32_t mask) const {
        uint8_t* p = At(x, y);
        auto q = [](float v, float top) { return std::lround(Sat(v) * top); };
        if (bytes != 4) {
            uint16_t v = uint16_t(q(c[0], 31)) << 11 | uint16_t(q(c[1], 63)) << 5 | uint16_t(q(c[2], 31));
            std::memcpy(p, &v, 2);
            return;
        }
        const int channel[4] = {2, 1, 0, 3};  // bytes b g r a
        for (int i = 0; i < 4; i++)
            if (mask >> i & 1) p[i] = uint8_t(q(c[channel[i]], 255));
    }
    void LoadDepth(uint32_t x, uint32_t y, uint32_t& z, uint32_t& stencil) const {
        const uint8_t* p = At(x, y);
        if (Depth16()) {
            uint16_t v;
            std::memcpy(&v, p, 2);
            z = v;
            stencil = 0;
            return;
        }
        uint32_t v;
        std::memcpy(&v, p, 4);
        z = v >> 8;
        stencil = v & 0xFF;
    }
    void StoreDepth(uint32_t x, uint32_t y, uint32_t z, uint32_t stencil) const {
        uint8_t* p = At(x, y);
        if (Depth16()) {
            uint16_t v = uint16_t(z);
            std::memcpy(p, &v, 2);
            return;
        }
        uint32_t v = z << 8 | (stencil & 0xFF);
        std::memcpy(p, &v, 4);
    }
};

Plane OpenPlane(const Surface& s) {
    Plane p;
    TextureDesc d;
    if (!s.header || !Describe(s.header, d)) return p;
    p.ok = true;
    p.data = d.data;
    p.format = d.format;
    p.w = d.width;
    p.h = d.height;
    p.swizzled = !d.linear;
    p.bytes = Format(d.format).bits / 8;
    p.pitch = d.linear ? d.pitch : d.width * p.bytes;
    return p;
}

// textures, level 0 decoded to floats

struct Image {
    uint32_t w = 1, h = 1, faces = 1;
    bool depth16 = false;
    std::vector<Vec> texels[6];
    uint32_t lo = 0, hi = 0;  // guest bytes it was read from
    uint64_t sig = 0;         // a small texture: hash of all its bytes (the palette is in the key)
    uint64_t stripe[8] = {};  // a big one: hashes of its 8 byte stripes (offset 8k in every 64 bytes), checked in turns
    uint32_t next_stripe = 0;
    uint32_t verified = 0;    // the frame the signature was last compared in
    size_t bytes = 0;
    uint64_t id = 0;
    uint64_t key = 0;  // g_images key: address, format, size, palette
    std::vector<uint8_t> rgba8[6];  // the GPU path's copies of the faces
};
std::unordered_map<uint64_t, std::shared_ptr<Image>> g_images;  // a lookup per texture stage per draw: hashed, not a tree
// the last images found, in front of g_images (its chains missed the cache on every stage of every draw); an entry
// holds while g_images loses nothing (an element stays where it is as others come in)
uint64_t g_images_gen = 1;
struct FrontImage {
    uint64_t key = 0, gen = 0;
    std::shared_ptr<Image>* image = nullptr;
};
FrontImage g_front_images[256];
uint32_t g_frame = 1;
size_t g_image_bytes = 0;

// one word per cache line: cheap enough to run once per texture per frame
// 8 bytes in every 64, which is quick for big textures; a small one is hashed whole, as the title may change only a
// few of its bytes (NFS Most Wanted draws the race route into its minimap's 128x128 tiles with the CPU, which the
// sampled hash missed: the minimap showed no route)
// a texture up to this size is hashed whole at every check; bigger ones a quarter at a time (two stripes of eight)
constexpr size_t kWholeHash = 4u << 10;
// one stripe of a big texture: the 8 bytes at offset 8k of every 64. A check hashes two stripes in turn, so every
// byte is compared within four checks (12 frames): a sampled hash alone missed a few changed bytes in a big texture
// for good (the race map kept the last race's route)
uint64_t StripeHash(const uint8_t* p, size_t n, uint32_t k) {
    uint64_t h = 1469598103934665603ull ^ k;
    for (size_t i = size_t(k) * 8; i + 8 <= n; i += 64) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    return h ^ n;
}
uint64_t SampleHash(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const size_t step = n <= (64u << 10) ? 8 : 64;
    for (size_t i = 0; i + 8 <= n; i += step) {
        uint64_t w;
        std::memcpy(&w, p + i, 8);
        h = (h ^ w) * 1099511628211ull;
    }
    return h ^ n;
}

// a P8 texture's colours: its stage's D3DPalette, 256 entries less by the size code in the top two bits
struct Palette {
    uint32_t colours[256] = {};
    uint32_t va = 0;
    bool ok = false;
};
Palette ReadPalette(uint32_t header) {
    Palette pal;
    if (!header) return pal;
    pal.va = mem::VirtualFromPhysical(kernel::R32(header + 4));
    uint32_t code = kernel::R32(header) >> 30;
    std::memcpy(pal.colours, kernel::HostPtr(pal.va), (256u >> code) * 4);
    pal.ok = true;
    return pal;
}

// DXT1/3/5 blocks
void DecodeBlocks(const TextureDesc& d, Host host, const std::vector<uint8_t>& in, std::vector<Vec>& out) {
    uint32_t bw = std::max(1u, (d.width + 3) / 4), bh = std::max(1u, (d.height + 3) / 4);
    uint32_t block = host == Host::BC1 ? 8 : 16;
    out.assign(size_t(d.width) * d.height, Vec{});
    for (uint32_t by = 0; by < bh; by++)
        for (uint32_t bx = 0; bx < bw; bx++) {
            const uint8_t* b = &in[(size_t(by) * bw + bx) * block];
            const uint8_t* cb = host == Host::BC1 ? b : b + 8;
            uint16_t e0 = uint16_t(cb[0] | cb[1] << 8), e1 = uint16_t(cb[2] | cb[3] << 8);
            uint32_t sel = uint32_t(cb[4] | cb[5] << 8 | cb[6] << 16 | uint32_t(cb[7]) << 24);
            Vec pal[4] = {Rgb565(e0), Rgb565(e1)};
            if (host != Host::BC1 || e0 > e1) {
                for (int i = 0; i < 3; i++) {
                    pal[2][i] = (2 * pal[0][i] + pal[1][i]) / 3;
                    pal[3][i] = (pal[0][i] + 2 * pal[1][i]) / 3;
                }
                pal[2][3] = pal[3][3] = 1;
            } else {
                for (int i = 0; i < 3; i++) pal[2][i] = (pal[0][i] + pal[1][i]) / 2;
                pal[2][3] = 1;
                pal[3] = V(0, 0, 0, 0);
            }
            float alpha[16];
            if (host == Host::BC2) {
                for (int i = 0; i < 16; i++) alpha[i] = ((b[i / 2] >> ((i & 1) * 4)) & 15) / 15.f;
            } else if (host == Host::BC3) {
                float ramp[8] = {b[0] / 255.f, b[1] / 255.f};
                if (b[0] > b[1]) {
                    for (int i = 1; i < 7; i++) ramp[i + 1] = ((7 - i) * ramp[0] + i * ramp[1]) / 7;
                } else {
                    for (int i = 1; i < 5; i++) ramp[i + 1] = ((5 - i) * ramp[0] + i * ramp[1]) / 5;
                    ramp[6] = 0;
                    ramp[7] = 1;
                }
                uint64_t idx = 0;
                for (int i = 0; i < 6; i++) idx |= uint64_t(b[2 + i]) << (8 * i);
                for (int i = 0; i < 16; i++) alpha[i] = ramp[(idx >> (3 * i)) & 7];
            }
            for (int i = 0; i < 16; i++) {
                uint32_t x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x >= d.width || y >= d.height) continue;
                Vec c = pal[(sel >> (2 * i)) & 3];
                if (host != Host::BC1) c[3] = alpha[i];
                out[size_t(y) * d.width + x] = c;
            }
        }
}

// DXT1/3/5 blocks straight to RGBA8 (R first), for the GPU path: no float image in between
void DecodeBlocks8(const TextureDesc& d, Host host, const std::vector<uint8_t>& in, std::vector<uint8_t>& out) {
    uint32_t bw = std::max(1u, (d.width + 3) / 4), bh = std::max(1u, (d.height + 3) / 4);
    uint32_t block = host == Host::BC1 ? 8 : 16;
    out.assign(size_t(d.width) * d.height * 4, 0);
    auto expand = [](uint16_t c, uint8_t* o) {
        uint32_t r = (c >> 11) & 31, g = (c >> 5) & 63, b = c & 31;
        o[0] = uint8_t((r * 255 + 15) / 31);
        o[1] = uint8_t((g * 255 + 31) / 63);
        o[2] = uint8_t((b * 255 + 15) / 31);
        o[3] = 255;
    };
    for (uint32_t by = 0; by < bh; by++)
        for (uint32_t bx = 0; bx < bw; bx++) {
            size_t at = (size_t(by) * bw + bx) * block;
            if (at + block > in.size()) return;
            const uint8_t* b = &in[at];
            const uint8_t* cb = host == Host::BC1 ? b : b + 8;
            uint16_t e0 = uint16_t(cb[0] | cb[1] << 8), e1 = uint16_t(cb[2] | cb[3] << 8);
            uint32_t sel = uint32_t(cb[4] | cb[5] << 8 | cb[6] << 16 | uint32_t(cb[7]) << 24);
            uint8_t pal[4][4];
            expand(e0, pal[0]);
            expand(e1, pal[1]);
            if (host != Host::BC1 || e0 > e1) {
                for (int i = 0; i < 3; i++) {
                    pal[2][i] = uint8_t((2 * pal[0][i] + pal[1][i] + 1) / 3);
                    pal[3][i] = uint8_t((pal[0][i] + 2 * pal[1][i] + 1) / 3);
                }
                pal[2][3] = pal[3][3] = 255;
            } else {
                for (int i = 0; i < 3; i++) pal[2][i] = uint8_t((pal[0][i] + pal[1][i] + 1) / 2);
                pal[2][3] = 255;
                pal[3][0] = pal[3][1] = pal[3][2] = pal[3][3] = 0;
            }
            uint8_t alpha[16];
            if (host == Host::BC2) {
                for (int i = 0; i < 16; i++) alpha[i] = uint8_t(((b[i / 2] >> ((i & 1) * 4)) & 15) * 17);
            } else if (host == Host::BC3) {
                uint32_t ramp[8] = {b[0], b[1]};
                if (b[0] > b[1]) {
                    for (int i = 1; i < 7; i++) ramp[i + 1] = ((7 - i) * ramp[0] + i * ramp[1] + 3) / 7;
                } else {
                    for (int i = 1; i < 5; i++) ramp[i + 1] = ((5 - i) * ramp[0] + i * ramp[1] + 2) / 5;
                    ramp[6] = 0;
                    ramp[7] = 255;
                }
                uint64_t idx = 0;
                for (int i = 0; i < 6; i++) idx |= uint64_t(b[2 + i]) << (8 * i);
                for (int i = 0; i < 16; i++) alpha[i] = uint8_t(ramp[(idx >> (3 * i)) & 7]);
            }
            for (int i = 0; i < 16; i++) {
                uint32_t x = bx * 4 + (i & 3), y = by * 4 + (i >> 2);
                if (x >= d.width || y >= d.height) continue;
                uint8_t* o = &out[(size_t(y) * d.width + x) * 4];
                std::memcpy(o, pal[(sel >> (2 * i)) & 3], 4);
                if (host != Host::BC1) o[3] = alpha[i];
            }
        }
}

// a P8 stage's colours as the GPU's texture palette register holds them (NV097_SET_TEXTURE_PALETTE: the colours'
// physical address in the top bits, the entry count code in bits 2-3, 0 for 256 entries)
Palette PaletteFromRegister(uint32_t reg) {
    Palette pal;
    if (!reg) return pal;
    pal.va = mem::VirtualFromPhysical(reg & ~0x3Fu);
    uint32_t code = (reg >> 2) & 3;
    std::memcpy(pal.colours, kernel::HostPtr(pal.va), (256u >> code) * 4);
    pal.ok = true;
    return pal;
}

// hw_palette: the stage's palette register; the GPU samples with that, which inlined SetPalette code moves without
// touching the device's palette pointer (NFS Most Wanted's finish line flare smoke decoded near black with the old one)
std::shared_ptr<Image> FindImage(uint32_t header, uint32_t palette_header, uint32_t hw_palette = 0) {
    TextureDesc d;
    if (!header || !Describe(header, d)) return nullptr;
    uint32_t pal_va = 0, pal_code = 0;
    if (d.format == 0x0B && hw_palette) {
        pal_va = mem::VirtualFromPhysical(hw_palette & ~0x3Fu);
        pal_code = (hw_palette >> 2) & 3;
    } else if (d.format == 0x0B && palette_header) {
        pal_va = mem::VirtualFromPhysical(kernel::R32(palette_header + 4));
        pal_code = kernel::R32(palette_header) >> 30;
    }
    // a P8 texture is keyed by its palette's colours, not where they are (the render thread hands draws copies of the
    // palette at addresses of their own). The colours and their hash are kept per palette address and read again every
    // 6 frames: reading and hashing 1 KB for every P8 stage of every draw was a few percent of the render thread
    struct PalEntry {
        Palette pal;
        uint64_t hash = 0;
        uint32_t read = 0;
    };
    static std::unordered_map<uint64_t, PalEntry> pal_cache;
    static const Palette no_palette;
    const Palette* palp = &no_palette;
    uint64_t pal_key = 0;
    if (d.format == 0x0B && pal_va) {
        if (pal_cache.size() > 4096) pal_cache.clear();  // copies at ever new addresses: start over now and then
        PalEntry& e = pal_cache[uint64_t(pal_va) << 2 | pal_code];
        if (!e.read || g_frame - e.read >= 6) {
            e.pal = hw_palette ? PaletteFromRegister(hw_palette) : ReadPalette(palette_header);
            e.hash = e.pal.ok ? SampleHash(reinterpret_cast<const uint8_t*>(e.pal.colours), sizeof(e.pal.colours)) : uint64_t(pal_va);
            e.read = g_frame;
        }
        palp = &e.pal;
        pal_key = e.hash;
    }
    const Palette& pal = *palp;
    uint64_t key = uint64_t(d.data) << 32 ^ uint64_t(d.format) << 24 ^ uint64_t(d.width) << 12 ^ d.height ^ (d.cube ? 1ull << 63 : 0) ^ pal_key * 0x9E3779B97F4A7C15ull;
    // a texture checked in the last 3 frames is not hashed again (about 300 textures a race frame took 1-2 ms of
    // hashing every frame; what the CPU writes into one, a movie frame or the minimap, shows within 3 frames)
    // checked again every 6 frames (50 ms at 120 fps): what the CPU writes into a texture shows by then
    FrontImage& front = g_front_images[(key ^ key >> 17 ^ key >> 41) & 255];
    if (front.gen == g_images_gen && front.key == key && g_frame - (*front.image)->verified < 6) return *front.image;
    auto hit = g_images.find(key);
    if (hit != g_images.end() && g_frame - hit->second->verified < 6) {
        front = {key, g_images_gen, &hit->second};
        return hit->second;
    }
    size_t total = d.cube ? FaceSize(d) * 6 : TotalSize(d);
    const uint8_t* bytes = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
    const bool whole = total <= kWholeHash;
    uint64_t sig = whole ? SampleHash(bytes, total) : 0;
    if (auto it = hit; it != g_images.end()) {
        Image& im = *it->second;
        bool same;
        if (whole) {
            same = im.sig == sig;
        } else {
            const uint32_t k0 = im.next_stripe & 7, k1 = (im.next_stripe + 1) & 7;
            im.next_stripe += 2;
            same = StripeHash(bytes, total, k0) == im.stripe[k0] && StripeHash(bytes, total, k1) == im.stripe[k1];
        }
        if (same) {
            im.verified = g_frame;
            return it->second;
        }
        g_image_bytes -= im.bytes;
        g_images.erase(it);
        g_images_gen++;
    }
    // the new image's signature: the whole hash, or every stripe
    auto sign = [&](Image& im) {
        im.sig = sig;
        if (!whole)
            for (uint32_t k = 0; k < 8; k++) im.stripe[k] = StripeHash(bytes, total, k);
    };
    if (g_image_bytes > (400u << 20)) {
        g_images.clear();
        g_images_gen++;
        g_image_bytes = 0;
    }
    auto img = std::make_shared<Image>();
    const FormatInfo& fi = Format(d.format);
    img->w = d.width;
    img->h = d.height;
    img->faces = d.cube ? 6 : 1;
    img->depth16 = Depth16Format(d.format);
    img->lo = d.data;
    img->hi = d.data + (d.cube ? FaceSize(d) * 6 : TotalSize(d));
    img->key = key;
    const uint8_t* guest = static_cast<const uint8_t*>(kernel::HostPtr(d.data));
    if (g_gpu_mode) {
        // straight to RGBA8: the float image cost a 16 byte texel and two conversions per texel, a stall each time
        // a scene loads its textures
        size_t total_bytes = 0;
        for (uint32_t f = 0; f < img->faces; f++) {
            std::vector<uint8_t> raw;
            uint32_t pitch;
            DecodeLevel(d, 0, f, guest, pal.ok ? pal.colours : nullptr, raw, pitch);
            std::vector<uint8_t>& out = img->rgba8[f];
            if (fi.compressed) {
                DecodeBlocks8(d, fi.host, raw, out);
            } else if (fi.host == Host::R32F) {
                out.assign(size_t(d.width) * d.height * 4, 0);
                for (size_t i = 0; i < size_t(d.width) * d.height && i * 4 + 4 <= raw.size(); i++) {
                    float v;
                    std::memcpy(&v, &raw[i * 4], 4);
                    out[i * 4] = uint8_t(std::lround(Sat(v) * 255.f));
                }
            } else {
                out.resize(size_t(d.width) * d.height * 4);
                for (uint32_t y = 0; y < d.height; y++) {
                    const uint8_t* row = &raw[size_t(y) * pitch];
                    uint8_t* o = &out[size_t(y) * d.width * 4];
                    for (uint32_t x = 0; x < d.width; x++, row += 4, o += 4) {
                        o[0] = row[2];
                        o[1] = row[1];
                        o[2] = row[0];
                        o[3] = row[3];
                    }
                }
            }
            total_bytes += out.size();
        }
        sign(*img);
        img->verified = g_frame;
        img->id = ++g_image_ids;
        img->bytes = total_bytes;
        g_image_bytes += img->bytes;
        g_images[key] = img;
        g_images_gen++;
        return img;
    }
    for (uint32_t f = 0; f < img->faces; f++) {
        std::vector<uint8_t> raw;
        uint32_t pitch;
        DecodeLevel(d, 0, f, guest, pal.ok ? pal.colours : nullptr, raw, pitch);
        std::vector<Vec>& out = img->texels[f];
        if (fi.compressed) {
            DecodeBlocks(d, fi.host, raw, out);
            continue;
        }
        out.resize(size_t(d.width) * d.height);
        if (fi.host == Host::R32F) {
            for (size_t i = 0; i < out.size(); i++) {
                float v;
                std::memcpy(&v, &raw[i * 4], 4);
                out[i] = V(v, 0, 0, 0);
            }
            continue;
        }
        for (uint32_t y = 0; y < d.height; y++)
            for (uint32_t x = 0; x < d.width; x++) {
                const uint8_t* p = &raw[size_t(y) * pitch + x * 4];
                out[size_t(y) * d.width + x] = V(p[2] / 255.f, p[1] / 255.f, p[0] / 255.f, p[3] / 255.f);
            }
    }
    sign(*img);
    img->verified = g_frame;
    img->id = ++g_image_ids;
    for (uint32_t f = 0; f < img->faces; f++) img->bytes += img->texels[f].size() * sizeof(Vec);
    g_image_bytes += img->bytes;
    g_images[key] = img;
    g_images_gen++;
    return img;
}

// guest bytes written: the images read from them are stale
void Written(uint32_t lo, uint32_t hi) {
    for (auto it = g_images.begin(); it != g_images.end();) {
        if (it->second->lo < hi && lo < it->second->hi) {
            g_image_bytes -= it->second->bytes;
            it = g_images.erase(it);
            g_images_gen++;
        }
        else ++it;
    }
}

// a texture stage's image and D3DTSS address and filter state
struct Unit {
    std::shared_ptr<Image> img;
    uint32_t address_u = 1, address_v = 1, mag = 2, min = 2;
    Vec border;

    // D3DTADDRESS: 1 wrap, 2 mirror, 4 border, else clamp
    static int Address(int i, int n, uint32_t mode, bool& outside) {
        if (mode == 1) {
            i %= n;
            return i < 0 ? i + n : i;
        }
        if (mode == 2) {
            int period = 2 * n;
            i %= period;
            if (i < 0) i += period;
            return i < n ? i : period - 1 - i;
        }
        if (mode == 4 && (i < 0 || i >= n)) outside = true;
        return std::clamp(i, 0, n - 1);
    }
    Vec Texel(int face, int x, int y) const {
        bool outside = false;
        x = Address(x, int(img->w), address_u, outside);
        y = Address(y, int(img->h), address_v, outside);
        if (outside) return border;
        return img->texels[face][size_t(y) * img->w + x];
    }
    // point when both filters are, else bilinear at texel centres
    Vec Sample(int face, float u, float v) const {
        if (!img) return V(0, 0, 0, 1);
        float sx = u * img->w - 0.5f, sy = v * img->h - 0.5f;
        if (mag < 2 && min < 2) return Texel(face, int(std::floor(u * img->w)), int(std::floor(v * img->h)));
        int x = int(std::floor(sx)), y = int(std::floor(sy));
        float fx = sx - x, fy = sy - y;
        Vec t00 = Texel(face, x, y), t10 = Texel(face, x + 1, y), t01 = Texel(face, x, y + 1), t11 = Texel(face, x + 1, y + 1);
        Vec r;
        for (int i = 0; i < 4; i++) r[i] = (t00[i] * (1 - fx) + t10[i] * fx) * (1 - fy) + (t01[i] * (1 - fx) + t11[i] * fx) * fy;
        return r;
    }
    // a cube along a direction; a plain texture takes x y as its coordinates
    Vec SampleDir(float x, float y, float z) const {
        if (!img) return V(0, 0, 0, 1);
        if (img->faces < 6) return Sample(0, x, y);
        float ax = std::fabs(x), ay = std::fabs(y), az = std::fabs(z), major, s, t;
        int face;
        if (ax >= ay && ax >= az) {
            face = x >= 0 ? 0 : 1;
            major = ax;
            s = x >= 0 ? -z : z;
            t = -y;
        } else if (ay >= az) {
            face = y >= 0 ? 2 : 3;
            major = ay;
            s = x;
            t = y >= 0 ? z : -z;
        } else {
            face = z >= 0 ? 4 : 5;
            major = az;
            s = z >= 0 ? x : -x;
            t = -y;
        }
        if (major == 0) major = 1;
        Unit edge = *this;
        edge.address_u = edge.address_v = 3;
        return edge.Sample(face, (s / major + 1) * 0.5f, (t / major + 1) * 0.5f);
    }
};

// vertex programs

// the microcode fields, as shaders.cpp names them
struct Tok {
    const uint32_t* t;
    uint32_t f(int sub, int bit, int len) const { return (t[sub] >> bit) & ((1u << len) - 1); }
};

// magnitude clamped between 1 / 2^64 and 2^64, sign kept
float MagClamp(float v) {
    float m = std::clamp(std::fabs(v), 5.42101e-20f, 1.84467e+19f);
    return v < 0 ? -m : m;
}
float MacMul(float a, float b) { return (a == 0 || b == 0) ? 0 : a * b; }
Vec Splat(float v) { return V(v, v, v, v); }

// 1 mov, 2 mul, 3 add, 4 mad, 5 dp3, 6 dph, 7 dp4, 8 dst, 9 min, 10 max, 11 slt, 12 sge, 13 arl
Vec Mac(uint32_t op, const Vec& a, const Vec& b, const Vec& c) {
    Vec r;
    switch (op) {
        case 1:
        case 13: return a;
        case 2: for (int i = 0; i < 4; i++) r[i] = MacMul(a[i], b[i]); return r;
        case 3: for (int i = 0; i < 4; i++) r[i] = a[i] + c[i]; return r;
        case 4: for (int i = 0; i < 4; i++) r[i] = MacMul(a[i], b[i]) + c[i]; return r;
        case 5: return Splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        case 6: return Splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + b[3]);
        case 7: return Splat(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
        case 8: return V(1, a[1] * b[1], a[2], b[3]);
        case 9: for (int i = 0; i < 4; i++) r[i] = std::min(a[i], b[i]); return r;
        case 10: for (int i = 0; i < 4; i++) r[i] = std::max(a[i], b[i]); return r;
        case 11: for (int i = 0; i < 4; i++) r[i] = a[i] < b[i] ? 1.f : 0.f; return r;
        case 12: for (int i = 0; i < 4; i++) r[i] = a[i] >= b[i] ? 1.f : 0.f; return r;
        default: return r;
    }
}

// 1 mov, 2 rcp, 3 rcc, 4 rsq, 5 exp, 6 log, 7 lit
Vec Ilu(uint32_t op, const Vec& c) {
    float v = c[0];
    switch (op) {
        case 1: return c;
        case 2: return Splat(1.0f / v);
        case 3: return Splat(MagClamp(1.0f / v));
        case 4: return Splat(v == 0 ? INFINITY : 1.0f / std::sqrt(std::fabs(v)));
        case 5: {
            float whole = std::floor(v);
            return V(std::exp2(whole), v - whole, std::exp2(v), 1);
        }
        case 6: {
            float mag = std::fabs(v);
            if (mag == 0) return V(-INFINITY, 1, -INFINITY, 1);
            float ex = std::floor(std::log2(mag));
            return V(ex, mag / std::exp2(ex), std::log2(mag), 1);
        }
        case 7: {
            const float cap = 127.99609375f;
            float power = std::clamp(c[3], -cap, cap), diffuse = std::max(c[0], 0.f), spec = std::max(c[1], 0.f);
            return V(1, diffuse, diffuse > 0 ? std::exp2(power * std::log2(spec)) : 0, 1);
        }
        default: return Vec{};
    }
}

// a vertex program instruction with its fields taken apart once, not once per vertex
struct Operand {
    bool neg = false, rel = false;
    uint8_t sw[4] = {0, 1, 2, 3};
    uint8_t reg = 0, mux = 0, in = 0;
    uint8_t scalar = 0;  // scalar ops read x
    uint8_t cr = 0;
};
struct Instr {
    uint8_t mac = 0, ilu = 0, r = 0, omask = 0, out = 0, mask_ilu = 0, mask_mac = 0;
    bool use_ilu = false, out_en = false, last = false;
    Operand op[3];
};
std::vector<Instr> DecodeProgram(const std::vector<uint32_t>& prog) {
    std::vector<Instr> out;
    for (size_t n = 0; n + 4 <= prog.size(); n += 4) {
        Tok k{&prog[n]};
        Instr in;
        in.mac = uint8_t(k.f(1, 21, 4));
        in.ilu = uint8_t(k.f(1, 25, 3));
        in.r = uint8_t(k.f(3, 20, 4));
        in.omask = uint8_t(k.f(3, 12, 4));
        in.use_ilu = k.f(3, 2, 1) == 1;
        in.out_en = k.f(3, 11, 1) != 0;
        in.out = uint8_t(k.f(3, 3, 8) & 0xF);
        in.mask_ilu = uint8_t(k.f(3, 16, 4));
        in.mask_mac = in.ilu && in.r == 1 ? 0 : uint8_t(k.f(3, 24, 4));
        in.last = k.f(3, 0, 1) != 0;
        for (int which = 0; which < 3; which++) {
            Operand& o = in.op[which];
            uint32_t sw[4], reg, mux;
            if (which == 0) {
                o.neg = k.f(1, 8, 1);
                sw[0] = k.f(1, 6, 2), sw[1] = k.f(1, 4, 2), sw[2] = k.f(1, 2, 2), sw[3] = k.f(1, 0, 2);
                reg = k.f(2, 28, 4), mux = k.f(2, 26, 2);
            } else if (which == 1) {
                o.neg = k.f(2, 25, 1);
                sw[0] = k.f(2, 23, 2), sw[1] = k.f(2, 21, 2), sw[2] = k.f(2, 19, 2), sw[3] = k.f(2, 17, 2);
                reg = k.f(2, 13, 4), mux = k.f(2, 11, 2);
            } else {
                o.neg = k.f(2, 10, 1);
                sw[0] = k.f(2, 8, 2), sw[1] = k.f(2, 6, 2), sw[2] = k.f(2, 4, 2), sw[3] = k.f(2, 2, 2);
                reg = k.f(2, 0, 2) << 2 | k.f(3, 30, 2), mux = k.f(3, 28, 2);
                uint32_t ilu = k.f(1, 25, 3);
                if (ilu >= 2 && ilu <= 6) sw[1] = sw[2] = sw[3] = sw[0];
            }
            for (int i = 0; i < 4; i++) o.sw[i] = uint8_t(sw[i]);
            o.reg = uint8_t(reg);
            o.mux = uint8_t(mux);
            o.in = uint8_t(k.f(1, 9, 4));
            o.cr = uint8_t(k.f(1, 13, 8));
            o.rel = k.f(3, 1, 1) != 0;
        }
        out.push_back(in);
        if (in.last) break;
    }
    return out;
}

struct VertexProgram {
    const State& s;
    Vec in[16];
    Vec temp[12], out[16];  // out 0 oPos, 3 oD0, 4 oD1, 5 oFog, 6 oPts, 7 oB0, 8 oB1, 9 to 12 oT0 to oT3
    int a0 = 0;

    // temp 12 is oPos
    Vec& Reg(uint32_t r) { return r == 12 ? out[0] : temp[r]; }

    Vec Operand(const struct Operand& o) {
        Vec base;
        if (o.mux == 1) {
            base = o.reg == 12 ? out[0] : temp[o.reg > 12 ? 0 : o.reg];
        } else if (o.mux == 2) {
            base = in[o.in];
        } else {
            int cr = int(o.cr);
            if (o.rel) cr = std::clamp(cr + a0, 0, 191);
            const auto& c = s.vsc[size_t(cr)];
            base = V(c[0], c[1], c[2], c[3]);
        }
        Vec r;
        for (int i = 0; i < 4; i++) r[i] = o.neg ? -base[o.sw[i]] : base[o.sw[i]];
        return r;
    }

    static void Masked(Vec& dst, const Vec& src, uint32_t xyzw) {
        for (int i = 0; i < 4; i++)
            if (xyzw >> (3 - i) & 1) dst[i] = src[i];
    }

    void Run(const std::vector<Instr>& prog) {
        out[3] = out[7] = out[9] = out[10] = out[11] = out[12] = V(0, 0, 0, 1);
        out[6] = V(1, 1, 1, 1);
        for (const Instr& k : prog) {
            uint32_t mac = k.mac, ilu = k.ilu;
            if (mac || ilu) {
                Vec a = Operand(k.op[0]), b = Operand(k.op[1]), c = Operand(k.op[2]);
                Vec mr = mac ? Mac(mac, a, b, c) : Vec{}, ir = ilu ? Ilu(ilu, c) : Vec{};
                uint32_t r = k.r, omask = k.omask;
                bool use_ilu = k.use_ilu;
                // the output register is written before the temporaries
                if (omask && k.out_en && ((use_ilu && ilu) || (!use_ilu && mac && mac != 13))) {
                    const Vec& src = use_ilu ? ir : mr;
                    if (k.out == 5) {
                        // oFog takes one lane per mask bit, from x (as gpu/shaders.cpp writes it): a program writing
                        // oFog through its w lane left x at 0, which fogged every pixel to the fog colour (gray screen)
                        int lanes = __builtin_popcount(omask);
                        for (int i = 0; i < lanes; i++) out[5][i] = src[i];
                    } else {
                        Masked(out[k.out], src, omask);
                    }
                }
                if (ilu) {
                    uint32_t dst = mac ? 1 : r;
                    if (dst <= 12) Masked(Reg(dst), ir, k.mask_ilu);
                }
                if (mac == 13) {
                    a0 = int(std::floor(mr[0] + 0.001f));
                } else if (mac) {
                    if (r <= 12) Masked(Reg(r), mr, k.mask_mac);
                }
            }
        }
    }
};

float FogAmount(const State& s, float d) {
    if (!s.rs[kRsFogEnable]) return 1;
    float start = AsFloat(s.rs[kRsFogStart]), end = AsFloat(s.rs[kRsFogEnd]), density = AsFloat(s.rs[kRsFogDensity]);
    switch (s.rs[kRsFogTableMode]) {
        case 1: return std::exp(-d * density);
        case 2: return std::exp(-(d * density) * (d * density));
        case 3: return end == start ? 1 : (end - d) / (end - start);
        default: return 1;
    }
}

// per fragment shading

enum Attr { kD0, kD1, kB0, kB1, kFog, kT0, kAttrs = kT0 + 4 };

struct Fragment {
    Vec v0, v1, fog;  // fog: colour, amount in w
    Vec tc[4];
};

// what every fragment of a draw shares
struct Material {
    PixelKey key;
    TexInfo info[4];
    Unit unit[4];
    Vec pc[18];  // combiner c0 c1 per stage, then the final combiner's
    Vec tfactor;
    const State* s = nullptr;
};

// unorm to byte b, less lo under 128 or hi from 128, over d
float SByte(float u, float lo, float hi, float d) {
    float b = u * 255;
    return (b - (b >= 128 ? hi : lo)) / d;
}
// a dot stage's operand: the source stage's rgb, raw or read back as signed bytes
Vec DotInput(uint32_t map, const Vec& t) {
    float lo, hi, d;
    switch (map) {
        case 1: lo = 128, hi = 128, d = 127; break;
        case 2: lo = -0.5f, hi = 255.5f, d = 127.5f; break;
        case 3: lo = 0, hi = 256, d = 127; break;
        default: return t;
    }
    return V(SByte(t[0], lo, hi, d), SByte(t[1], lo, hi, d), SByte(t[2], lo, hi, d), 0);
}
float Dot3(const Vec& a, const Vec& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

struct Combiners {
    const Material& m;
    const Fragment& f;
    Vec t[4], v0, v1, r0, r1;
    Vec e, fprod;  // final combiner E and F
    int stage = 0;
    bool clamp_sum = false, inv_v1 = false, inv_r0 = false;

    // a texture stage with rect coordinates takes texels
    void Norm(int i, float& u, float& v) const {
        if (m.info[i].rect && m.unit[i].img) {
            u /= m.unit[i].img->w;
            v /= m.unit[i].img->h;
        }
    }
    // the depth texel against ref by D3DRS_SHADOWFUNC, point sampled
    Vec Shadow(int i, float u, float v, float ref, bool compare) const {
        const Unit& u0 = m.unit[i];
        float top = u0.img && u0.img->depth16 ? 65535.f : 16777215.f;
        float texel = u0.Sample(0, u, v)[0] * top, z = compare ? std::clamp(ref, 0.f, top) : 0;
        bool pass;
        switch (m.s->rs[kRsShadowFunc]) {  // GL compare enums, as in Passes
            case 0x200: pass = false; break;
            case 0x201: pass = texel < z; break;
            case 0x202: pass = texel == z; break;
            case 0x203: pass = texel <= z; break;
            case 0x204: pass = texel > z; break;
            case 0x205: pass = texel != z; break;
            case 0x206: pass = texel >= z; break;
            default: pass = true; break;
        }
        return Splat(pass ? 1.f : 0.f);
    }

    // texture stage i per its mode; false kills the fragment
    bool Fetch(int i, float dp[4]) {
        const PixelKey& k = m.key;
        const Unit& u0 = m.unit[i];
        const TexInfo& info = m.info[i];
        uint32_t mode = (k.tex_modes >> (i * 5)) & 0x1F;
        uint32_t src = i == 1 ? 0 : i == 2 ? (k.input_texture >> 16) & 0xF : i == 3 ? (k.input_texture >> 20) & 0xF : 0;
        uint32_t map = i == 0 ? 0 : (k.dot_mapping >> ((i - 1) * 4)) & 0xF;
        const Vec& q = f.tc[i];
        auto dot = [&] { dp[i] = Dot3(q, DotInput(map, t[src])); };
        switch (mode) {
            case 0: t[i] = V(0, 0, 0, 1); break;
            case 1:
            case 2: {
                float w = q[3] == 0 ? 1 : q[3], u = q[0] / w, v = q[1] / w;
                Norm(i, u, v);
                if (info.shadow) t[i] = mode == 1 ? Shadow(i, u, v, 0, false) : Shadow(i, u, v, q[2] / w, true);
                else if (mode == 1 && info.kind == kTexCube) t[i] = u0.SampleDir(u, v, 1);
                else t[i] = u0.Sample(0, u, v);
                break;
            }
            case 3: t[i] = info.kind == kTexCube ? u0.SampleDir(q[0], q[1], q[2]) : u0.Sample(0, q[0], q[1]); break;
            case 4: t[i] = q; break;
            case 5:  // clip plane: a set compare_mode bit kills at >= 0, a clear one below 0
                for (int j = 0; j < 4; j++) {
                    bool ge = (k.compare_mode >> (i * 4 + j)) & 1;
                    if (ge ? q[j] >= 0 : q[j] < 0) return false;
                }
                t[i] = Vec{};
                break;
            case 6:
            case 7: {  // bump env map: signed du dv through the 2x2 matrix, 7 scales by luminance
                const auto& ts = m.s->tss[size_t(i)];
                Vec in = t[src];
                float du = SByte(in[2], 0, 256, 127), dv = SByte(in[1], 0, 256, 127);
                float u = q[0] + (AsFloat(ts[kTssBumpEnvMat00]) * du + AsFloat(ts[kTssBumpEnvMat10]) * dv);
                float v = q[1] + (AsFloat(ts[kTssBumpEnvMat01]) * du + AsFloat(ts[kTssBumpEnvMat11]) * dv);
                Norm(i, u, v);
                t[i] = u0.Sample(0, u, v);
                if (mode == 7) {
                    float lum = in[0] * AsFloat(ts[kTssBumpEnvLScale]) + AsFloat(ts[kTssBumpEnvLOffset]);
                    for (int j = 0; j < 4; j++) t[i][j] *= lum;
                }
                break;
            }
            case 9: {
                dot();
                float u = dp[i - 1], v = dp[i];
                Norm(i, u, v);
                t[i] = u0.Sample(0, u, v);
                break;
            }
            case 0xA:
            case 0x11: dot(); t[i] = Vec{}; break;
            case 0xB: {  // reflect diffuse at stage 2: stage 3's product early for the normal
                dot();
                float next = Dot3(f.tc[3], DotInput((k.dot_mapping >> 8) & 0xF, t[(k.input_texture >> 20) & 0xF]));
                t[i] = u0.SampleDir(dp[i - 1], dp[i], next);
                break;
            }
            case 0xC: {  // reflect specular at stage 3: eye from the three w
                dot();
                Vec n = V(dp[i - 2], dp[i - 1], dp[i], 0), eye = V(f.tc[i - 2][3], f.tc[i - 1][3], f.tc[i][3], 0);
                float nn = Dot3(n, n), ne = Dot3(n, eye);
                Vec r;
                for (int j = 0; j < 3; j++) r[j] = 2 * n[j] * ne / nn - eye[j];
                t[i] = u0.SampleDir(r[0], r[1], r[2]);
                break;
            }
            case 0xD: dot(); t[i] = u0.Sample(0, dp[i - 2], dp[i - 1]); break;
            case 0xE: dot(); t[i] = u0.SampleDir(dp[i - 2], dp[i - 1], dp[i]); break;
            case 0xF: t[i] = u0.Sample(0, t[src][3], t[src][0]); break;
            case 0x10: t[i] = u0.Sample(0, t[src][1], t[src][2]); break;
            default: t[i] = Vec{}; break;
        }
        return !(mode && mode != 4 && mode != 5 && (k.alphakill >> i & 1) && t[i][3] == 0);
    }

    Vec Var(uint32_t reg) const {
        uint32_t flags = m.key.count >> 8;
        switch (reg) {
            case 1: return m.pc[stage == 8 ? 16 : ((flags & 0x10) ? stage * 2 : 0)];
            case 2: return m.pc[stage == 8 ? 17 : ((flags & 0x100) ? stage * 2 + 1 : 1)];
            case 3: return f.fog;
            case 4: return v0;
            case 5: return v1;
            case 8: return t[0];
            case 9: return t[1];
            case 10: return t[2];
            case 11: return t[3];
            case 12: return r0;
            case 13: return r1;
            case 14: {  // v1 + r0, each maybe inverted, maybe clamped
                Vec r;
                for (int i = 0; i < 3; i++) {
                    float a = inv_v1 ? 1 - v1[i] : v1[i], b = inv_r0 ? 1 - r0[i] : r0[i];
                    r[i] = a + b;
                    if (clamp_sum) r[i] = Sat(r[i]);
                }
                return r;
            }
            case 15: return V(e[0] * fprod[0], e[1] * fprod[1], e[2] * fprod[2], 0);
            default: return Vec{};
        }
    }

    // an input byte: register, channel (alpha or the colour/blue), input mapping
    void Input(uint32_t v, bool alpha, float r[3]) const {
        Vec x = Var(v & 0xF);
        bool a = v & 0x10;
        for (int i = 0; i < 3; i++) {
            float c = a ? x[3] : alpha ? x[2] : x[i];
            // unsigned, unsigned invert, expand, expand negate, half bias, half bias negate, signed, signed negate
            switch (v & 0xE0) {
                case 0x00: c = std::max(c, 0.f); break;
                case 0x20: c = 1 - Sat(c); break;
                case 0x40: c = 2 * std::max(c, 0.f) - 1; break;
                case 0x60: c = -2 * std::max(c, 0.f) + 1; break;
                case 0x80: c = std::max(c, 0.f) - 0.5f; break;
                case 0xA0: c = -std::max(c, 0.f) + 0.5f; break;
                case 0xC0: break;
                default: c = -c; break;
            }
            r[i] = c;
        }
    }

    // bias, shift left 1, bias then shift left 1, shift left 2, shift right 1
    static float OutMap(float v, uint32_t map) {
        switch (map) {
            case 0x08: return v - 0.5f;
            case 0x10: return v * 2;
            case 0x18: return (v - 0.5f) * 2;
            case 0x20: return v * 4;
            case 0x30: return v / 2;
            default: return v;
        }
    }

    Vec* Dest(uint32_t reg) {
        switch (reg) {
            case 4: return &v0;
            case 5: return &v1;
            case 8: return &t[0];
            case 9: return &t[1];
            case 10: return &t[2];
            case 11: return &t[3];
            case 12: return &r0;
            case 13: return &r1;
            default: return nullptr;
        }
    }

    struct Write {
        Vec* dst;
        int comp;
        float v;
    };
    // a stage makes at most two halves of three registers with four components each: no heap per fragment
    struct WriteList {
        Write a[32];
        int n = 0;
        void clear() { n = 0; }
        void push_back(const Write& x) { a[n++] = x; }
        const Write* begin() const { return a; }
        const Write* end() const { return a + n; }
    };
    // one half of a general combiner stage; rgb and alpha both read before either writes
    void Half(uint32_t inputs, uint32_t output, bool alpha, WriteList& w) {
        float A[3], B[3], C[3], D[3];
        Input(inputs >> 24, alpha, A);
        Input((inputs >> 16) & 0xFF, alpha, B);
        Input((inputs >> 8) & 0xFF, alpha, C);
        Input(inputs & 0xFF, alpha, D);
        uint32_t fl = output >> 12, map = fl & 0x38;
        float ab_dot = A[0] * B[0] + A[1] * B[1] + A[2] * B[2], cd_dot = C[0] * D[0] + C[1] * D[1] + C[2] * D[2];
        bool pick_cd = (m.key.count >> 8 & 1) ? r0[3] >= 0.5f : (uint32_t(r0[3] * 255) & 1) == 1;
        float ab[3], cd[3], ms[3];
        for (int i = 0; i < (alpha ? 1 : 3); i++) {
            float x = (fl & 2) && !alpha ? ab_dot : A[i] * B[i];
            float y = (fl & 1) && !alpha ? cd_dot : C[i] * D[i];
            float sum = (fl & 4) ? (pick_cd ? y : x) : x + y;
            ab[i] = std::clamp(OutMap(x, map), -1.f, 1.f);
            cd[i] = std::clamp(OutMap(y, map), -1.f, 1.f);
            ms[i] = std::clamp(OutMap(sum, map), -1.f, 1.f);
        }
        auto put = [&](uint32_t reg, const float* v, bool blue_to_alpha) {
            Vec* d = Dest(reg);
            if (!d) return;
            if (alpha) {
                w.push_back({d, 3, v[0]});
                return;
            }
            for (int i = 0; i < 3; i++) w.push_back({d, i, v[i]});
            if (blue_to_alpha) w.push_back({d, 3, v[2]});
        };
        put((output >> 4) & 0xF, ab, fl & 0x80);
        put(output & 0xF, cd, fl & 0x40);
        put((output >> 8) & 0xF, ms, false);
    }

    bool Shade(Vec& out) {
        float dp[4] = {};
        for (int i = 0; i < 4; i++)
            if (!Fetch(i, dp)) return false;
        const PixelKey& k = m.key;
        v0 = f.v0;
        v1 = f.v1;
        r0 = V(0, 0, 0, (k.tex_modes & 0x1F) ? t[0][3] : 1);
        r1 = Vec{};
        uint32_t n = std::min<uint32_t>(k.count & 0xFF, 8);
        WriteList w;
        for (uint32_t st = 0; st < n; st++) {
            stage = int(st);
            w.clear();
            Half(k.rgb_in[st], k.rgb_out[st], false, w);
            Half(k.a_in[st], k.a_out[st], true, w);
            for (const Write& x : w) (*x.dst)[x.comp] = x.v;
        }
        if (!k.final_abcd && !k.final_efg) {
            for (int i = 0; i < 3; i++) out[i] = f.fog[i] * (1 - f.fog[3]) + r0[i] * f.fog[3];
            out[3] = r0[3];
            return true;
        }
        // final combiner: D + lerp(C, B, A) in rgb, G in alpha
        stage = 8;
        uint32_t fl = k.final_efg & 0xFF;
        clamp_sum = fl & 0x80;
        inv_v1 = fl & 0x40;
        inv_r0 = fl & 0x20;
        float E[3], F[3], A[3], B[3], C[3], D[3], G[3];
        Input(k.final_efg >> 24, false, E);
        Input((k.final_efg >> 16) & 0xFF, false, F);
        e = V(E[0], E[1], E[2], 0);
        fprod = V(F[0], F[1], F[2], 0);
        Input(k.final_abcd >> 24, false, A);
        Input((k.final_abcd >> 16) & 0xFF, false, B);
        Input((k.final_abcd >> 8) & 0xFF, false, C);
        Input(k.final_abcd & 0xFF, false, D);
        Input((k.final_efg >> 8) & 0xFF, true, G);
        for (int i = 0; i < 3; i++) out[i] = D[i] + C[i] * (1 - A[i]) + B[i] * A[i];
        out[3] = G[0];
        return true;
    }
};

// D3D fixed function texture stages
bool ShadeStages(const Material& m, const Fragment& f, Vec& out) {
    Vec cur = f.v0, tmp;
    for (int i = 0; i < 4; i++) {
        const uint32_t* st = m.key.ff[i];  // colorop, arg1, arg2, arg0, alphaop, arg1, arg2, arg0, result
        if (st[0] <= 1) break;
        const Unit& u0 = m.unit[i];
        Vec tex;
        if (m.info[i].kind == kTexCube) {
            tex = u0.SampleDir(f.tc[i][0], f.tc[i][1], f.tc[i][2]);
        } else if (m.info[i].kind) {
            float u = f.tc[i][0], v = f.tc[i][1];
            // the console's stages divide by q under a vertex program or D3DTTFF_PROJECTED
            if ((m.key.proj >> i & 1) && f.tc[i][3] != 0) {
                u /= f.tc[i][3];
                v /= f.tc[i][3];
            }
            if (m.info[i].rect && u0.img) {
                u /= u0.img->w;
                v /= u0.img->h;
            }
            tex = u0.Sample(0, u, v);
        } else {
            tex = V(1, 1, 1, 1);
        }
        if ((m.key.alphakill >> i & 1) && tex[3] == 0) return false;
        // D3DTA: source, then alpha replicate, then complement
        auto arg = [&](uint32_t a) {
            Vec x;
            switch (a & 0xF) {
                case 0: x = f.v0; break;
                case 2: x = tex; break;
                case 3: x = m.tfactor; break;
                case 4: x = f.v1; break;
                case 5: x = tmp; break;
                default: x = cur; break;
            }
            if (a & 0x20) x = Splat(x[3]);
            if (a & 0x10)
                for (int j = 0; j < 4; j++) x[j] = 1 - x[j];
            return x;
        };
        // D3DTOP on channel j
        auto op = [&](uint32_t o, const Vec& a1, const Vec& a2, const Vec& a0, int j, bool alpha) -> float {
            float x = a1[j], y = a2[j], z = a0[j];
            switch (o) {
                case 2: return x;
                case 3: return y;
                case 4: return x * y;
                case 5: return x * y * 2;
                case 6: return x * y * 4;
                case 7: return x + y;
                case 8: return x + y - 0.5f;
                case 9: return (x + y - 0.5f) * 2;
                case 10: return x - y;
                case 11: return x + y * (1 - x);
                case 12: return y + (x - y) * f.v0[3];
                case 13: return y + (x - y) * cur[3];
                case 14: return y + (x - y) * tex[3];
                case 15: return y + (x - y) * m.tfactor[3];
                case 16: return x + y * (1 - tex[3]);
                case 18: return alpha ? x : x + y * a1[3];
                case 19: return alpha ? x : x * y + a1[3];
                case 20: return alpha ? x : x + y * (1 - a1[3]);
                case 21: return alpha ? x : (1 - x) * y + a1[3];
                case 22: {
                    float d = 0;
                    for (int q = 0; q < 3; q++) d += (a1[q] - 0.5f) * 2 * (a2[q] - 0.5f) * 2;
                    return Sat(d);
                }
                case 23: return z + x * y;
                case 24: return y + (x - y) * z;
                default: return cur[alpha ? 3 : j];
            }
        };
        Vec r;
        Vec c1 = arg(st[1]), c2 = arg(st[2]), c0 = arg(st[3]);
        for (int j = 0; j < 3; j++) r[j] = Sat(op(st[0], c1, c2, c0, j, false));
        if (st[4] <= 1) {
            r[3] = cur[3];
        } else {
            Vec a1 = arg(st[5]), a2 = arg(st[6]), a0 = arg(st[7]);
            r[3] = Sat(op(st[4], a1, a2, a0, 3, true));
        }
        (st[8] == 5 ? tmp : cur) = r;
    }
    for (int j = 0; j < 3; j++) cur[j] = Sat(cur[j] + f.v1[j]);
    for (int j = 0; j < 3; j++) out[j] = f.fog[j] * (1 - f.fog[3]) + cur[j] * f.fog[3];
    out[3] = cur[3];
    return true;
}

// output merger, GL enums as the pushbuffer holds them

bool Passes(uint32_t func, uint32_t a, uint32_t b) {
    switch (func) {
        case 0x200: return false;
        case 0x201: return a < b;
        case 0x202: return a == b;
        case 0x203: return a <= b;
        case 0x204: return a > b;
        case 0x205: return a != b;
        case 0x206: return a >= b;
        default: return true;
    }
}
// GL ops: zero, replace, saturating incr and decr, invert, wrapping incr and decr, else keep
uint32_t StencilResult(uint32_t op, uint32_t v, uint32_t ref) {
    switch (op) {
        case 0: return 0;
        case 0x1E01: return ref;
        case 0x1E02: return std::min(v + 1, 255u);
        case 0x1E03: return v ? v - 1 : 0;
        case 0x150A: return ~v & 255;
        case 0x8507: return (v + 1) & 255;
        case 0x8508: return (v - 1) & 255;
        default: return v;
    }
}
// GL blend factors, 0x8001 to 0x8004 take the constant colour
float BlendFactor(uint32_t f, const Vec& src, const Vec& dst, const Vec& k, int ch) {
    switch (f) {
        case 0: return 0;
        case 1: return 1;
        case 0x300: return src[ch];
        case 0x301: return 1 - src[ch];
        case 0x302: return src[3];
        case 0x303: return 1 - src[3];
        case 0x304: return dst[3];
        case 0x305: return 1 - dst[3];
        case 0x306: return dst[ch];
        case 0x307: return 1 - dst[ch];
        case 0x308: return ch == 3 ? 1 : std::min(src[3], 1 - dst[3]);
        case 0x8001: return k[ch];
        case 0x8002: return 1 - k[ch];
        case 0x8003: return k[3];
        case 0x8004: return 1 - k[3];
        default: return 1;
    }
}

// rows of one triangle on a few worker threads: the pixels of a draw do not overlap, so only the rows are split
struct RowPool {
    std::vector<std::thread> threads;
    std::mutex m;
    std::condition_variable cv, done_cv;
    std::function<void(int)> fn;
    std::atomic<int> next{0};
    std::atomic<int> end{0};
    int generation = 0, busy = 0;
    bool stop = false;
    RowPool() {
        unsigned n = std::thread::hardware_concurrency();
        n = n > 2 ? std::min(n, 6u) - 1 : 1;
        for (unsigned i = 0; i < n; i++) threads.emplace_back([this] { Work(); });
    }
    void Drain(const std::function<void(int)>& f) {
        for (;;) {
            int y = next.fetch_add(1);
            if (y >= end) break;
            f(y);
        }
    }
    void Work() {
        int seen = 0;
        std::unique_lock<std::mutex> lock(m);
        for (;;) {
            cv.wait(lock, [&] { return stop || generation != seen; });
            if (stop) return;
            seen = generation;
            busy++;
            std::function<void(int)> f = fn;
            lock.unlock();
            Drain(f);
            lock.lock();
            if (--busy == 0) done_cv.notify_all();
        }
    }
    void Run(int y0, int y1, const std::function<void(int)>& f) {
        {
            std::lock_guard<std::mutex> lock(m);
            fn = f;
            next = y0;
            end = y1 + 1;
            generation++;
        }
        cv.notify_all();
        Drain(f);
        std::unique_lock<std::mutex> lock(m);
        done_cv.wait(lock, [&] { return busy == 0; });
    }
};

template <class F>
void ParallelRows(int y0, int y1, bool parallel, F& row) {
    if (!parallel) {
        for (int y = y0; y <= y1; y++) row(y);
        return;
    }
    static RowPool* pool = new RowPool;
    pool->Run(y0, y1, [&](int y) { row(y); });
}

uint32_t g_draws = 0;  // this frame's

// triangles

struct Shaded {
    Vec pos;  // screen x y, depth units z, w
    Vec attr[kAttrs];
    float rawfog = 0;
};

// clip space vertex, attributes not yet divided
struct Clip {
    float h[4];
    Vec attr[kAttrs];
};

// screen vertex with attributes over w
struct Screen {
    float x, y, z, iw;
    Vec attr[kAttrs];
};

}  // namespace

void BeginFrame() {
    g_frame++;
    g_draws = 0;
}

void Clear(const State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) {
    Plane rt = OpenPlane(s.rt), zs = OpenPlane(s.zs);
    // the given rects, else the viewport
    auto each = [&](const Plane& p, auto&& fn) {
        std::vector<std::array<int, 4>> area;
        if (count && rects) {
            for (uint32_t i = 0; i < count; i++) {
                int32_t r[4];
                std::memcpy(r, rects + i * 16, 16);
                area.push_back({r[0], r[1], r[2], r[3]});
            }
        } else {
            int x = int(s.viewport.x), y = int(s.viewport.y);
            area.push_back({x, y, x + int(s.viewport.w), y + int(s.viewport.h)});
        }
        for (auto& r : area)
            for (int y = std::max(r[1], 0); y < std::min<int>(r[3], int(p.h)); y++)
                for (int x = std::max(r[0], 0); x < std::min<int>(r[2], int(p.w)); x++) fn(uint32_t(x), uint32_t(y));
    };
    // D3DCLEAR: 1 z, 2 stencil, 0x10 to 0x80 red, green, blue, alpha
    if ((flags & 0xF0) && rt.ok) {
        uint32_t mask = (flags & 0x10 ? 4u : 0u) | (flags & 0x20 ? 2u : 0u) | (flags & 0x40 ? 1u : 0u) | (flags & 0x80 ? 8u : 0u);
        Vec c = Unpack(color);
        each(rt, [&](uint32_t x, uint32_t y) { rt.Store(x, y, c, mask); });
        Written(rt.data, rt.End());
    }
    if ((flags & 3) && zs.ok) {
        float top = zs.Depth16() ? 65535.f : 16777215.f;
        each(zs, [&](uint32_t x, uint32_t y) {
            uint32_t oz, os;
            zs.LoadDepth(x, y, oz, os);
            zs.StoreDepth(x, y, (flags & 1) ? uint32_t(z * top) : oz, (flags & 2) ? stencil : os);
        });
        Written(zs.data, zs.End());
    }
}

void Copy(const Surface& src, const Surface& dst, uint32_t count, uint32_t rects, uint32_t points) {
    Plane a = OpenPlane(src), b = OpenPlane(dst);
    if (!a.ok || !b.ok) return;
    uint32_t n = count && rects ? count : 1;
    for (uint32_t i = 0; i < n; i++) {
        int x0 = 0, y0 = 0, x1 = int(a.w), y1 = int(a.h), dx = 0, dy = 0;
        if (count && rects) {
            x0 = int(kernel::R32(rects + i * 16));
            y0 = int(kernel::R32(rects + i * 16 + 4));
            x1 = int(kernel::R32(rects + i * 16 + 8));
            y1 = int(kernel::R32(rects + i * 16 + 12));
            dx = x0;
            dy = y0;
        }
        if (points) {
            dx = int(kernel::R32(points + i * 8));
            dy = int(kernel::R32(points + i * 8 + 4));
        }
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < x1; x++) {
                int tx = dx + x - x0, ty = dy + y - y0;
                if (x < 0 || y < 0 || x >= int(a.w) || y >= int(a.h) || tx < 0 || ty < 0 || tx >= int(b.w) || ty >= int(b.h)) continue;
                if (a.bytes == b.bytes) std::memcpy(b.At(uint32_t(tx), uint32_t(ty)), a.At(uint32_t(x), uint32_t(y)), a.bytes);
                else b.Store(uint32_t(tx), uint32_t(ty), a.Load(uint32_t(x), uint32_t(y)), 15);
            }
    }
    Written(b.data, b.End());
}

void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) {
    uint32_t draw = g_draws++;
    g_dbg = DrawDbg{};
    // kept between draws: its vectors keep their capacity instead of being allocated for every draw
    static thread_local Assembled as;
    // a draw whose vertex program runs on the GPU takes the guest's vertices as they are, no float copy of each
    g_assemble_raw = g_packet && g_allow_raw && (s.vs_handle & 1) && s.vs && !s.vs->program.empty() &&
                     s.simple[kMShadeMode >> 2] != 0x1D00 && !s.rs[kRsTwoSidedLighting];
    const bool assembled = Assemble(s, prim, start, count, indices, as);
    g_assemble_raw = false;
    if (!assembled) return;
    if (as.topology != Topology::Triangles) {
        return;
    }
    Plane rt = OpenPlane(s.rt);
    Plane zs = s.rs[kRsZEnable] || s.rs[kRsStencilEnable] ? OpenPlane(s.zs) : Plane{};
    if (!rt.ok && !zs.ok) return;
    if (rt.ok && zs.ok && (rt.w != zs.w || rt.h != zs.h)) zs.ok = false;
    auto simple = [&](uint32_t method) { return s.simple[method >> 2]; };

    Material mat;
    mat.s = &s;
    uint32_t target_stage[4] = {}, stage_format[4] = {}, stage_w[4] = {}, stage_h[4] = {}, stage_stride[4] = {};
    for (int i = 0; i < 4; i++) {
        TextureDesc d;
        if (!s.textures[size_t(i)] || !Describe(s.textures[size_t(i)], d)) continue;
        TexInfo& ti = mat.info[i];
        ti.kind = d.cube ? kTexCube : d.depth > 1 ? kTex3D : kTex2D;
        ti.rect = d.linear;
        if (Format(d.format).depth) ti.shadow = d.format == 0x2C || d.format == 0x30 ? 2 : 1;
        Unit& u = mat.unit[i];
        const auto& ts = s.tss[size_t(i)];
        stage_format[i] = d.format;
        stage_w[i] = d.width;
        stage_h[i] = d.height;
        // a texture over a GPU render target is that target: the GPU samples it, nothing is decoded
        if (g_packet && g_target_test && ti.kind == kTex2D && !ti.shadow && g_target_test(d.data, d.format, d.width, d.height)) {
            target_stage[i] = d.data;
        } else if (g_packet && g_target_test && ti.kind == kTexCube) {
            // a cube the title renders a face at a time (NFS Most Wanted's car reflections): its six faces are targets
            bool all = true;
            for (uint32_t f = 0; f < 6 && all; f++) all = g_target_test(d.data + f * FaceSize(d), d.format, d.width, d.height);
            if (all) {
                target_stage[i] = d.data;
                stage_stride[i] = FaceSize(d);
            }
        }
        if (!target_stage[i]) u.img = FindImage(s.textures[size_t(i)], s.palettes[size_t(i)], s.simple[(0x1B24u + uint32_t(i) * 0x40u) >> 2]);
        u.address_u = ts[kTssAddressU];
        u.address_v = ts[kTssAddressV];
        u.mag = ti.shadow ? 1 : ts[kTssMagFilter];
        u.min = ti.shadow ? 1 : ts[kTssMinFilter];
        u.border = Unpack(ts[kTssBorderColor]);
    }
    mat.key = MakePixelKey(s, mat.info);
    if (mat.key.programmable) {
        // the combiner registers are what the hardware runs: the title's inlined D3D code writes them after SetPixelShader,
        // and the shader's own definition may only hold part of it (the library's fixed function setup)
        PixelKey& k = mat.key;
        for (int i = 0; i < 8; i++) {
            k.a_in[i] = simple(kMPsAlphaInputs + i * 4);
            k.a_out[i] = simple(kMPsAlphaOutputs + i * 4);
            k.rgb_in[i] = simple(kMPsRgbInputs + i * 4);
            k.rgb_out[i] = simple(kMPsRgbOutputs + i * 4);
        }
        k.final_abcd = simple(kMPsFinalAbcd);
        k.final_efg = simple(kMPsFinalEfgh);
        k.count = simple(kMPsCombinerCount);
        k.compare_mode = simple(kMPsCompareMode);
        k.dot_mapping = simple(kMPsDotMapping);
        k.input_texture = simple(kMPsInputTexture);
        // the stage programs too: the hardware register, as the shader's definition may not hold them
        uint32_t hw_modes = simple(0x1E70);
        if (hw_modes) {
            k.tex_modes = 0;
            for (int i = 0; i < 4; i++) {
                uint32_t m = (hw_modes >> (i * 5)) & 0x1F;
                const TexInfo& t = mat.info[i];
                if (!t.kind && m != 4 && m != 5 && m != 0xA && m != 0x11) m = 0;
                else if (m >= 1 && m <= 3) m = t.kind == kTexCube ? 3 : (t.kind == kTex3D || t.shadow) ? 2 : 1;
                else if (m == 0xD || m == 0xE) m = t.kind == kTexCube ? 0xE : 0xD;
                k.tex_modes |= m << (i * 5);
            }
        }
    }
    for (int i = 0; i < 8; i++) {
        mat.pc[i * 2] = Unpack(simple(kMPsConstant0 + i * 4));
        mat.pc[i * 2 + 1] = Unpack(simple(kMPsConstant1 + i * 4));
    }
    mat.pc[16] = Unpack(simple(kMPsFinalConstant0));
    mat.pc[17] = Unpack(simple(kMPsFinalConstant1));
    mat.tfactor = Unpack(s.rs[kRsTextureFactor]);
    if (g_packet) {
        DrawPacket& pk = *g_packet;
        pk.key = mat.key;
        for (int i = 0; i < 4; i++) {
            pk.info[i] = mat.info[i];
            pk.stage[i] = GlStage{};
            pk.hold[i].reset();
            const Unit& u = mat.unit[i];
            pk.stage[i].format = stage_format[i];
            if (target_stage[i]) {
                pk.stage[i].target = target_stage[i];
                pk.stage[i].face_stride = stage_stride[i];
                pk.stage[i].w = stage_w[i];
                pk.stage[i].h = stage_h[i];
            } else if (u.img && !u.img->rgba8[0].empty()) {
                pk.stage[i].id = u.img->id;
                pk.stage[i].slot = u.img->key;
                pk.stage[i].w = u.img->w;
                pk.stage[i].h = u.img->h;
                pk.stage[i].rgba = u.img->rgba8[0].data();
                for (uint32_t f = 0; f < u.img->faces && f < 6; f++) pk.stage[i].faces[f] = u.img->rgba8[f].data();
                // a cube needs all six faces, else it is read as a plain texture
                if (pk.info[i].kind == kTexCube && u.img->faces < 6) pk.info[i].kind = kTex2D;
                pk.hold[i] = u.img;
            } else {
                pk.info[i].kind = kTexNone;
            }
            pk.stage[i].address_u = u.address_u;
            pk.stage[i].address_v = u.address_v;
            pk.stage[i].mag = u.mag;
            pk.stage[i].min = u.min;
            pk.stage[i].mip = s.tss[size_t(i)][kTssMipFilter];
            for (int j = 0; j < 4; j++) pk.stage[i].border[j] = u.border[j];
        }
        for (int i = 0; i < 18; i++)
            for (int j = 0; j < 4; j++) pk.pc[i][j] = mat.pc[i][j];
        for (int j = 0; j < 4; j++) pk.tfactor[j] = mat.tfactor[j];
        Vec fc = Unpack(s.rs[kRsFogColor]);
        for (int j = 0; j < 4; j++) pk.fog_color[j] = fc[j];
    }

    // vertices
    bool program = (s.vs_handle & 1) && s.vs && !s.vs->program.empty();
    // the GPU runs the vertex program: the inputs go to it as they are (flat and two sided shading stay on the CPU)
    if (g_packet && g_allow_raw && program && as.topology == Topology::Triangles && simple(kMShadeMode) != 0x1D00 &&
        !s.rs[kRsTwoSidedLighting]) {
        DrawPacket& pk = *g_packet;
        pk.raw = true;
        pk.in.swap(as.vertices);
        pk.raw_indices.swap(as.indices);
        pk.as = &as;
        pk.raw_count = as.index_count ? as.index_count : as.draw_count;
        pk.regs = as.regs;
        pk.vertex_count = as.vertex_count;
        pk.program = &s.vs->program;
        if (!s.vs->program_key) s.vs->program_key = Hash(s.vs->program.data(), s.vs->program.size() * 4) ^ s.vs->program.size();
        pk.program_hash = s.vs->program_key;
        pk.fogp[0] = AsFloat(s.rs[kRsFogStart]);
        pk.fogp[1] = AsFloat(s.rs[kRsFogEnd]);
        pk.fogp[2] = AsFloat(s.rs[kRsFogDensity]);
        pk.fogp[3] = s.rs[kRsFogEnable] ? float(s.rs[kRsFogTableMode] & 3) : -1.f;
        pk.specular = s.rs[kRsSpecularEnable] != 0;
        pk.cull = s.rs[kRsCullMode];
        pk.front_face = s.rs[kRsFrontFace];
        return;
    }
    int regs = 0, slot[16];
    for (int r = 0; r < 16; r++) slot[r] = (as.regs >> r) & 1 ? regs++ : -1;
    auto input = [&](uint32_t v, int r) { return &as.vertices[(size_t(v) * regs + slot[r]) * 4]; };
    uint32_t gw = rt.ok ? rt.w : zs.w, gh = rt.ok ? rt.h : zs.h;
    std::vector<Shaded> verts(as.vertex_count);
    static std::map<uint64_t, std::vector<Instr>> programs;
    const std::vector<Instr>* decoded = nullptr;
    if (program) {
        uint64_t h = Hash(s.vs->program.data(), s.vs->program.size() * 4) ^ s.vs->program.size();
        auto it = programs.find(h);
        if (it == programs.end()) it = programs.emplace(h, DecodeProgram(s.vs->program)).first;
        decoded = &it->second;
    }
    // the fixed function transform, once a draw rather than once a vertex
    float wvp[16];
    if (!program && !as.pretransformed) {
        auto mul = [](const float* x, const float* y, float* r) {
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    r[i * 4 + j] = x[i * 4] * y[j] + x[i * 4 + 1] * y[4 + j] + x[i * 4 + 2] * y[8 + j] + x[i * 4 + 3] * y[12 + j];
        };
        float wv[16];
        mul(s.transforms[256].data(), s.transforms[2].data(), wv);
        mul(wv, s.transforms[3].data(), wvp);
    }
    auto shade_vertex = [&](uint32_t vi) {
        VertexProgram vp{s};
        for (int r = 0; r < 16; r++) {
            if (slot[r] >= 0) {
                const float* p = input(vi, r);
                vp.in[r] = V(p[0], p[1], p[2], p[3]);
            } else {
                vp.in[r] = r == 3 ? V(1, 1, 1, 1) : V(0, 0, 0, 1);
            }
        }
        Shaded& o = verts[vi];
        if (program) {
            vp.Run(*decoded);
            o.pos = vp.out[0];
            o.pos[3] = MagClamp(o.pos[3]);
            o.attr[kD0] = vp.out[3];
            o.attr[kD1] = vp.out[4];
            o.attr[kB0] = vp.out[7];
            o.attr[kB1] = vp.out[8];
            o.attr[kFog] = Splat(s.rs[kRsFogEnable] && s.rs[kRsFogTableMode] == 0 ? Sat(vp.out[5][0]) : FogAmount(s, vp.out[5][0]));
            o.rawfog = vp.out[5][0];
            for (int t = 0; t < 4; t++) o.attr[kT0 + t] = vp.out[9 + t];
        } else {
            const Vec& q = vp.in[0];
            float fog_distance;
            if (as.pretransformed) {
                // half a pixel on: Direct3D's pixel centres are on whole coordinates, GL's half way between, so a
                // target sized pass from -0.5 to w - 0.5 missed the last column and row in GL. Those stayed stale and
                // NFS Most Wanted's nitro zoom blur smeared them into grey bands along the right and bottom
                o.pos = V(q[0] + 0.5f, q[1] + 0.5f, q[2] * s.zscale, q[3] == 0 ? 1 : 1 / q[3]);
                fog_distance = std::fabs(q[2]);
            } else {
                // world, view, projection, then the viewport, as the D3D12 path builds it
                float cp[4];
                for (int j = 0; j < 4; j++) cp[j] = q[0] * wvp[j] + q[1] * wvp[4 + j] + q[2] * wvp[8 + j] + wvp[12 + j];
                float w = MagClamp(cp[3]);
                const Viewport& v = s.viewport;
                o.pos = V(v.x + (cp[0] / w + 1) * 0.5f * v.w, v.y + (1 - cp[1] / w) * 0.5f * v.h,
                          (v.min_z + cp[2] / w * (v.max_z - v.min_z)) * s.zscale, w);
                // table fog runs on the clip space depth, as gpu/shaders.cpp's fixed function path does: the depth
                // buffer's 0..1 is never near the title's fog start and end, so the fog never showed
                fog_distance = std::fabs(cp[2]);
            }
            o.attr[kD0] = vp.in[3];
            o.attr[kD1] = slot[4] >= 0 ? vp.in[4] : Vec{};
            o.attr[kB0] = o.attr[kD0];
            o.attr[kB1] = o.attr[kD1];
            // no fog table on the fixed pipeline: the console's fog generator takes the vertex's specular alpha (0 when
            // the vertex has no specular colour), not a distance. Taking 1 here fogged every screen pass of NFS Most
            // Wanted's post processing to its fog colour, a flat gray, as its final combiner mixes r0 to c0 by fog.a.
            if (s.rs[kRsFogEnable] && s.rs[kRsFogTableMode] == 0)
                o.attr[kFog] = Splat(slot[4] >= 0 ? Sat(vp.in[4][3]) : 0.f);
            else
                o.attr[kFog] = Splat(FogAmount(s, fog_distance));
            for (int t = 0; t < 4; t++) o.attr[kT0 + t] = slot[9 + t] >= 0 ? vp.in[9 + t] : V(0, 0, 0, 1);
        }
        for (int a : {kD0, kD1, kB0, kB1})
            for (int j = 0; j < 4; j++) o.attr[a][j] = Sat(o.attr[a][j]);
        if (!s.rs[kRsSpecularEnable]) o.attr[kD1] = o.attr[kB1] = V(0, 0, 0, 1);
    };
    // vertices are independent: big draws are shaded in chunks on the worker threads (a race frame has hundreds of
    // vertex program draws, which on one core took over 100 ms a frame)
    constexpr uint32_t kChunk = 64;
    const uint32_t chunks = (as.vertex_count + kChunk - 1) / kChunk;
    auto chunk = [&](int c) {
        const uint32_t end = std::min(as.vertex_count, uint32_t(c + 1) * kChunk);
        for (uint32_t vi = uint32_t(c) * kChunk; vi < end; vi++) shade_vertex(vi);
    };
    if (chunks) ParallelRows(0, int(chunks) - 1, program && as.vertex_count >= 256, chunk);

    // vertex fog that is zero on every vertex would paint the whole draw in the fog colour (a flat gray arena);
    // that means the program's fog inputs were not set up, so such a draw is drawn unfogged
    // (the unfogging guard that was here blacked out the screen: the fog value is read by the final combiner)
    // raster and output state
    bool blend = simple(kMBlendEnable) != 0;
    uint32_t src_factor = simple(kMSrcBlend), dst_factor = simple(kMDestBlend), blend_op = simple(kMBlendOp);
    uint32_t cw = simple(kMColorWriteEnable);
    uint32_t cmask = (cw >> 16 & 1 ? 4u : 0u) | (cw >> 8 & 1 ? 2u : 0u) | (cw & 1 ? 1u : 0u) | (cw >> 24 & 1 ? 8u : 0u);
    bool ztest = zs.ok && s.rs[kRsZEnable], zwrite = ztest && simple(kMZWriteEnable);
    uint32_t zfunc = simple(kMZFunc);
    bool stencil = zs.ok && s.rs[kRsStencilEnable];
    uint32_t sfunc = simple(kMStencilFunc), sref = simple(kMStencilRef) & 0xFF, smask = simple(kMStencilMask) & 0xFF,
             swmask = simple(kMStencilWriteMask) & 0xFF;
    uint32_t op_fail = s.rs[kRsStencilFail], op_zfail = simple(kMStencilZFail), op_pass = simple(kMStencilPass);
    bool atest = simple(kMAlphaTestEnable) != 0;
    uint32_t afunc = simple(kMAlphaFunc), aref = simple(kMAlphaRef) & 0xFF;
    uint32_t cull = s.rs[kRsCullMode];
    bool flat = simple(kMShadeMode) == 0x1D00, two_sided = s.rs[kRsTwoSidedLighting] != 0;  // 0x1D00 GL_FLAT
    Vec blend_color = Unpack(simple(kMBlendColor));
    int left = int(s.viewport.x), top = int(s.viewport.y), right = left + int(s.viewport.w), bottom = top + int(s.viewport.h);
    if (s.scissor_count) {
        left = std::max(left, int(s.scissors[0]));
        top = std::max(top, int(s.scissors[1]));
        right = std::min(right, int(s.scissors[2]));
        bottom = std::min(bottom, int(s.scissors[3]));
    }
    left = std::max(left, 0);
    top = std::max(top, 0);
    right = std::min(right, int(gw));
    bottom = std::min(bottom, int(gh));
    // alpha only writes (the title's alpha clears) never reach the screen and cost a full screen of shading
    if (cmask == 8 && !ztest && !zwrite && !stencil) return;

    const float near_w = 1e-6f;
    // the GPU path shares one converted vertex between the triangles that use it
    std::vector<int32_t> vmap;
    if (g_packet) vmap.assign(as.vertex_count, -1);
    auto shared_vertex = [&](uint32_t i) -> uint32_t {
        if (vmap[i] >= 0) return uint32_t(vmap[i]);
        const Shaded& v = verts[i];
        GlVertex gv{};
        for (int j = 0; j < 3; j++) gv.clip[j] = v.pos[j] * v.pos[3];
        gv.clip[3] = v.pos[3];
        for (int j = 0; j < 4; j++) {
            gv.v0[j] = v.attr[kD0][j];
            gv.v1[j] = v.attr[kD1][j];
            for (int t = 0; t < 4; t++) gv.tc[t][j] = v.attr[kT0 + t][j];
        }
        gv.fog[0] = v.attr[kFog][0];
        g_packet->verts.push_back(gv);
        vmap[i] = int32_t(g_packet->verts.size() - 1);
        return uint32_t(vmap[i]);
    };
    for (size_t ti = 0; ti + 2 < as.indices.size(); ti += 3) {
        if (g_packet) {
            const uint32_t ia = as.indices[ti], ib = as.indices[ti + 1], ic = as.indices[ti + 2];
            const Shaded &va = verts[ia], &vb = verts[ib], &vc = verts[ic];
            // the GPU clips against the same volume (0 <= z <= zscale*w), so triangles that cross w = 0 go through
            // unclipped: a CPU clip at w = 1e-6 leaves vertices a million pixels away, which the guard band mangles
            const bool simple_tri = !flat && !two_sided;
            if (simple_tri) {
                float area;
                if (va.pos[3] > near_w && vb.pos[3] > near_w && vc.pos[3] > near_w) {
                    area = (vb.pos[0] - va.pos[0]) * (vc.pos[1] - va.pos[1]) - (vc.pos[0] - va.pos[0]) * (vb.pos[1] - va.pos[1]);
                } else {
                    // orientation of the visible part: sign of det [x y w] over the three corners
                    const Shaded* t3[3] = {&va, &vb, &vc};
                    float m[3][3];
                    bool ok = true;
                    for (int i = 0; i < 3; i++) {
                        m[i][0] = t3[i]->pos[0] * t3[i]->pos[3];
                        m[i][1] = t3[i]->pos[1] * t3[i]->pos[3];
                        m[i][2] = t3[i]->pos[3];
                        for (int j = 0; j < 3; j++) ok = ok && std::isfinite(m[i][j]);
                    }
                    if (!ok) continue;
                    area = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                           m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
                    // det [x y w] > 0 matches a positive screen area at w > 0 (y down)
                }
                if (area == 0 || !std::isfinite(area)) continue;
                bool clockwise = area > 0;
                if ((cull == 0x900 && clockwise) || (cull == 0x901 && !clockwise)) continue;
                bool front_face = s.rs[kRsFrontFace] == 0x901 ? !clockwise : clockwise;
                if (!flat && !(two_sided && !front_face)) {
                    g_packet->indices.push_back(shared_vertex(ia));
                    g_packet->indices.push_back(shared_vertex(ib));
                    g_packet->indices.push_back(shared_vertex(ic));
                    continue;
                }
            }
        }
        Clip tri[3];
        for (int i = 0; i < 3; i++) {
            const Shaded& v = verts[as.indices[ti + i]];
            for (int j = 0; j < 3; j++) tri[i].h[j] = v.pos[j] * v.pos[3];
            tri[i].h[3] = v.pos[3];
            std::copy(v.attr, v.attr + kAttrs, tri[i].attr);
        }
        // flat shading takes the first vertex's colours
        const Vec* provoking = tri[0].attr;
        Vec flat_colour[4] = {provoking[kD0], provoking[kD1], provoking[kB0], provoking[kB1]};
        // clipped against w > near_w: a triangle leaves at most four corners
        Clip poly[4];
        int corners = 0;
        for (int i = 0; i < 3; i++) {
            const Clip &p = tri[i], &q = tri[(i + 1) % 3];
            bool p_in = p.h[3] > near_w, q_in = q.h[3] > near_w;
            if (p_in) poly[corners++] = p;
            if (p_in == q_in) continue;
            float t = (near_w - p.h[3]) / (q.h[3] - p.h[3]);
            Clip& m = poly[corners++];
            for (int j = 0; j < 4; j++) m.h[j] = p.h[j] + (q.h[j] - p.h[j]) * t;
            for (int a = 0; a < kAttrs; a++)
                for (int j = 0; j < 4; j++) m.attr[a][j] = p.attr[a][j] + (q.attr[a][j] - p.attr[a][j]) * t;
        }
        if (corners < 3) continue;
        Screen sv[4];
        for (int i = 0; i < corners; i++) {
            float iw = 1 / poly[i].h[3];
            sv[i].x = poly[i].h[0] * iw;
            sv[i].y = poly[i].h[1] * iw;
            sv[i].z = poly[i].h[2] * iw;
            sv[i].iw = iw;
            for (int a = 0; a < kAttrs; a++)
                for (int j = 0; j < 4; j++) sv[i].attr[a][j] = poly[i].attr[a][j] * iw;
        }
        for (int fan = 1; fan + 1 < corners; fan++) {
            const Screen &p0 = sv[0], &p1 = sv[fan], &p2 = sv[fan + 1];
            float area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
            if (area == 0 || !std::isfinite(area)) continue;
            bool clockwise = area > 0;  // y down
            // GL_CW 0x900, GL_CCW 0x901
            if ((cull == 0x900 && clockwise) || (cull == 0x901 && !clockwise)) continue;
            bool front = s.rs[kRsFrontFace] == 0x901 ? !clockwise : clockwise;
            if (g_packet) {
                bool back = two_sided && !front;
                for (int c : {0, fan, fan + 1}) {
                    const Clip& pn = poly[c];
                    GlVertex gv{};
                    for (int j = 0; j < 4; j++) {
                        gv.clip[j] = pn.h[j];
                        gv.v0[j] = flat ? flat_colour[back ? 2 : 0][j] : pn.attr[back ? kB0 : kD0][j];
                        gv.v1[j] = flat ? flat_colour[back ? 3 : 1][j] : pn.attr[back ? kB1 : kD1][j];
                        for (int t = 0; t < 4; t++) gv.tc[t][j] = pn.attr[kT0 + t][j];
                    }
                    gv.fog[0] = pn.attr[kFog][0];
                    g_packet->verts.push_back(gv);
                    g_packet->indices.push_back(uint32_t(g_packet->verts.size() - 1));
                }
                continue;
            }
            int x0 = std::max(left, int(std::floor(std::min({p0.x, p1.x, p2.x})))), x1 = std::min(right - 1, int(std::ceil(std::max({p0.x, p1.x, p2.x}))));
            int y0 = std::max(top, int(std::floor(std::min({p0.y, p1.y, p2.y})))), y1 = std::min(bottom - 1, int(std::ceil(std::max({p0.y, p1.y, p2.y}))));
            if (x0 > x1 || y0 > y1) continue;
            g_tris++;
            g_pixels += uint64_t(y1 - y0 + 1) * uint64_t(x1 - x0 + 1);
            float inv_area = 1 / area;
            // only the attributes the shading reads are interpolated
            int need[kAttrs], nneed = 0;
            for (int a = 0; a < kAttrs; a++) {
                bool used = true;
                if (a == kB0 || a == kB1) used = two_sided;
                else if (a >= kT0) used = mat.key.programmable ? (((mat.key.tex_modes >> ((a - kT0) * 5)) & 0x1F) != 0 || mat.info[a - kT0].kind) : mat.info[a - kT0].kind != 0;
                if (used || flat) need[nneed++] = a;
            }
            auto row = [&](int y) {
                for (int x = x0; x <= x1; x++) {
                    // barycentrics at the pixel centre
                    float cx = x + 0.5f, cy = y + 0.5f;
                    float b0 = ((p1.x - cx) * (p2.y - cy) - (p2.x - cx) * (p1.y - cy)) * inv_area;
                    float b1 = ((p2.x - cx) * (p0.y - cy) - (p0.x - cx) * (p2.y - cy)) * inv_area;
                    float b2 = 1 - b0 - b1;
                    if (b0 < 0 || b1 < 0 || b2 < 0) continue;
                    float z = b0 * p0.z + b1 * p1.z + b2 * p2.z;
                    if (z > s.zscale && z < s.zscale * 1.0001f) z = s.zscale;
                    if (z < 0 || z > s.zscale) continue;
                    float iw = b0 * p0.iw + b1 * p1.iw + b2 * p2.iw;
                    Vec attr[kAttrs] = {};
                    const float inv_w = 1 / iw;
                    for (int ni = 0; ni < nneed; ni++) {
                        const int a = need[ni];
                        for (int j = 0; j < 4; j++) attr[a][j] = (b0 * p0.attr[a][j] + b1 * p1.attr[a][j] + b2 * p2.attr[a][j]) * inv_w;
                    }
                    if (flat) std::copy(flat_colour, flat_colour + 4, attr);
                    Fragment frag;
                    bool back = two_sided && !front;
                    frag.v0 = attr[back ? kB0 : kD0];
                    frag.v1 = attr[back ? kB1 : kD1];
                    frag.fog = Unpack(s.rs[kRsFogColor]);
                    frag.fog[3] = Sat(attr[kFog][0]);
                    for (int t = 0; t < 4; t++) frag.tc[t] = attr[kT0 + t];
                    Vec col;
                    bool keep = mat.key.programmable ? Combiners{mat, frag}.Shade(col) : ShadeStages(mat, frag, col);
                    if (!keep) continue;
                    if (atest && !Passes(afunc, uint32_t(int(std::lround(Sat(col[3]) * 255))), aref)) continue;
                    uint32_t oz = 0, os = 0, zi = uint32_t(std::floor(z));
                    if (zs.ok) zs.LoadDepth(uint32_t(x), uint32_t(y), oz, os);
                    auto stencil_op = [&](uint32_t op) { return (os & ~swmask) | (StencilResult(op, os, sref) & swmask); };
                    if (stencil && !Passes(sfunc, sref & smask, os & smask)) {
                        zs.StoreDepth(uint32_t(x), uint32_t(y), oz, stencil_op(op_fail));
                        continue;
                    }
                    if (ztest && !Passes(zfunc, zi, oz)) {
                        if (stencil) zs.StoreDepth(uint32_t(x), uint32_t(y), oz, stencil_op(op_zfail));
                        continue;
                    }
                    if (zs.ok && (zwrite || stencil)) zs.StoreDepth(uint32_t(x), uint32_t(y), zwrite ? zi : oz, stencil ? stencil_op(op_pass) : os);
                    if (!rt.ok || !cmask) continue;
                    for (int j = 0; j < 4; j++) col[j] = Sat(col[j]);
                    if (blend) {
                        Vec d = rt.Load(uint32_t(x), uint32_t(y)), r;
                        for (int j = 0; j < 4; j++) {
                            float a = col[j] * BlendFactor(src_factor, col, d, blend_color, j);
                            float b = d[j] * BlendFactor(dst_factor, col, d, blend_color, j);
                            switch (blend_op) {  // GL subtract, reverse subtract, min, max, else add
                                case 0x800A: r[j] = a - b; break;
                                case 0x800B: r[j] = b - a; break;
                                case 0x8007: r[j] = std::min(col[j], d[j]); break;
                                case 0x8008: r[j] = std::max(col[j], d[j]); break;
                                default: r[j] = a + b; break;
                            }
                        }
                        col = r;
                    }
                    rt.Store(uint32_t(x), uint32_t(y), col, cmask);
                }
            };
            ParallelRows(y0, y1, (y1 - y0 + 1) * (x1 - x0 + 1) > 3000, row);
        }
    }
    if (g_packet) return;
    if (rt.ok && cmask) Written(rt.data, rt.End());
    if (zs.ok) Written(zs.data, zs.End());
}

// the surface's colour as RGBA8888 (R in the low byte), top row first
bool ReadRgba(const Surface& s, uint32_t& w, uint32_t& h, std::vector<uint32_t>& out) {
    Plane p = OpenPlane(s);
    if (!p.ok) return false;
    w = p.w; h = p.h;
    out.resize(size_t(w) * h);
    if (p.bytes == 4) {
        // memory B G R A to R G B A, opaque
        for (uint32_t y = 0; y < h; y++) {
            uint32_t* o = &out[size_t(y) * w];
            if (!p.swizzled) {
                const uint8_t* row = p.At(0, y);
                for (uint32_t x = 0; x < w; x++) {
                    uint32_t c;
                    std::memcpy(&c, row + x * 4, 4);
                    o[x] = (c >> 16 & 0xFF) | (c & 0xFF00) | (c & 0xFF) << 16 | 0xFF000000u;
                }
            } else {
                for (uint32_t x = 0; x < w; x++) {
                    uint32_t c;
                    std::memcpy(&c, p.At(x, y), 4);
                    o[x] = (c >> 16 & 0xFF) | (c & 0xFF00) | (c & 0xFF) << 16 | 0xFF000000u;
                }
            }
        }
        return true;
    }
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            Vec c = p.Load(x, y);
            out[size_t(y) * w + x] = uint32_t(Sat(c[0]) * 255) | uint32_t(Sat(c[1]) * 255) << 8 |
                                     uint32_t(Sat(c[2]) * 255) << 16 | 0xFF000000u;
        }
    return true;
}

void SetGpuMode(bool on) { g_gpu_mode = on; }
void SetTargetTest(TargetTest test) { g_target_test = test; }

bool Prepare(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices, DrawPacket& out, bool allow_raw) {
    out.raw = false;
    out.raw_indices.clear();
    out.as = nullptr;
    out.raw_count = 0;
    g_allow_raw = allow_raw;
    out.verts.clear();
    out.indices.clear();
    g_packet = &out;
    Draw(s, prim, start, count, indices);
    g_packet = nullptr;
    g_allow_raw = false;
    return out.raw ? out.raw_count > 0 : !out.indices.empty();
}

bool HeaderRange(uint32_t header, uint32_t& lo, uint32_t& hi, uint32_t& w, uint32_t& h) {
    TextureDesc d;
    if (!header || !Describe(header, d)) return false;
    lo = d.data;
    hi = d.data + (d.cube ? FaceSize(d) * 6 : TotalSize(d));
    w = d.width;
    h = d.height;
    return true;
}

void Invalidate(uint32_t lo, uint32_t hi) { Written(lo, hi); }

bool WriteRgba(const Surface& s, const uint32_t* rgba, uint32_t w, uint32_t h) {
    Plane p = OpenPlane(s);
    if (!p.ok || p.w != w || p.h != h) return false;
    if (p.bytes == 4) {
        // R G B A to memory B G R A
        for (uint32_t y = 0; y < h; y++) {
            const uint32_t* in = &rgba[size_t(y) * w];
            for (uint32_t x = 0; x < w; x++) {
                uint32_t c = in[x], m = (c >> 16 & 0xFF) | (c & 0xFF00FF00u) | (c & 0xFF) << 16;
                std::memcpy(p.At(x, y), &m, 4);
            }
        }
        Written(p.data, p.End());
        return true;
    }
    for (uint32_t y = 0; y < h; y++)
        for (uint32_t x = 0; x < w; x++) {
            uint32_t c = rgba[size_t(y) * w + x];
            p.Store(x, y, V((c & 255) / 255.f, (c >> 8 & 255) / 255.f, (c >> 16 & 255) / 255.f, (c >> 24) / 255.f), 15);
        }
    Written(p.data, p.End());
    return true;
}

bool TestShade(const PixelKey& key, const TexInfo info[4], const uint8_t* const rgba[4], const uint32_t size[4][2], const float v0[4],
               const float v1[4], const float tc[4][4], const float pc[18][4], const float tfactor[4], const float fog[4], const float bump[4][6], float out[4]) {
    Material m;
    m.key = key;
    State st;
    m.s = &st;
    auto bits = [](float f) {
        uint32_t u;
        std::memcpy(&u, &f, 4);
        return u;
    };
    for (int i = 0; i < 4; i++) {
        st.tss[size_t(i)][kTssBumpEnvMat00] = bits(bump[i][0]);
        st.tss[size_t(i)][kTssBumpEnvMat01] = bits(bump[i][1]);
        st.tss[size_t(i)][kTssBumpEnvMat10] = bits(bump[i][2]);
        st.tss[size_t(i)][kTssBumpEnvMat11] = bits(bump[i][3]);
        st.tss[size_t(i)][kTssBumpEnvLScale] = bits(bump[i][4]);
        st.tss[size_t(i)][kTssBumpEnvLOffset] = bits(bump[i][5]);
    }
    std::shared_ptr<Image> imgs[4];
    for (int i = 0; i < 4; i++) {
        m.info[i] = info[i];
        if (!info[i].kind) continue;
        auto img = std::make_shared<Image>();
        img->w = size[i][0];
        img->h = size[i][1];
        img->faces = info[i].kind == kTexCube ? 6 : 1;
        size_t face_texels = size_t(img->w) * img->h;
        for (uint32_t f = 0; f < img->faces; f++) {
            img->texels[f].resize(face_texels);
            for (size_t p = 0; p < face_texels; p++)
                for (int j = 0; j < 4; j++) img->texels[f][p][j] = rgba[i][(f * face_texels + p) * 4 + size_t(j)] / 255.f;
        }
        imgs[i] = img;
        m.unit[i].img = img;
        m.unit[i].address_u = m.unit[i].address_v = 1;
        m.unit[i].mag = m.unit[i].min = 1;
    }
    for (int i = 0; i < 18; i++)
        for (int j = 0; j < 4; j++) m.pc[i][j] = pc[i][j];
    for (int j = 0; j < 4; j++) m.tfactor[j] = tfactor[j];
    Fragment f;
    for (int j = 0; j < 4; j++) {
        f.v0[j] = v0[j];
        f.v1[j] = v1[j];
        f.fog[j] = fog[j];
        for (int t = 0; t < 4; t++) f.tc[t][j] = tc[t][j];
    }
    Vec col;
    bool keep = key.programmable ? Combiners{m, f}.Shade(col) : ShadeStages(m, f, col);
    for (int j = 0; j < 4; j++) out[j] = Sat(col[j]);
    return keep;
}

}  // namespace reference
