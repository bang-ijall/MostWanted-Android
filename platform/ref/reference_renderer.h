#pragma once
#include <cstdint>
#include <atomic>
#include <string>
#include <vector>

#include <memory>
#include "gpu/shaders.h"
#include "gpu/state.h"
#include "gpu/vertex.h"

// CPU renderer for the headless harness: draws the title's calls into its own surfaces in guest memory, so a frame
// can be looked at without a GPU. Follows the same decoding of the XDK state as the SDK's shader translator.
namespace reference {
struct DrawDbg {
    uint32_t inside = 0, zfail = 0, shadefail = 0, shaded = 0, written = 0, nonblack = 0;
    float z = 0, zscale = 0, col[4] = {0, 0, 0, 0};
};
extern DrawDbg g_dbg;

extern std::string g_dump_dir;
extern std::atomic<uint64_t> g_prep_us, g_pixels, g_tris, g_img_us, g_vs_us, g_nontri, g_prepare_us;

void BeginFrame();  // drops decoded textures and restarts the draw count
void Clear(const gpu::State& s, uint32_t count, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil);
void Copy(const gpu::Surface& src, const gpu::Surface& dst, uint32_t count, uint32_t rects_va, uint32_t points_va);
void Draw(const gpu::State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices);
// 24 bit TGA of a surface's colour
bool ReadRgba(const gpu::Surface& s, uint32_t& w, uint32_t& h, std::vector<uint32_t>& out);


// the GPU backend takes a draw as clipped screen triangles plus what the pixel stage needs, and draws them itself
struct GlVertex {
    float clip[4];   // x * w, y * w (pixels), z * w (depth units), w
    float v0[4], v1[4];
    float fog[4];    // amount in x
    float tc[4][4];
};
struct GlStage {
    uint64_t id = 0;    // changes when the texture's bytes do
    uint64_t slot = 0;  // stays while the texture's address, format and size do: the GPU copy is updated in place
    uint32_t target = 0;  // nonzero: the stage reads the GPU render target over this guest address, rgba is null
    uint32_t format = 0;  // the texture's console format
    uint32_t face_stride = 0;  // a cube over six render targets: bytes from one face's target to the next
    uint32_t w = 0, h = 0;
    const uint8_t* rgba = nullptr;  // w * h * 4, R first
    const uint8_t* faces[6] = {};   // a cube's six, +x -x +y -y +z -z
    uint32_t address_u = 1, address_v = 1, mag = 2, min = 2, mip = 0;
    float border[4] = {};
};
struct DrawPacket {
    std::vector<GlVertex> verts;
    std::vector<uint32_t> indices;  // a triangle list over verts
    gpu::PixelKey key;
    gpu::TexInfo info[4];
    GlStage stage[4];
    std::shared_ptr<void> hold[4];
    float pc[18][4] = {};
    float tfactor[4] = {};
    float fog_color[4] = {};
    // raw: the draw's vertex program runs on the GPU. verts and indices stay empty; the inputs as the input assembler
    // made them (vertex_count * popcount(regs) vec4s, registers in order) and a triangle list over them come instead
    bool raw = false;
    std::vector<float> in;
    std::vector<uint32_t> raw_indices;
    uint16_t regs = 0;
    uint32_t vertex_count = 0;
    const std::vector<uint32_t>* program = nullptr;
    uint64_t program_hash = 0;
    float fogp[4] = {};      // start, end, density, mode: -1 fog off, 0 the program's oFog as is, 1 exp, 2 exp2, 3 linear
    bool specular = true;    // D3DRS_SPECULARENABLE
    uint32_t cull = 0, front_face = 0;
    const gpu::Assembled* as = nullptr;  // the draw's vertex buffers, formats and indices, valid until the next Prepare
    uint32_t raw_count = 0;              // indices, or vertices of an unindexed draw
};
void SetGpuMode(bool on);  // images keep RGBA8 only, and Written is not needed for them
// the GPU backend says which textures are its render targets: those stages are not decoded from guest memory
using TargetTest = bool (*)(uint32_t data, uint32_t format, uint32_t w, uint32_t h);
void SetTargetTest(TargetTest test);
// allow_raw: a draw with a vertex program may come back raw, for the GPU to run the program
bool Prepare(const gpu::State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices, DrawPacket& out,
             bool allow_raw = false);
// the guest bytes a texture or surface header describes
bool HeaderRange(uint32_t header, uint32_t& lo, uint32_t& hi, uint32_t& w, uint32_t& h);
// guest bytes were written by the GPU path: decoded images over them are stale
void Invalidate(uint32_t lo, uint32_t hi);
// RGBA8 (R in the low byte), top row first, into the surface
bool WriteRgba(const gpu::Surface& s, const uint32_t* rgba, uint32_t w, uint32_t h);

// test hook: one fragment through the CPU shading, textures given as w x h RGBA8, point sampled and wrapped
bool TestShade(const gpu::PixelKey& key, const gpu::TexInfo info[4], const uint8_t* const rgba[4], const uint32_t size[4][2],
               const float v0[4], const float v1[4], const float tc[4][4], const float pc[18][4], const float tfactor[4],
               const float fog[4] /*colour, amount*/, const float bump[4][6] /*m00 m01 m10 m11 lscale loffset*/, float out[4]);

}  // namespace reference
