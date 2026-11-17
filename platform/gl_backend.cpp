// GPU renderer: OpenGL ES 3 through EGL. The CPU reference renderer still does the vertex side (assembly, vertex
// programs, clipping, culling); the GPU rasterises its triangles and runs the pixel stage as GLSL from gl_shader.cpp.
// Render targets live on the GPU and are read back into guest memory only when the title looks at that memory.
// A texture over a render target is that target: the GPU samples it in place, nothing goes through guest memory.
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl32.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "frame.h"
#include "gl_shader.h"
#include "gpu/backend.h"
#include "power.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include <dxgiformat.h>
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "reference_renderer.h"

namespace gpu {
namespace {

constexpr float kDepthScale = 0.99995f;

// colour formats a render target and a texture over it can share: same channels, the alias reads what was drawn
bool TargetColourFormat(uint32_t f) {
    switch (f) {
        case 0x05: case 0x11:  // R5G6B5
        case 0x06: case 0x12:  // A8R8G8B8
        case 0x07: case 0x1E:  // X8R8G8B8
            return true;
        default: return false;
    }
}
// formats read with alpha 1
bool OpaqueFormat(uint32_t f) { return f == 0x05 || f == 0x11 || f == 0x07 || f == 0x1E; }

struct Target {
    uint32_t data = 0, end = 0, w = 0, h = 0, format = 0;
    uint64_t last_world = 0;    // the last frame this target took a whole frame's drawing (over 64 draws)
    uint32_t sw = 0, sh = 0;    // the GPU image's size: the console's times the render scale
    Surface surf;
    GLuint tex = 0, read_fbo = 0;
    GLuint copy = 0;            // the target as it was, for a draw that reads the target it draws to
    GLint swizzle_a = GL_ALPHA, copy_swizzle_a = GL_ALPHA;
    uint64_t gen = 1;           // counts the GPU's writes to it, for copies made from it
    unsigned long long last_draw = 0;  // draws_seen_ at its latest draw or clear
    uint32_t frame_draws = 0;          // draws into it since the last present
    uint64_t known[8] = {};      // guest memory when the GPU copy last matched it: hashes of its 8 byte stripes
    uint32_t next_stripe = 0;    // the stripe compared next
    unsigned long long checked = ~0ull;  // the frame guest memory was last compared
    bool dirty = false;
};
struct DepthBuf {
    uint32_t data = 0, w = 0, h = 0, sw = 0, sh = 0;
    GLuint tex = 0;               // GL_DEPTH24_STENCIL8, readable: titles sample their depth buffer as colour
    GLuint colour = 0, fbo = 0;   // the depth as the console's memory holds it, read as A8R8G8B8
    bool initialised = false;
    bool written = true;          // depth or stencil changed since the colour copy was made
};
struct Prog {
    GLuint id = 0;
    GLint pc = -1, tf = -1, fogc = -1, at = -1, surf = -1, bm = -1, bl = -1, tsz = -1;
    GLint c = -1, fogp = -1, vsmisc = -1;  // a raw draw's vertex program
    std::vector<GLint> c_at;                // u_c[i]
    std::vector<float> cache;               // the constants last sent
    std::vector<float> ucache;              // surf, pc, tfactor, fog colour, alpha test last sent
    std::vector<float> tszc, vsc;           // texture sizes; fog and specular of a raw draw, last sent
    bool ok = false;
};
struct GlTex {
    GLuint id = 0;
    uint64_t image = 0;  // reference image id it holds
    uint32_t w = 0, h = 0;
    int levels = 1;
    bool cube = false;
    uint64_t used = 0;
};
// a draw counted for a visibility test: its query and the pixels it covers
struct Counted {
    GLuint query = 0;
    double area = 0;
    int passed = -1;  // -1 not read yet
    // for <shots>/vislog: the test's depth range (z / w), depth function and write, colour mask
    float zmin = 0, zmax = 0, wmin = 0;
    uint32_t zfunc = 0, flags = 0;
    bool logged = false;
};
struct Report {
    uint32_t va = 0;
    std::vector<std::shared_ptr<Counted>> counted;
    unsigned long long frame = 0;
};

// GL state as last set by Draw, so unchanged state is not sent again; Invalidate when other code changed it
struct GlState {
    int caps[5];  // blend, depth test, stencil test, scissor test, cull face
    GLuint fbo;
    int vw, vh;
    std::vector<uint32_t> front, scissor, cmask, zfunc, zmask, stencil, blend;
    GlState() { Invalidate(); }
    void Invalidate() {
        for (int& c : caps) c = -1;
        fbo = ~0u;
        vw = vh = -1;
        for (auto* v : {&front, &scissor, &cmask, &zfunc, &zmask, &stencil, &blend}) v->clear();
    }
    void Cap(GLenum cap, int i, bool on) {
        if (caps[i] == int(on)) return;
        caps[i] = int(on);
        if (on) glEnable(cap);
        else glDisable(cap);
    }
    void Fbo(GLuint f) {
        if (fbo == f) return;
        fbo = f;
        glBindFramebuffer(GL_FRAMEBUFFER, f);
    }
    void Viewport(int w, int h) {
        if (vw == w && vh == h) return;
        vw = w;
        vh = h;
        glViewport(0, 0, w, h);
    }
    template <class F>
    void Set(std::vector<uint32_t>& slot, std::initializer_list<uint32_t> v, F&& apply) {
        if (slot.size() == v.size() && std::equal(v.begin(), v.end(), slot.begin())) return;
        slot.assign(v);
        apply();
    }
};

class GlBackend;
GlBackend* g_self = nullptr;
bool IsTargetThunk(uint32_t data, uint32_t format, uint32_t w, uint32_t h);

class GlBackend final : public Backend {
public:
    ~GlBackend() override {
        if (g_self == this) g_self = nullptr;
        reference::SetTargetTest(nullptr);
        // the counted draws hand their queries back to free_queries_, which is destroyed before these members
        reports_.clear();
        counted_.clear();
    }

    // creates the context on the calling thread and releases it, the drawing thread takes it later
    bool Setup() {
        dpy_ = eglGetDisplay(EGL_DEFAULT_DISPLAY);
        EGLint major = 0, minor = 0;
        if (dpy_ == EGL_NO_DISPLAY || !eglInitialize(dpy_, &major, &minor)) {
            return false;
        }
        const EGLint cfg_attr[] = {EGL_SURFACE_TYPE, EGL_PBUFFER_BIT | EGL_WINDOW_BIT, EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT,
                                   EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_ALPHA_SIZE, 8, EGL_NONE};
        EGLConfig cfg;
        EGLint n = 0;
        if (!eglChooseConfig(dpy_, cfg_attr, &cfg, 1, &n) || n < 1) {
            return false;
        }
        const EGLint pb[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
        surf_ = eglCreatePbufferSurface(dpy_, cfg, pb);
        const EGLint ctx_attr[] = {EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE};
        eglBindAPI(EGL_OPENGL_ES_API);
        ctx_ = eglCreateContext(dpy_, cfg, EGL_NO_CONTEXT, ctx_attr);
        // finished frames to the window as EGLImages (platform::PresentChain), when the window and EGL can
        auto& chain = platform::GetPresentChain();
        p_image_target_ = reinterpret_cast<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>(eglGetProcAddress("glEGLImageTargetTexture2DOES"));
        p_create_sync_ = reinterpret_cast<PFNEGLCREATESYNCKHRPROC>(eglGetProcAddress("eglCreateSyncKHR"));
        p_wait_sync_ = reinterpret_cast<PFNEGLWAITSYNCKHRPROC>(eglGetProcAddress("eglWaitSyncKHR"));
        p_destroy_sync_ = reinterpret_cast<PFNEGLDESTROYSYNCKHRPROC>(eglGetProcAddress("eglDestroySyncKHR"));
        p_dup_fd_ = reinterpret_cast<PFNEGLDUPNATIVEFENCEFDANDROIDPROC>(eglGetProcAddress("eglDupNativeFenceFDANDROID"));
        p_client_buffer_ = reinterpret_cast<PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC>(eglGetProcAddress("eglGetNativeClientBufferANDROID"));
        p_create_image_ = reinterpret_cast<PFNEGLCREATEIMAGEKHRPROC>(eglGetProcAddress("eglCreateImageKHR"));
        p_destroy_image_ = reinterpret_cast<PFNEGLDESTROYIMAGEKHRPROC>(eglGetProcAddress("eglDestroyImageKHR"));
        if (chain.window_ready && p_image_target_ && p_create_sync_ && p_wait_sync_ && p_destroy_sync_ && p_dup_fd_ &&
            p_client_buffer_ && p_create_image_ && p_destroy_image_)
            chain.active = true;
        if (ctx_ == EGL_NO_CONTEXT || surf_ == EGL_NO_SURFACE) {
            return false;
        }
        if (!eglMakeCurrent(dpy_, surf_, surf_, ctx_)) {
            return false;
        }
        const char* r = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
        const char* v = reinterpret_cast<const char*>(glGetString(GL_VERSION));
        driver_ = std::string(r ? r : "") + "|" + (v ? v : "");
        InitObjects();
        eglMakeCurrent(dpy_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        owner_ = std::thread::id();
        g_self = this;
        reference::SetTargetTest(&IsTargetThunk);
        return true;
    }

    bool Init(uint32_t w, uint32_t h) override {
        kx_ = std::clamp(platform::RenderScaleX().load(), 0.5f, 6.f);
        ky_ = std::clamp(platform::RenderScaleY().load(), 0.5f, 6.f);
        if (!reference::g_dump_dir.empty()) {
            cache_dir_ = (std::filesystem::path(reference::g_dump_dir).parent_path() / "glcache").string();
            std::error_code ec;
            std::filesystem::create_directories(cache_dir_, ec);
        }
        reference::SetGpuMode(true);
        reference::BeginFrame();
        return true;
    }

    // the texture at data is exactly one of the GPU's render targets, in a format that reads it as drawn
    bool IsTarget(uint32_t data, uint32_t format, uint32_t w, uint32_t h) const {
        if (!TargetColourFormat(format)) return false;
        for (const auto& t : targets_)
            if (t->data == data && t->w == w && t->h == h)
                return TargetColourFormat(t->format) && Format(t->format).bits == Format(format).bits;
        // a depth buffer read as a 32 bit colour texture (the console's trick for depth fog and depth of field)
        if (Format(format).bits == 32)
            for (const auto& d : depths_)
                if (d->data == data && d->w == w && d->h == h) return true;
        return false;
    }

    void Clear(const State& s, uint32_t n, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) override {
        Ensure();
        Dirty();
        last_clear_color_ = color; last_clear_flags_ = flags; last_clear_z_ = z; last_clear_n_ = n;
        Target* rt = FindTarget(s.rt, true);
        if (!rt) return;
        CheckGuest(*rt);
        DepthBuf* zs = nullptr;
        if (flags & 3) zs = FindDepth(s.zs, *rt);
        glBindFramebuffer(GL_FRAMEBUFFER, Fbo(*rt, zs));
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glEnable(GL_SCISSOR_TEST);
        uint32_t mask = (flags & 0x10 ? 4u : 0u) | (flags & 0x20 ? 2u : 0u) | (flags & 0x40 ? 1u : 0u) | (flags & 0x80 ? 8u : 0u);
        glColorMask((mask & 4) != 0, (mask & 2) != 0, (mask & 1) != 0, (mask & 8) != 0);
        glDepthMask(GL_TRUE);
        glStencilMask(0xFF);
        glClearColor(float((color >> 16) & 255) / 255.f, float((color >> 8) & 255) / 255.f, float(color & 255) / 255.f, float(color >> 24) / 255.f);
        glClearDepthf(((2 * z - 1) * kDepthScale + 1) * 0.5f);
        glClearStencil(int(stencil & 0xFF));
        GLbitfield bits = 0;
        if ((flags & 0xF0) && mask) bits |= GL_COLOR_BUFFER_BIT;
        if ((flags & 1) && zs) bits |= GL_DEPTH_BUFFER_BIT;
        if ((flags & 3) && zs) zs->written = true;
        if ((flags & 2) && zs) bits |= GL_STENCIL_BUFFER_BIT;
        if (bits) {
            if (n && rects) {
                for (uint32_t i = 0; i < n; i++) {
                    int32_t r[4];
                    std::memcpy(r, rects + i * 16, 16);
                    Scissor(r[0], r[1], r[2], r[3], *rt);
                    glClear(bits);
                }
            } else {
                int x = int(s.viewport.x), y = int(s.viewport.y);
                Scissor(x, y, x + int(s.viewport.w), y + int(s.viewport.h), *rt);
                glClear(bits);
            }
            if (bits & GL_COLOR_BUFFER_BIT) {
                rt->dirty = true;
                rt->gen++;
                rt->last_draw = ++draws_seen_;
            }
        }
    }

    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override {
        Ensure();
        draws_seen_++;
        // a texture over a render target's memory, read as something other than that target, is read back first;
        // one that is the target is sampled on the GPU
        for (uint32_t th : s.textures) {
            TextureDesc d;
            if (!th || !Describe(th, d)) continue;
            if (!d.cube && d.depth == 1 && IsTarget(d.data, d.format, d.width, d.height)) continue;
            if (d.cube && CubeOfTargets(d)) continue;
            uint32_t lo, hi, w, h;
            if (reference::HeaderRange(th, lo, hi, w, h)) SyncRange(lo, hi, "texture");
        }
        // a draw counted for a visibility test needs its screen positions on the CPU; the others run their vertex
        // program on the GPU (on the CPU it took most of a race frame's time)
        const bool counted = s.simple[kMCountEnable >> 2] != 0;
        if (!reference::Prepare(s, prim, start, count, indices, pk_, !counted)) return;
        last_fog_color_ = s.rs[kRsFogColor];
        Target* rt = FrameBufferFor(FindTarget(s.rt, true));
        if (!rt) return;
        if (pending_clear_ && frame_buffers_.count(rt->data)) ApplyPendingClear(rt);
        CheckGuest(*rt);
        bool want_zs = s.rs[kRsZEnable] || s.rs[kRsStencilEnable];
        DepthBuf* zs = want_zs ? FindDepth(s.zs, *rt) : nullptr;

        const bool atest = s.simple[kMAlphaTestEnable >> 2] != 0;
        Prog* prog = GetProgram(pk_, atest);
        if (!prog || !prog->ok) return;

        // textures first: a target read by its own draw is copied, which uses the read framebuffer. The work of
        // finding or updating a texture binds on unit 7; units 0 to 3 get the draw's textures, only where they changed
        glActiveTexture(GL_TEXTURE7);
        GLuint tex_id[4] = {}, smp_id[4] = {};
        GLenum tex_target[4] = {};
        for (int i = 0; i < 4; i++) {
            if (!pk_.info[i].kind) continue;
            const reference::GlStage& stg = pk_.stage[i];
            bool cube = pk_.info[i].kind == kTexCube;
            if (stg.target && cube) {
                tex_target[i] = GL_TEXTURE_CUBE_MAP;
                tex_id[i] = CubeTexture(stg);
                smp_id[i] = SamplerFor(stg, true, true);
            } else if (stg.target) {
                tex_target[i] = GL_TEXTURE_2D;
                tex_id[i] = TargetTexture(stg, rt);
                smp_id[i] = SamplerFor(stg, true, false);
            } else {
                tex_target[i] = cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
                tex_id[i] = TextureFor(stg, pk_.info[i].rect, cube);
                smp_id[i] = SamplerFor(stg, pk_.info[i].rect, cube);
            }
        }
        for (int i = 0; i < 4; i++) {
            if (!tex_target[i]) continue;
            const uint64_t want = uint64_t(tex_target[i]) << 32 | tex_id[i];
            if (unit_tex_[i] != want) {
                glActiveTexture(GL_TEXTURE0 + i);
                glBindTexture(tex_target[i], tex_id[i]);
                unit_tex_[i] = want;
            }
            if (unit_smp_[i] != smp_id[i]) {
                glBindSampler(GLuint(i), smp_id[i]);
                unit_smp_[i] = smp_id[i];
            }
        }
        glActiveTexture(GL_TEXTURE7);

        // state through gs_: the calls the GPU already has are not sent again (about 700 draws a race frame)
        gs_.Fbo(Fbo(*rt, zs));
        gs_.Viewport(int(rt->sw), int(rt->sh));
        // the CPU culled its own triangles; the GPU culls a raw draw's. Screen y runs down on the console and up in
        // GL, so the console's clockwise (GL_CW, 0x900, culled) is GL's counter clockwise
        if (pk_.raw && (pk_.cull == 0x900 || pk_.cull == 0x901)) {
            gs_.Cap(GL_CULL_FACE, 4, true);
            gs_.Set(gs_.front, {pk_.cull == 0x900 ? uint32_t(GL_CW) : uint32_t(GL_CCW)}, [&] { glCullFace(GL_BACK); glFrontFace(pk_.cull == 0x900 ? GL_CW : GL_CCW); });
        } else {
            gs_.Cap(GL_CULL_FACE, 4, false);
        }
        // scissor: viewport and scissor rect
        int left = int(s.viewport.x), top = int(s.viewport.y), right = left + int(s.viewport.w), bottom = top + int(s.viewport.h);
        if (s.scissor_count) {
            left = std::max(left, int(s.scissors[0]));
            top = std::max(top, int(s.scissors[1]));
            right = std::min(right, int(s.scissors[2]));
            bottom = std::min(bottom, int(s.scissors[3]));
        }
        gs_.Cap(GL_SCISSOR_TEST, 3, true);
        {
            int x0 = int(SX(std::max(left, 0))), y0 = int(SY(std::max(top, 0)));
            int x1 = int(SX(std::min(right, int(rt->w)))), y1 = int(SY(std::min(bottom, int(rt->h))));
            int sw = std::max(x1 - x0, 0), sh = std::max(y1 - y0, 0);
            gs_.Set(gs_.scissor, {uint32_t(x0), uint32_t(y0), uint32_t(sw), uint32_t(sh)}, [&] { glScissor(x0, y0, sw, sh); });
        }

        uint32_t cw = s.simple[kMColorWriteEnable >> 2];
        uint32_t cmask = (cw >> 16 & 1 ? 4u : 0u) | (cw >> 8 & 1 ? 2u : 0u) | (cw & 1 ? 1u : 0u) | (cw >> 24 & 1 ? 8u : 0u);
        gs_.Set(gs_.cmask, {cmask}, [&] { glColorMask((cmask & 4) != 0, (cmask & 2) != 0, (cmask & 1) != 0, (cmask & 8) != 0); });
        bool ztest = zs && s.rs[kRsZEnable], zwrite = ztest && s.simple[kMZWriteEnable >> 2];
        gs_.Cap(GL_DEPTH_TEST, 1, ztest);
        if (ztest) {
            uint32_t zf = s.simple[kMZFunc >> 2];
            gs_.Set(gs_.zfunc, {zf}, [&] { glDepthFunc(zf); });
        }
        gs_.Set(gs_.zmask, {zwrite ? 1u : 0u}, [&] { glDepthMask(zwrite ? GL_TRUE : GL_FALSE); });
        if (zs && (zwrite || s.rs[kRsStencilEnable])) zs->written = true;
        if (zs && s.rs[kRsStencilEnable]) {
            gs_.Cap(GL_STENCIL_TEST, 2, true);
            uint32_t sref = s.simple[kMStencilRef >> 2] & 0xFF, smask = s.simple[kMStencilMask >> 2] & 0xFF;
            uint32_t sf = s.simple[kMStencilFunc >> 2], f0 = s.rs[kRsStencilFail], f1 = s.simple[kMStencilZFail >> 2], f2 = s.simple[kMStencilPass >> 2];
            uint32_t wm = s.simple[kMStencilWriteMask >> 2] & 0xFF;
            gs_.Set(gs_.stencil, {sf, sref, smask, f0, f1, f2, wm}, [&] {
                glStencilFunc(sf, int(sref), smask);
                glStencilOp(f0, f1, f2);
                glStencilMask(wm);
            });
        } else {
            gs_.Cap(GL_STENCIL_TEST, 2, false);
            gs_.Set(gs_.stencil, {0, 0, 0, 0, 0, 0, 0}, [&] { glStencilMask(0); });
        }
        if (s.simple[kMBlendEnable >> 2]) {
            gs_.Cap(GL_BLEND, 0, true);
            uint32_t bc = s.simple[kMBlendColor >> 2], sb = s.simple[kMSrcBlend >> 2], db = s.simple[kMDestBlend >> 2];
            uint32_t op = s.simple[kMBlendOp >> 2];
            op = op == 0x800A || op == 0x800B || op == 0x8007 || op == 0x8008 ? op : GL_FUNC_ADD;
            gs_.Set(gs_.blend, {bc, sb, db, op}, [&] {
                glBlendColor(float((bc >> 16) & 255) / 255.f, float((bc >> 8) & 255) / 255.f, float(bc & 255) / 255.f, float(bc >> 24) / 255.f);
                glBlendFunc(sb, db);
                glBlendEquation(op);
            });
        } else {
            gs_.Cap(GL_BLEND, 0, false);
        }

        if (cur_program_ != prog->id) {
            glUseProgram(prog->id);
            cur_program_ = prog->id;
        }
        {
            // the draw's pixel stage constants, sent when they differ from what this program last had
            float u[84];
            u[0] = 2.f / float(rt->w), u[1] = 2.f / float(rt->h), u[2] = 2.f / s.zscale, u[3] = 0.f;
            std::memcpy(u + 4, &pk_.pc[0][0], 72 * 4);
            std::memcpy(u + 76, pk_.tfactor, 16);
            std::memcpy(u + 80, pk_.fog_color, 16);
            float at[4] = {float(s.simple[kMAlphaFunc >> 2]), float(s.simple[kMAlphaRef >> 2] & 0xFF), 0, 0};
            if (prog->ucache.size() != 88 || std::memcmp(prog->ucache.data(), u, sizeof u) || std::memcmp(prog->ucache.data() + 84, at, 16)) {
                prog->ucache.assign(u, u + 84);
                prog->ucache.insert(prog->ucache.end(), at, at + 4);
                glUniform4fv(prog->surf, 1, u);
                glUniform4fv(prog->pc, 18, u + 4);
                glUniform4fv(prog->tf, 1, u + 76);
                glUniform4fv(prog->fogc, 1, u + 80);
                glUniform4fv(prog->at, 1, at);
            }
        }
        if (prog->bm >= 0) {
            float bm[16] = {}, bl[16] = {};
            auto f = [](uint32_t b) {
                float v;
                std::memcpy(&v, &b, 4);
                return v;
            };
            for (int i = 0; i < 4; i++) {
                const auto& ts = s.tss[size_t(i)];
                bm[i * 4] = f(ts[kTssBumpEnvMat00]);
                bm[i * 4 + 1] = f(ts[kTssBumpEnvMat01]);
                bm[i * 4 + 2] = f(ts[kTssBumpEnvMat10]);
                bm[i * 4 + 3] = f(ts[kTssBumpEnvMat11]);
                bl[i * 4] = f(ts[kTssBumpEnvLScale]);
                bl[i * 4 + 1] = f(ts[kTssBumpEnvLOffset]);
            }
            glUniform4fv(prog->bm, 4, bm);
            if (prog->bl >= 0) glUniform4fv(prog->bl, 4, bl);
        }

        if (prog->tsz >= 0) {
            // texel coordinates (linear textures) divide by the console's size, whatever size the GPU image is
            float tsz[16] = {};
            for (int i = 0; i < 4; i++) {
                tsz[i * 4] = float(std::max(pk_.stage[i].w, 1u));
                tsz[i * 4 + 1] = float(std::max(pk_.stage[i].h, 1u));
            }
            if (prog->tszc.size() != 16 || std::memcmp(prog->tszc.data(), tsz, sizeof tsz)) {
                prog->tszc.assign(tsz, tsz + 16);
                glUniform4fv(prog->tsz, 4, tsz);
            }
        }
        if (pk_.raw) {
            // the program's constants, sent from the first to the last one that changed since this program last drew
            const float* cv = &s.vsc[0][0];
            if (prog->cache.size() != 768) {
                prog->cache.assign(cv, cv + 768);
                glUniform4fv(prog->c, 192, cv);
            } else {
                // the first and the last register that changed: 16 registers (256 bytes) a compare, from each end, so
                // an unchanged run is passed by a vectorised compare rather than a register at a time
                int lo = -1, hi = -1;
                const float* old = prog->cache.data();
                for (int b = 0; b < 192; b += 16)
                    if (std::memcmp(old + b * 4, cv + b * 4, 256)) {
                        for (int i = b; i < b + 16; i++)
                            if (std::memcmp(old + i * 4, cv + i * 4, 16)) { lo = i; break; }
                        break;
                    }
                if (lo >= 0)
                    for (int b = 176; b >= lo - 15; b -= 16)
                        if (std::memcmp(old + b * 4, cv + b * 4, 256)) {
                            for (int i = b + 15; i >= b; i--)
                                if (std::memcmp(old + i * 4, cv + i * 4, 16)) { hi = i; break; }
                            break;
                        }
                if (lo >= 0) {
                    std::memcpy(&prog->cache[size_t(lo) * 4], cv + lo * 4, size_t(hi - lo + 1) * 16);
                    glUniform4fv(prog->c_at[size_t(lo)], hi - lo + 1, cv + lo * 4);
                }
            }
            float vsu[8] = {pk_.fogp[0], pk_.fogp[1], pk_.fogp[2], pk_.fogp[3], pk_.specular ? 1.f : 0.f, 0.f, 0.f, 0.f};
            if (prog->vsc.size() != 8 || std::memcmp(prog->vsc.data(), vsu, sizeof vsu)) {
                prog->vsc.assign(vsu, vsu + 8);
                glUniform4fv(prog->fogp, 1, vsu);
                glUniform4fv(prog->vsmisc, 1, vsu + 4);
            }
        }
        if (pk_.raw) UploadRaw(pk_);
        else Upload(pk_);
        std::shared_ptr<Counted> c;
        if (counted) {
            c = NewCounted();
            c->area = CoveredPixels(pk_, left, top, right, bottom);
            glBeginQuery(GL_ANY_SAMPLES_PASSED, c->query);
        }
        const void* ix_at = reinterpret_cast<const void*>(idx_off_);
        if (!pk_.raw) glDrawElements(GL_TRIANGLES, GLsizei(pk_.indices.size()), index_type_, ix_at);
        else if (draw_arrays_) glDrawArrays(draw_mode_, 0, GLsizei(pk_.raw_count));
        else if (base_vertex_) glDrawElementsBaseVertex(draw_mode_, GLsizei(pk_.raw_count), index_type_, ix_at, base_vertex_);
        else glDrawElements(draw_mode_, GLsizei(pk_.raw_count), index_type_, ix_at);
        if (counted) {
            glEndQuery(GL_ANY_SAMPLES_PASSED);
            counted_.push_back(std::move(c));
        }
        if (cmask) {
            rt->dirty = true;
            rt->gen++;
            rt->last_draw = draws_seen_;
            rt->frame_draws++;
        }
    }

    void Copy(const Surface& src, const Surface& dst, uint32_t n, uint32_t rects, uint32_t points, float, float) override {
        Ensure();
        Dirty();
        uint32_t lo, hi, w, h;
        if (src.header && reference::HeaderRange(src.header, lo, hi, w, h)) SyncRange(lo, hi, "copy source");
        if (dst.header && reference::HeaderRange(dst.header, lo, hi, w, h)) {
            SyncRange(lo, hi, "copy destination");
            reference::Copy(src, dst, n, rects, points);
            // the destination is now newer than a target over it
            for (auto& t : targets_)
                if (t->data < hi && lo < t->end) {
                    UploadGuest(*t);
                }
            return;
        }
        reference::Copy(src, dst, n, rects, points);
    }

    void Present(const Surface& front) override {
        Ensure();
        if (!power_registered_) {
            power_registered_ = true;
            power::RegisterThread(power::kRender);  // the render thread, for Android's performance hints
        }
        Dirty();
        {
            const float kx = std::clamp(platform::RenderScaleX().load(), 0.5f, 6.f), ky = std::clamp(platform::RenderScaleY().load(), 0.5f, 6.f);
            if (std::fabs(kx - kx_) > 0.001f || std::fabs(ky - ky_) > 0.001f) Rescale(kx, ky);
        }
        auto& f = platform::GetFrame();
        Target* t = nullptr;
        uint32_t lo = 0, hi = 0, w = 0, h = 0;
        if (SurfaceRange(front, lo, hi, w, h))
            for (auto& x : targets_)
                if (x->data == lo && x->w == w && x->h == h) t = x.get();
        // a title that double buffers by itself (NFS Most Wanted moves its two frame buffers' memory around on its own)
        // finishes the frame in either one while the device's front surface keeps pointing at one of them: the frame
        // shown is the full size target it drew last this frame, else every other frame showed the unfinished one
        // a buffer cleared for the next frame is the last one touched, so the frame shown is the full size target that
        // took the most draws this frame: the one with the world and the HUD over it
        if (pending_clear_) ApplyPendingClear(FindTarget(pending_clear_->s.rt, false));  // nothing was drawn: as asked
        if (t) frame_buffers_.insert(t->data);
        if (t) {
            Target* most = t;
            for (auto& x : targets_)
                if (x->w == t->w && x->h == t->h && x->frame_draws > most->frame_draws) most = x.get();
            t = most;
        }
        present_draws_ = draws_seen_;
        shown_ = t;
        for (auto& x : targets_) {
            if (x->frame_draws > 64) x->last_world = frame_;
            x->frame_draws = 0;
        }
        if (t) {
            CheckGuest(*t);
            // read into a pixel buffer now and hand over the one read a present ago: the read does not stall the
            // title's thread on the GPU finishing the frame
            const int cur = pbo_next_, prev = cur ^ 1;
            const size_t bytes = size_t(t->sw) * t->sh * 4;
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[cur]);
            if (pbo_cap_[cur] < bytes) {
                glBufferData(GL_PIXEL_PACK_BUFFER, GLsizeiptr(bytes), nullptr, GL_STREAM_READ);
                pbo_cap_[cur] = bytes;
            }
            const GLuint shown_fbo = t->read_fbo;
            const bool direct = PresentDirect(shown_fbo, t->sw, t->sh, t->tex);
            // through memory only without the direct way
            if (direct) goto readback_done;
            glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[cur]);
            glBindFramebuffer(GL_FRAMEBUFFER, shown_fbo);
            glPixelStorei(GL_PACK_ALIGNMENT, 4);
            glReadPixels(0, 0, int(t->sw), int(t->sh), GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            pbo_w_[cur] = t->sw;
            pbo_h_[cur] = t->sh;
            pbo_full_[cur] = true;
            glFlush();
            if (pbo_full_[prev]) {
                const size_t pbytes = size_t(pbo_w_[prev]) * pbo_h_[prev] * 4;
                glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_[prev]);
                if (const void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, GLsizeiptr(pbytes), GL_MAP_READ_BIT)) {
                    // one copy into a buffer of our own, swapped in: the window draws it without blending, so the
                    // alpha is left as the GPU wrote it (forcing it opaque pixel by pixel cost milliseconds a frame)
                    const size_t npx = size_t(pbo_w_[prev]) * pbo_h_[prev];
                    staging_.resize(npx);
                    std::memcpy(staging_.data(), p, npx * 4);
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
                    std::lock_guard lock(f.lock);
                    f.w = pbo_w_[prev];
                    f.h = pbo_h_[prev];
                    f.rgba.swap(staging_);
                    f.serial++;
                }
                pbo_full_[prev] = false;
            }
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
            pbo_next_ = prev;
        readback_done:
            glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        } else if (SurfaceRange(front, lo, hi, w, h)) {
            // a buffer the GPU never drew: whatever the title put in memory
            pbo_full_[0] = pbo_full_[1] = false;
            SyncRange(lo, hi, "present");
            std::lock_guard lock(f.lock);
            if (reference::ReadRgba(front, f.w, f.h, f.rgba)) f.serial++;
        }
        RingNextFrame();
        frame_++;
        RingFrameStarts();
        VertexPool().next = 0;
        IndexPool().next = 0;
        DeliverReports(false);
        // the images over guest memory are kept; textures the GPU has not used for a while are dropped
        if (frame_ % 60 == 0) Evict();
        reference::BeginFrame();
        if (ring_overflow_) {
            ring_overflow_ = 0;
        }
    }

    void Sync(uint32_t data_va, uint32_t size) override {
        Ensure();
        Dirty();
        SyncRange(data_va, data_va + std::max(size, 1u), "lock");
    }
    void SyncLate(uint32_t data_va) override {
        // a NOOVERWRITE lock may still read what the GPU drew there
        Ensure();
        Dirty();
        SyncRange(data_va, data_va + 1, "lock no-overwrite");
    }
    void Flush() override {
        if (owner_ == std::this_thread::get_id()) glFlush();
    }
    // the console's gamma ramp (D3DGAMMARAMP: 256 red, 256 green, 256 blue), which its GPU applies to the picture on
    // the way to the screen: kept here and applied as the frame is handed to the window (PresentDirect)
    void SetGamma(const uint8_t* ramp) override {
        if (!ramp) return;
        bool identity = true;
        for (int c = 0; c < 3 && identity; c++)
            for (int i = 0; i < 256; i++)
                if (std::abs(int(ramp[c * 256 + i]) - i) > 1) { identity = false; break; }
        if (std::memcmp(gamma_, ramp, 768) != 0) {
            std::memcpy(gamma_, ramp, 768);
            gamma_dirty_ = true;
        }
        gamma_identity_ = identity;
    }

    // visibility tests: draws while counting get a query each, a report sums the pixels of those that passed
    void ClearReport() override { counted_.clear(); }
    void GetReport(uint32_t report) override {
        if (!report) return;
        reports_.push_back({report, counted_, frame_});
    }

private:
    void InitObjects() {
        glGenVertexArrays(1, &vao_);
        BindVao(vao_);
        glGenBuffers(2, pbo_);
    }

    // which texture coordinate sets the pixel stage reads
    static uint32_t UsedTc(const reference::DrawPacket& pk) {
        uint32_t m = 0;
        const PixelKey& k = pk.key;
        if (k.programmable) {
            for (int i = 0; i < 4; i++) {
                uint32_t mode = (k.tex_modes >> (i * 5)) & 0x1F;
                if (mode) m |= 1u << i;
                if (mode >= 9) m = 0xF;
            }
        } else {
            for (int i = 0; i < 4; i++) {
                if (k.ff[i][0] <= 1) break;
                if (pk.info[i].kind) m |= 1u << i;
            }
        }
        return m;
    }

    // one buffer per draw, reused three frames later: a buffer earlier draws of the frame still read is never
    // written. Writing into one (glBufferSubData, or even an unsynchronized map) makes mobile drivers copy the whole
    // buffer or wait for the GPU, which cost a millisecond a draw and held races under 1 fps.
    struct StreamPool {
        std::vector<GLuint> bufs;
        std::vector<size_t> caps;
        size_t next = 0;
    };
    static constexpr int kPoolFrames = 3;
    GLuint Stream(StreamPool& p, GLenum target, const void* data, size_t bytes) {
        if (p.next == p.bufs.size()) {
            GLuint b;
            glGenBuffers(1, &b);
            p.bufs.push_back(b);
            p.caps.push_back(0);
        }
        const size_t i = p.next++;
        glBindBuffer(target, p.bufs[i]);
        if (p.caps[i] < bytes) {
            size_t cap = 4096;
            while (cap < bytes) cap *= 2;
            glBufferData(target, GLsizeiptr(cap), nullptr, GL_STREAM_DRAW);
            p.caps[i] = cap;
        }
        glBufferSubData(target, 0, GLsizeiptr(bytes), data);
        return p.bufs[i];
    }
    StreamPool& VertexPool() { return vpool_[frame_ % kPoolFrames]; }

    // the vertices and indices of a frame go into one big buffer each, mapped once for good (GL_EXT_buffer_storage)
    // and written with plain copies: a draw is a memcpy and an offset instead of a buffer update the driver has to
    // track. Each buffer has a slice per frame in flight; a slice is written again only after the GPU passed the
    // fence left at the end of its frame. A draw that does not fit its slice falls back to Stream
    static constexpr size_t kRingSlice[2] = {24u << 20, 4u << 20};  // vertices, indices: bytes per frame
    bool RingInit() {
        if (ring_tried_) return ring_ok_;
        ring_tried_ = true;
        const char* ext = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
        auto storage = reinterpret_cast<void (*)(GLenum, GLsizeiptr, const void*, GLbitfield)>(eglGetProcAddress("glBufferStorageEXT"));
        if (!ext || !std::strstr(ext, "GL_EXT_buffer_storage") || !storage) {
            return false;
        }
        for (int w = 0; w < 2; w++) {
            const GLenum target = w ? GL_ELEMENT_ARRAY_BUFFER : GL_ARRAY_BUFFER;
            const size_t size = kRingSlice[w] * kPoolFrames;
            BindVao(0);
            glGenBuffers(1, &ring_buf_[w]);
            glBindBuffer(target, ring_buf_[w]);
            const GLbitfield flags = GL_MAP_WRITE_BIT | 0x0040 /* PERSISTENT */ | 0x0080 /* COHERENT */;
            storage(target, GLsizeiptr(size), nullptr, flags);
            ring_map_[w] = static_cast<uint8_t*>(glMapBufferRange(target, 0, GLsizeiptr(size), flags));
            if (!ring_map_[w]) {
                return false;
            }
        }
        ring_ok_ = true;
        return true;
    }
    // data into the frame's slice of ring w: the buffer bound to target and the offset, or Stream's buffer at 0
    size_t Put(int w, GLenum target, const void* data, size_t bytes, GLuint* buf = nullptr) {
        if (ring_ok_) {
            const size_t at = (ring_head_[w] + 15) & ~size_t(15);
            if (at + bytes <= kRingSlice[w]) {
                const size_t off = kRingSlice[w] * size_t(frame_ % kPoolFrames) + at;
                std::memcpy(ring_map_[w] + off, data, bytes);
                ring_head_[w] = at + bytes;
                // the array buffer binding is global, the element one the vertex array's: both only when it changed
                GLuint& bound = w ? ring_bound_elem_ : ring_bound_array_;
                if (bound != ring_buf_[w]) {
                    glBindBuffer(target, ring_buf_[w]);
                    bound = ring_buf_[w];
                }
                if (buf) *buf = ring_buf_[w];
                return off;
            }
            ring_overflow_++;
        }
        const GLuint b = Stream(w ? IndexPool() : VertexPool(), target, data, bytes);
        (w ? ring_bound_elem_ : ring_bound_array_) = b;
        if (buf) *buf = b;
        return 0;
    }
    // the frame's end: a fence after its slices; the next frame's slices free once the GPU passed theirs
    void RingNextFrame() {
        if (!ring_ok_) return;
        const int done = int(frame_ % kPoolFrames);
        if (ring_fence_[done]) glDeleteSync(ring_fence_[done]);
        ring_fence_[done] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
    void RingFrameStarts() {
        if (!ring_tried_) RingInit();  // between frames, where no vertex array is half set up
        if (!ring_ok_) return;
        const int slot = int(frame_ % kPoolFrames);
        if (ring_fence_[slot]) {
            glClientWaitSync(ring_fence_[slot], GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000ull);
            glDeleteSync(ring_fence_[slot]);
            ring_fence_[slot] = nullptr;
        }
        ring_head_[0] = ring_head_[1] = 0;
    }


    StreamPool& IndexPool() { return ipool_[frame_ % kPoolFrames]; }

    // how GL reads an input register the input assembler left in the guest's own format
    struct AttrFormat {
        GLint size = 4;
        GLenum type = GL_FLOAT;
        GLboolean norm = GL_FALSE;
        bool integer = false;
        uint32_t variant = 0;  // see glshader::VertexProgramSource
    };
    static AttrFormat FormatOf(uint8_t dxgi, uint32_t decode) {
        const bool w1 = decode == kDecodeW1 || decode == kDecodeIntW1;
        switch (dxgi) {
            case DXGI_FORMAT_R32_FLOAT: return {1};
            case DXGI_FORMAT_R32G32_FLOAT: return {2};
            case DXGI_FORMAT_R32G32B32_FLOAT: return {3, GL_FLOAT, GL_FALSE, false, decode == kDecodeFloat2H ? 3u : 0u};
            case DXGI_FORMAT_R32G32B32A32_FLOAT: return {4};
            case DXGI_FORMAT_B8G8R8A8_UNORM: return {4, GL_UNSIGNED_BYTE, GL_TRUE, false, 1};
            case DXGI_FORMAT_R16_SNORM: return {1, GL_SHORT, GL_TRUE};
            case DXGI_FORMAT_R16G16_SNORM: return {2, GL_SHORT, GL_TRUE};
            case DXGI_FORMAT_R16G16B16A16_SNORM: return {w1 ? 3 : 4, GL_SHORT, GL_TRUE};
            case DXGI_FORMAT_R8_UNORM: return {1, GL_UNSIGNED_BYTE, GL_TRUE};
            case DXGI_FORMAT_R8G8_UNORM: return {2, GL_UNSIGNED_BYTE, GL_TRUE};
            case DXGI_FORMAT_R8G8B8A8_UNORM: return {w1 ? 3 : 4, GL_UNSIGNED_BYTE, GL_TRUE};
            case DXGI_FORMAT_R16_SINT: return {1, GL_SHORT, GL_FALSE};
            case DXGI_FORMAT_R16G16_SINT: return {2, GL_SHORT, GL_FALSE};
            case DXGI_FORMAT_R16G16B16A16_SINT: return {w1 ? 3 : 4, GL_SHORT, GL_FALSE};
            case DXGI_FORMAT_R32_UINT: return {1, GL_UNSIGNED_INT, GL_FALSE, true, 2};
            default: return {};
        }
    }
    // the shader variant bits of a raw draw's registers
    static uint32_t Variants(const reference::DrawPacket& pk) {
        uint32_t v = 0;
        for (int r = 0; r < 16; r++)
            if (pk.regs >> r & 1) v |= FormatOf(pk.as->elements[size_t(r)].format, uint32_t(pk.as->decode >> (r * 4)) & 15).variant << (r * 2);
        return v;
    }

    // a raw draw: the guest's vertex buffers as they are (or the input assembler's float copy where it made one),
    // each register read in its own format, the guest's 16 bit indices as they are where it drew with them
    void UploadRaw(const reference::DrawPacket& pk) {
        const gpu::Assembled& a = *pk.as;
        BindVao(vao_);
        GLuint bufs[16] = {};
        size_t offs[16] = {};
        for (uint32_t b = 0; b < a.buffer_count && b < 16; b++) {
            const void* src = a.buffers[b].src;
            const size_t bytes = std::max<size_t>(a.buffers[b].bytes, 16);
            offs[b] = Put(0, GL_ARRAY_BUFFER, src, bytes, &bufs[b]);
        }
        // separate formats and buffer bindings (GL ES 3.1): a register's format is sent only when it changes, and each
        // buffer takes one call a draw, where a pointer per register cost the driver most of a draw's setup
        uint32_t used_buffers = 0;
        for (int r = 0; r < 16; r++) {
            if (pk.regs >> r & 1) {
                const gpu::VertexElement& e = a.elements[size_t(r)];
                AttrFormat f = FormatOf(e.format, uint32_t(a.decode >> (r * 4)) & 15);
                Attr(r, true);
                const uint64_t key = uint64_t(f.size) | uint64_t(f.type) << 4 | uint64_t(f.norm) << 20 | uint64_t(f.integer) << 21 |
                                     uint64_t(e.buffer & 15) << 22 | uint64_t(e.offset) << 26 | 1ull << 63;
                if (raw_fmt_[r] != key) {
                    raw_fmt_[r] = key;
                    if (f.integer) glVertexAttribIFormat(GLuint(r), f.size, f.type, GLuint(e.offset));
                    else glVertexAttribFormat(GLuint(r), f.size, f.type, f.norm, GLuint(e.offset));
                    glVertexAttribBinding(GLuint(r), GLuint(e.buffer));
                }
                used_buffers |= 1u << (e.buffer & 15);
            } else {
                Attr(r, false);
                if (r == 3) AttrDefault(r, 1.f, 1.f, 1.f, 1.f);
                else AttrDefault(r, 0.f, 0.f, 0.f, 1.f);
            }
        }
        for (uint32_t b = 0; b < 16; b++) {
            if (!(used_buffers >> b & 1)) continue;
            const GLsizei stride = GLsizei(a.buffers[b].stride);
            if (raw_vb_buf_[b] != bufs[b] || raw_vb_off_[b] != offs[b] || raw_vb_stride_[b] != stride) {
                raw_vb_buf_[b] = bufs[b];
                raw_vb_off_[b] = offs[b];
                raw_vb_stride_[b] = stride;
                glBindVertexBuffer(b, bufs[b], GLintptr(offs[b]), stride);
            }
        }
        draw_mode_ = a.strip ? GL_TRIANGLE_STRIP : GL_TRIANGLES;
        base_vertex_ = 0;
        draw_arrays_ = false;
        if (a.index_count && a.index16) {
            idx_off_ = Put(1, GL_ELEMENT_ARRAY_BUFFER, a.index_src, size_t(a.index_count) * 2);
            index_type_ = GL_UNSIGNED_SHORT;
            base_vertex_ = a.base_vertex;
        } else if (a.index_count) {
            const uint32_t* ix = static_cast<const uint32_t*>(a.index_src);
            if (a.vertex_count <= 0xFFFF) {
                index16_.resize(a.index_count);
                for (uint32_t i = 0; i < a.index_count; i++) index16_[i] = uint16_t(ix[i]);
                idx_off_ = Put(1, GL_ELEMENT_ARRAY_BUFFER, index16_.data(), index16_.size() * 2);
                index_type_ = GL_UNSIGNED_SHORT;
            } else {
                idx_off_ = Put(1, GL_ELEMENT_ARRAY_BUFFER, ix, size_t(a.index_count) * 4);
                index_type_ = GL_UNSIGNED_INT;
            }
        } else {
            draw_arrays_ = true;
        }
    }

    // the draw's vertices packed to what it reads: position, two colours as bytes, fog, the texture coordinates in use
    void Upload(const reference::DrawPacket& pk) {
        for (int r = 8; r < 16; r++) Attr(r, false);
        uint32_t mask = UsedTc(pk);
        int n = __builtin_popcount(mask);
        const size_t stride = 28 + 16 * size_t(n), bytes = stride * pk.verts.size();
        pack_.resize(bytes);
        uint8_t* out = pack_.data();
        auto b = [](float v) { return uint8_t(v <= 0 ? 0 : v >= 1 ? 255 : int(v * 255.f + 0.5f)); };
        for (const reference::GlVertex& v : pk.verts) {
            std::memcpy(out, v.clip, 16);
            for (int j = 0; j < 4; j++) {
                out[16 + j] = b(v.v0[j]);
                out[20 + j] = b(v.v1[j]);
            }
            std::memcpy(out + 24, v.fog, 4);
            size_t at = 28;
            for (int i = 0; i < 4; i++)
                if (mask >> i & 1) {
                    std::memcpy(out + at, v.tc[i], 16);
                    at += 16;
                }
            out += stride;
        }
        BindVao(vao_);
        RawFormatsLost();  // the pointers below set registers 0 to 7 and their bindings
        const uintptr_t base = Put(0, GL_ARRAY_BUFFER, pack_.data(), bytes);
        // indices, 16 bit when they fit
        if (pk.verts.size() <= 0xFFFF) {
            index16_.resize(pk.indices.size());
            for (size_t i = 0; i < pk.indices.size(); i++) index16_[i] = uint16_t(pk.indices[i]);
            idx_off_ = Put(1, GL_ELEMENT_ARRAY_BUFFER, index16_.data(), index16_.size() * 2);
            index_type_ = GL_UNSIGNED_SHORT;
        } else {
            idx_off_ = Put(1, GL_ELEMENT_ARRAY_BUFFER, pk.indices.data(), pk.indices.size() * 4);
            index_type_ = GL_UNSIGNED_INT;
        }
        auto ptr = [&](size_t o) { return reinterpret_cast<const void*>(base + o); };
        Attr(0, true);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, GLsizei(stride), ptr(0));
        Attr(1, true);
        glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, GLsizei(stride), ptr(16));
        Attr(2, true);
        glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, GLsizei(stride), ptr(20));
        Attr(3, true);
        glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, GLsizei(stride), ptr(24));
        size_t at = 28;
        for (int i = 0; i < 4; i++) {
            if (mask >> i & 1) {
                Attr(4 + i, true);
                glVertexAttribPointer(GLuint(4 + i), 4, GL_FLOAT, GL_FALSE, GLsizei(stride), ptr(at));
                at += 16;
            } else {
                Attr(4 + i, false);
                AttrDefault(4 + i, 0.f, 0.f, 0.f, 1.f);
            }
        }
    }

    // visibility tests

    std::shared_ptr<Counted> NewCounted() {
        GLuint q;
        if (!free_queries_.empty()) {
            q = free_queries_.back();
            free_queries_.pop_back();
        } else {
            glGenQueries(1, &q);
        }
        auto* c = new Counted;
        c->query = q;
        // the query goes back to the pool once no report holds it
        return std::shared_ptr<Counted>(c, [this](Counted* p) {
            free_queries_.push_back(p->query);
            delete p;
        });
    }
    // the screen pixels the draw's triangles cover inside the scissor: ES 3 queries only say whether any passed
    static double CoveredPixels(const reference::DrawPacket& pk, int left, int top, int right, int bottom) {
        double total = 0;
        const double limit = double(std::max(right - left, 0)) * double(std::max(bottom - top, 0));
        for (size_t i = 0; i + 2 < pk.indices.size(); i += 3) {
            const reference::GlVertex* v[3] = {&pk.verts[pk.indices[i]], &pk.verts[pk.indices[i + 1]], &pk.verts[pk.indices[i + 2]]};
            float x[3], y[3];
            bool ok = true;
            for (int j = 0; j < 3; j++) {
                float w = v[j]->clip[3];
                if (!(w > 0)) { ok = false; break; }
                x[j] = std::clamp(v[j]->clip[0] / w, float(left), float(right));
                y[j] = std::clamp(v[j]->clip[1] / w, float(top), float(bottom));
            }
            if (!ok) return limit;  // crosses the eye plane: count it as covering the view
            total += std::fabs(double((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))) * 0.5;
        }
        return std::min(total, limit);
    }
    // reports whose queries have finished, written as the console's GPU writes them: time, count, then 0 for done;
    // ones two presents old are waited for
    void DeliverReports(bool force) {
        constexpr uint32_t kTime = 0, kTimeHigh = 4, kCount = 8, kStatus = 12;
        while (!reports_.empty()) {
            Report& r = reports_.front();
            bool wait = force || frame_ >= r.frame + 2;
            for (auto& c : r.counted) {
                if (c->passed >= 0) continue;
                GLuint avail = 0;
                if (!wait) glGetQueryObjectuiv(c->query, GL_QUERY_RESULT_AVAILABLE, &avail);
                if (!wait && !avail) return;
                GLuint any = 0;
                glGetQueryObjectuiv(c->query, GL_QUERY_RESULT, &any);
                c->passed = any ? 1 : 0;
            }
            double pixels = 0;
            for (auto& c : r.counted)
                if (c->passed > 0) pixels += c->area;
            kernel::W32(r.va + kTime, uint32_t(frame_));
            kernel::W32(r.va + kTimeHigh, 0);
            kernel::W32(r.va + kCount, uint32_t(std::lround(pixels)));
            std::atomic_thread_fence(std::memory_order_release);
            kernel::W32(r.va + kStatus, 0);
            reports_.pop_front();
        }
    }




    // GL state was changed outside Draw's cached setters
    void Dirty() {
        gs_.Invalidate();
        ring_bound_array_ = ring_bound_elem_ = 0;
        TexturesMoved();
        for (int& a : attr_on_) a = -1;
        for (auto& d : attr_def_) d = {-1.f, -1.f, -1.f, -1.f};
        RawFormatsLost();
        bound_vao_ = ~0u;
    }
    // a texture was deleted or a unit bound outside Draw: units 0 to 3 are bound again
    void TexturesMoved() {
        for (auto& u : unit_tex_) u = ~0ull;
        for (auto& u : unit_smp_) u = ~0u;
    }
    void BindVao(GLuint v) {
        if (bound_vao_ == v) return;
        bound_vao_ = v;
        glBindVertexArray(v);
    }
    void RawFormatsLost() {
        for (auto& k : raw_fmt_) k = 0;
        for (auto& b : raw_vb_buf_) b = ~0u;
    }
    void Attr(int r, bool on) {
        if (attr_on_[r] == int(on)) return;
        attr_on_[r] = int(on);
        if (on) glEnableVertexAttribArray(GLuint(r));
        else glDisableVertexAttribArray(GLuint(r));
    }
    void AttrDefault(int r, float x, float y, float z, float w) {
        auto& d = attr_def_[r];
        if (d[0] == x && d[1] == y && d[2] == z && d[3] == w) return;
        d = {x, y, z, w};
        glVertexAttrib4f(GLuint(r), x, y, z, w);
    }

    // the context belongs to the thread that draws
    void Ensure() {
        auto me = std::this_thread::get_id();
        if (owner_ == me) return;
        eglMakeCurrent(dpy_, surf_, surf_, ctx_);
        owner_ = me;
    }

    void Scissor(int x0, int y0, int x1, int y1, const Target& t) {
        x0 = int(SX(std::max(x0, 0)));
        y0 = int(SY(std::max(y0, 0)));
        x1 = int(SX(std::min(x1, int(t.w))));
        y1 = int(SY(std::min(y1, int(t.h))));
        glScissor(x0, y0, std::max(x1 - x0, 0), std::max(y1 - y0, 0));
    }

    // the guest bytes of a render target or depth surface
    static bool SurfaceRange(const Surface& s, uint32_t& lo, uint32_t& hi, uint32_t& w, uint32_t& h) {
        TextureDesc d;
        if (!s.header || !DescribeSurface(s, d)) return false;
        lo = d.data;
        hi = d.data + TotalSize(d);
        w = d.width;
        h = d.height;
        return true;
    }
    // a draw into one of the frame buffers that has taken next to nothing this frame, while the other holds this
    // frame's world: NFS Most Wanted draws its nitro streaks and tyre smoke (16 fans) over the previous frame in the
    // other buffer, which here is black (its clear lands there), so they go over this frame's world instead, which the
    // post processing reads (TargetTexture)
    struct PendingClear {
        State s;
        uint32_t flags, color;
        float z;
        uint32_t stencil;
    };
    // the held clear (Clear), done on target: the state it was asked in, with target's memory as the render target
    void ApplyPendingClear(Target* target) {
        std::unique_ptr<PendingClear> pc = std::move(pending_clear_);
        if (!target) return;
        pc->s.rt = target->surf;
        applying_clear_ = true;
        Clear(pc->s, 0, nullptr, pc->flags, pc->color, pc->z, pc->stencil);
        applying_clear_ = false;
    }
    // a frame buffer that held a whole frame lately but not the frame before: the game stopped drawing into it to
    // keep that frame (the race result photo). Never held one, or long ago: a buffer the redirect is for
    bool Frozen(const Target& t) const { return t.last_world && t.last_world + 1 < frame_ && frame_ - t.last_world < 1800; }
    Target* FrameBufferFor(Target* rt) {
        if (!rt || !frame_buffers_.count(rt->data) || rt->frame_draws > 64) return rt;
        // a frozen buffer (the race result photo) takes its smoke itself, see TargetTexture
        if (Frozen(*rt)) return rt;
        for (auto& x : targets_)
            if (x.get() != rt && x->w == rt->w && x->h == rt->h && frame_buffers_.count(x->data) && x->frame_draws > 64 &&
                x->frame_draws > rt->frame_draws * 4) {
                return x.get();
            }
        return rt;
    }
    Target* FindTarget(const Surface& s, bool create) {
        uint32_t lo, hi, w, h;
        if (!SurfaceRange(s, lo, hi, w, h)) return nullptr;
        for (auto& t : targets_)
            if (t->data == lo && t->w == w && t->h == h) {
                t->surf = s;
                return t.get();
            }
        if (!create) return nullptr;
        auto t = std::make_unique<Target>();
        Dirty();
        t->data = lo;
        t->end = hi;
        t->w = w;
        t->h = h;
        t->sw = SX(w);
        t->sh = SY(h);
        t->surf = s;
        TextureDesc d;
        if (Describe(s.header, d)) t->format = d.format;
        glGenTextures(1, &t->tex);
        glBindTexture(GL_TEXTURE_2D, t->tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, GLsizei(t->sw), GLsizei(t->sh));
        glGenFramebuffers(1, &t->read_fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, t->read_fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, t->tex, 0);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(1, 1, 1, 1);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        targets_.push_back(std::move(t));
        Target* made = targets_.back().get();
        // whatever the title put in the memory before it drew there is where the target starts
        UploadGuest(*made);
        made->checked = frame_;
        return made;
    }

    // the depth buffer as the console's memory holds it, read through a colour texture: each texel is the D24S8 word
    // z << 8 | stencil, so A is the depth's top byte, R the middle, G the low and B the stencil (taken as 0)
    GLuint DepthAsColour(DepthBuf& d) {
        if (!d.written && d.colour) return d.colour;
        d.written = false;
        Dirty();
        cur_program_ = 0;
        if (!depth_prog_) {
            static const char* vs = "#version 300 es\n"
                                    "out vec2 v_uv;\n"
                                    "void main() { vec2 p = vec2(float((gl_VertexID & 1) << 2) - 1.0, float((gl_VertexID & 2) << 1) - 1.0);\n"
                                    "  v_uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }\n";
            // highp int: a fragment shader's default is mediump, which the Adreno runs as 16 bits, so q >> 16 wrapped
            // round to the low byte, and every effect the title builds on its depth (the cutscenes' depth of field,
            // the nitro blur, the finish smoke) read noise
            static const char* fs = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
                                    "uniform highp sampler2D u_depth;\nin vec2 v_uv;\nlayout(location = 0) out vec4 o;\n"
                                    "void main() {\n"
                                    "  float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;\n"
                                    // the GL depth back to the console's 0..1 (gl_backend keeps it scaled by kDepthScale)
                                    "  float z = clamp(((2.0 * d - 1.0) / 0.99995 + 1.0) * 0.5, 0.0, 1.0);\n"
                                    "  uint q = uint(z * 16777215.0 + 0.5);\n"
                                    "  o = vec4(float((q >> 8) & 255u), float(q & 255u), 0.0, float((q >> 16) & 255u)) / 255.0;\n"
                                    "}\n";
            GLuint v = Compile(GL_VERTEX_SHADER, vs), f = Compile(GL_FRAGMENT_SHADER, fs);
            depth_prog_ = glCreateProgram();
            glAttachShader(depth_prog_, v);
            glAttachShader(depth_prog_, f);
            glLinkProgram(depth_prog_);
            glDeleteShader(v);
            glDeleteShader(f);
            glUseProgram(depth_prog_);
            glUniform1i(glGetUniformLocation(depth_prog_, "u_depth"), 7);
            glGenVertexArrays(1, &depth_vao_);
        }
        if (!d.colour) {
            glGenTextures(1, &d.colour);
            glBindTexture(GL_TEXTURE_2D, d.colour);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, GLsizei(d.sw), GLsizei(d.sh));
            glGenFramebuffers(1, &d.fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, d.fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, d.colour, 0);
        }
        GLint unit = 0;
        glGetIntegerv(GL_ACTIVE_TEXTURE, &unit);
        glBindFramebuffer(GL_FRAMEBUFFER, d.fbo);
        glViewport(0, 0, GLsizei(d.sw), GLsizei(d.sh));
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_SCISSOR_TEST);
        glDisable(GL_CULL_FACE);
        glColorMask(1, 1, 1, 1);
        glActiveTexture(GL_TEXTURE7);
        glBindTexture(GL_TEXTURE_2D, d.tex);
        glBindSampler(7, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glUseProgram(depth_prog_);
        BindVao(depth_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GLenum(unit));
        return d.colour;
    }

    bool CubeOfTargets(const TextureDesc& d) const {
        for (uint32_t f = 0; f < 6; f++)
            if (!IsTarget(d.data + f * FaceSize(d), d.format, d.width, d.height)) return false;
        return true;
    }
    // a cube whose faces the title renders as six targets: a GL cube with each face copied from its target on the GPU,
    // again only when that target was drawn since
    struct Cube {
        GLuint tex = 0;
        uint32_t size = 0;
        uint64_t gen[6] = {};
    };
    GLuint CubeTexture(const reference::GlStage& stg) {
        Cube& c = cubes_[stg.target];
        const uint32_t size = std::max(SX(stg.w), SY(stg.w));  // square, at the finer of the two scales
        if (!c.tex || c.size != size) {
            if (c.tex) {
                glDeleteTextures(1, &c.tex);
                TexturesMoved();
            }
            c = Cube{};
            glGenTextures(1, &c.tex);
            glBindTexture(GL_TEXTURE_CUBE_MAP, c.tex);
            glTexStorage2D(GL_TEXTURE_CUBE_MAP, 1, GL_RGBA8, GLsizei(size), GLsizei(size));
            c.size = size;
        }
        glBindTexture(GL_TEXTURE_CUBE_MAP, c.tex);
        for (uint32_t f = 0; f < 6; f++) {
            Target* t = TargetAt(stg.target + f * stg.face_stride, stg.w, stg.h);
            if (!t) continue;
            CheckGuest(*t);
            if (c.gen[f] == t->gen) continue;
            c.gen[f] = t->gen;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, t->read_fbo);
            if (t->sw == size && t->sh == size) {
                glCopyTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, 0, 0, 0, 0, GLsizei(t->sw), GLsizei(t->sh));
                continue;
            }
            // the render scale differs across and down (a wide screen), so the face's target is not square while a
            // cube's faces must be: the face is scaled into the square (a copy wider than the face fails and left
            // the cube empty, the cars' reflections black)
            GLint draw_fbo = 0;
            glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
            const GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
            if (!cube_fbo_) glGenFramebuffers(1, &cube_fbo_);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, cube_fbo_);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, c.tex, 0);
            if (scissor) glDisable(GL_SCISSOR_TEST);
            glBlitFramebuffer(0, 0, GLint(t->sw), GLint(t->sh), 0, 0, GLint(size), GLint(size), GL_COLOR_BUFFER_BIT, GL_LINEAR);
            if (scissor) glEnable(GL_SCISSOR_TEST);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, GLuint(draw_fbo));
        }
        glBindTexture(GL_TEXTURE_CUBE_MAP, c.tex);
        return c.tex;
    }

    Target* TargetAt(uint32_t data, uint32_t w, uint32_t h) {
        for (auto& t : targets_)
            if (t->data == data && t->w == w && t->h == h) return t.get();
        return nullptr;
    }

    // a stage that samples a render target: the target's own texture, a copy of it when the draw writes it, with
    // alpha read as 1 for formats without one
    GLuint TargetTexture(const reference::GlStage& stg, Target* draw_rt) {
        Target* t = TargetAt(stg.target, stg.w, stg.h);
        // NFS Most Wanted points its frame buffers' memory around by itself, and its post processing then reads the
        // frame buffer it did not draw the world into this frame (the race showed black). A read of one of the frame
        // buffers that took next to nothing this frame, while the other holds the frame, goes to the one with the frame.
        // (keeping the other buffer's last frame instead, with the frame's first clear moved, let the game's blur feed
        // back on itself into white streaks: the game means this frame. <shots>/nofbredirect turns it off)
        // Only while the two alternate (t held a whole frame the frame before): a buffer left alone for longer is a
        // frame the game froze on purpose, the photo of the race result screen, read as it is
        if (t && frame_buffers_.count(t->data) && !Frozen(*t)) {
            for (auto& x : targets_)
                if (x.get() != t && x->w == t->w && x->h == t->h && frame_buffers_.count(x->data) && x->frame_draws > 64 &&
                    x->frame_draws > t->frame_draws * 4) {
                    t = x.get();
                    break;
                }
        }
        if (!t) {
            for (auto& d : depths_)
                if (d->data == stg.target && d->w == stg.w && d->h == stg.h) return DepthAsColour(*d);
            return 0;
        }
        CheckGuest(*t);
        GLint want = OpaqueFormat(stg.format) ? GL_ONE : GL_ALPHA;
        if (t != draw_rt) {
            glBindTexture(GL_TEXTURE_2D, t->tex);
            if (t->swizzle_a != want) {
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, want);
                t->swizzle_a = want;
            }
            return t->tex;
        }
        // the console reads the pixels it is about to overwrite: give the draw a copy
        if (!t->copy) {
            glGenTextures(1, &t->copy);
            glBindTexture(GL_TEXTURE_2D, t->copy);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, GLsizei(t->sw), GLsizei(t->sh));
        }
        glBindFramebuffer(GL_READ_FRAMEBUFFER, t->read_fbo);
        glBindTexture(GL_TEXTURE_2D, t->copy);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, GLsizei(t->sw), GLsizei(t->sh));
        if (t->copy_swizzle_a != want) {
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, want);
            t->copy_swizzle_a = want;
        }
        return t->copy;
    }

    DepthBuf* FindDepth(const Surface& s, const Target& rt) {
        uint32_t lo, hi, w, h;
        if (!SurfaceRange(s, lo, hi, w, h) || w < rt.w || h < rt.h) return nullptr;
        // a depth surface larger than the colour target (NFS Most Wanted's lobby: 1280x480 depth, 640x480 colour) is
        // used over the target's area, as the console does; refusing it left the car drawn with no depth test at all
        w = rt.w;
        h = rt.h;
        for (auto& d : depths_)
            if (d->data == lo && d->w == w && d->h == h) return d.get();
        auto d = std::make_unique<DepthBuf>();
        d->data = lo;
        d->w = w;
        d->h = h;
        d->sw = SX(w);
        d->sh = SY(h);
        glGenTextures(1, &d->tex);
        glBindTexture(GL_TEXTURE_2D, d->tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_DEPTH24_STENCIL8, GLsizei(d->sw), GLsizei(d->sh));
        depths_.push_back(std::move(d));
        return depths_.back().get();
    }

    GLuint Fbo(Target& rt, DepthBuf* zs) {
        if (!zs) return rt.read_fbo;
        uint64_t key = uint64_t(rt.data) << 32 | zs->data;
        auto it = fbos_.find(key);
        if (it != fbos_.end()) return it->second;
        Dirty();
        GLuint f;
        glGenFramebuffers(1, &f);
        glBindFramebuffer(GL_FRAMEBUFFER, f);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt.tex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, zs->tex, 0);
        if (!zs->initialised) {
            // a new depth buffer starts at the far plane, not at whatever the driver left in it
            zs->initialised = true;
            glDisable(GL_SCISSOR_TEST);
            glDepthMask(GL_TRUE);
            glStencilMask(0xFF);
            glClearDepthf(1.f);
            glClearStencil(0);
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
        }
        fbos_[key] = f;
        return f;
    }

    // guest memory in [lo, hi) is about to be read: bring dirty targets over it back
    void SyncRange(uint32_t lo, uint32_t hi, const char* why) {
        for (auto& t : targets_)
            if (t->dirty && t->data < hi && lo < t->end) Readback(*t);
    }
    void Readback(Target& t) {
        Dirty();
        readback_.resize(size_t(t.w) * t.h);
        // memory holds the console's size: a scaled image is filtered down to it on the GPU first
        glBindFramebuffer(GL_FRAMEBUFFER, t.sw == t.w && t.sh == t.h ? t.read_fbo : Shrunk(t));
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, int(t.w), int(t.h), GL_RGBA, GL_UNSIGNED_BYTE, readback_.data());
        reference::WriteRgba(t.surf, readback_.data(), t.w, t.h);
        t.dirty = false;
        // this write is the renderer's own: a target sharing that memory (the same surface seen at another size) must
        // not take it for the title's and reload itself from it, which showed a black or stale frame now and then
        for (auto& o : targets_)
            if (o->data < t.end && t.data < o->end) Remember(*o);
    }
    // console sized images for moving a scaled target to and from guest memory
    struct Scratch {
        GLuint tex = 0, fbo = 0;
    };
    Scratch& ScratchFor(uint32_t w, uint32_t h) {
        Scratch& sc = scratch_[uint64_t(w) << 32 | h];
        if (!sc.tex) {
            glGenTextures(1, &sc.tex);
            glBindTexture(GL_TEXTURE_2D, sc.tex);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, GLsizei(w), GLsizei(h));
            glGenFramebuffers(1, &sc.fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, sc.fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, sc.tex, 0);
        }
        return sc;
    }
    // a GPU wait for an Android native fence file (it is taken over); a native fence file for what is sent so far
    void WaitFd(int fd) {
        if (fd < 0) return;
        const EGLint attr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, fd, EGL_NONE};
        EGLSyncKHR sync = p_create_sync_(dpy_, EGL_SYNC_NATIVE_FENCE_ANDROID, attr);
        if (sync == EGL_NO_SYNC_KHR) { close(fd); return; }
        p_wait_sync_(dpy_, sync, 0);
        p_destroy_sync_(dpy_, sync);
    }
    int FenceFd() {
        const EGLint attr[] = {EGL_SYNC_NATIVE_FENCE_FD_ANDROID, EGL_NO_NATIVE_FENCE_FD_ANDROID, EGL_NONE};
        EGLSyncKHR sync = p_create_sync_(dpy_, EGL_SYNC_NATIVE_FENCE_ANDROID, attr);
        if (sync == EGL_NO_SYNC_KHR) return -1;
        glFlush();
        const int fd = p_dup_fd_(dpy_, sync);
        p_destroy_sync_(dpy_, sync);
        return fd;
    }
    // the frame to the window's buffer it is not showing (platform::PresentChain); false when that way is not open
    // the frame through the gamma ramp: each pixel's red, green and blue looked up in it, into the bound draw framebuffer
    void GammaPass(GLuint from_tex, uint32_t sw, uint32_t sh) {
        if (!gamma_prog_) {
            static const char* vs = "#version 300 es\nvoid main() { vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));\n"
                                    "gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0); }\n";
            static const char* fs = "#version 300 es\nprecision highp float;\nuniform highp sampler2D u_src;\nuniform highp sampler2D u_lut;\n"
                                    "layout(location = 0) out vec4 o;\nvoid main() {\n"
                                    "  vec4 c = texelFetch(u_src, ivec2(gl_FragCoord.xy), 0);\n"
                                    "  ivec3 i = ivec3(clamp(c.rgb * 255.0 + 0.5, 0.0, 255.0));\n"
                                    "  o = vec4(texelFetch(u_lut, ivec2(i.r, 0), 0).r, texelFetch(u_lut, ivec2(i.g, 0), 0).g,\n"
                                    "           texelFetch(u_lut, ivec2(i.b, 0), 0).b, c.a);\n}\n";
            GLuint v = Compile(GL_VERTEX_SHADER, vs), f = Compile(GL_FRAGMENT_SHADER, fs);
            gamma_prog_ = glCreateProgram();
            glAttachShader(gamma_prog_, v);
            glAttachShader(gamma_prog_, f);
            glLinkProgram(gamma_prog_);
            glDeleteShader(v);
            glDeleteShader(f);
            glUseProgram(gamma_prog_);
            glUniform1i(glGetUniformLocation(gamma_prog_, "u_src"), 6);
            glUniform1i(glGetUniformLocation(gamma_prog_, "u_lut"), 7);
            glGenTextures(1, &gamma_lut_);
            glBindTexture(GL_TEXTURE_2D, gamma_lut_);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 256, 1);
            glGenVertexArrays(1, &gamma_vao_);
            gamma_dirty_ = true;
        }
        glActiveTexture(GL_TEXTURE7);
        glBindSampler(7, 0);
        glBindTexture(GL_TEXTURE_2D, gamma_lut_);
        if (gamma_dirty_) {
            uint8_t lut[256 * 4];
            for (int i = 0; i < 256; i++) {
                lut[i * 4] = gamma_[i];
                lut[i * 4 + 1] = gamma_[256 + i];
                lut[i * 4 + 2] = gamma_[512 + i];
                lut[i * 4 + 3] = 255;
            }
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 256, 1, GL_RGBA, GL_UNSIGNED_BYTE, lut);
            gamma_dirty_ = false;
        }
        glActiveTexture(GL_TEXTURE6);
        glBindSampler(6, 0);
        glBindTexture(GL_TEXTURE_2D, from_tex);
        glViewport(0, 0, GLsizei(sw), GLsizei(sh));
        glDisable(GL_BLEND);
        glDisable(GL_DEPTH_TEST);
        glDisable(GL_STENCIL_TEST);
        glDisable(GL_CULL_FACE);
        glUseProgram(gamma_prog_);
        cur_program_ = gamma_prog_;
        BindVao(gamma_vao_);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        BindVao(0);
        glActiveTexture(GL_TEXTURE7);
        TexturesMoved();
    }
    bool PresentDirect(GLuint from, uint32_t sw, uint32_t sh, GLuint from_tex = 0) {
        auto& pc = platform::GetPresentChain();
        if (!pc.active) return false;
        std::unique_lock lock(pc.lock);
        pc.want_w = sw;
        pc.want_h = sh;
        if (pc.w != sw || pc.h != sh || !pc.buffer[0]) return true;  // the window makes buffers of this size first
        const int k = pc.Free(lock);
        if (k < 0 || !pc.buffer[k]) return true;
        WaitFd(pc.read_fd[k]);
        pc.read_fd[k] = -1;
        if (pc.write_fd[k] >= 0) {
            close(pc.write_fd[k]);
            pc.write_fd[k] = -1;
        }
        Dirty();
        if (!present_fbo_[k]) {
            glGenFramebuffers(1, &present_fbo_[k]);
            glGenTextures(1, &present_tex_[k]);
        }
        if (present_img_[k] != pc.buffer[k]) {
            // a new buffer: our own image of it, and a fresh texture over that (the old one may still be in use)
            if (present_egl_img_[k]) p_destroy_image_(dpy_, EGLImageKHR(present_egl_img_[k]));
            const EGLint attr[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
            present_egl_img_[k] = p_create_image_(dpy_, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID,
                                                  p_client_buffer_(static_cast<const AHardwareBuffer*>(pc.buffer[k])), attr);
            glDeleteTextures(1, &present_tex_[k]);
            glGenTextures(1, &present_tex_[k]);
            glBindTexture(GL_TEXTURE_2D, present_tex_[k]);
            p_image_target_(GL_TEXTURE_2D, GLeglImageOES(present_egl_img_[k]));
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, present_fbo_[k]);
            glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, present_tex_[k], 0);
            present_img_[k] = pc.buffer[k];
            TexturesMoved();
        }
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, present_fbo_[k]);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(1, 1, 1, 1);
        if (gamma_identity_ || !from_tex) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, from);
            glBlitFramebuffer(0, 0, int(sw), int(sh), 0, 0, int(sw), int(sh), GL_COLOR_BUFFER_BIT, GL_NEAREST);
        } else {
            GammaPass(from_tex, sw, sh);
        }
        pc.write_fd[k] = FenceFd();
        pc.Push(k);
        pc.serial++;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        Dirty();
        return true;
    }
    // the resolution setting changed while the game runs: every target redrawn at the new scale from what it holds
    // (filtered), the depth buffers made again (the next frame clears them) and the framebuffers that held them dropped
    void Rescale(float kx, float ky) {
        Dirty();
        for (auto& kv : fbos_) glDeleteFramebuffers(1, &kv.second);
        fbos_.clear();
        for (auto& t : targets_) {
            const uint32_t sw = std::max(1u, uint32_t(std::lround(t->w * kx))), sh = std::max(1u, uint32_t(std::lround(t->h * ky)));
            GLuint tex = 0, fbo = 0;
            glGenTextures(1, &tex);
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, GLsizei(sw), GLsizei(sh));
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_A, t->swizzle_a);
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
            glBindFramebuffer(GL_READ_FRAMEBUFFER, t->read_fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
            glDisable(GL_SCISSOR_TEST);
            glColorMask(1, 1, 1, 1);
            glBlitFramebuffer(0, 0, int(t->sw), int(t->sh), 0, 0, int(sw), int(sh), GL_COLOR_BUFFER_BIT, GL_LINEAR);
            glDeleteFramebuffers(1, &t->read_fbo);
            glDeleteTextures(1, &t->tex);
            if (t->copy) glDeleteTextures(1, &t->copy);
            t->copy = 0;
            t->copy_swizzle_a = GL_ALPHA;
            t->tex = tex;
            t->read_fbo = fbo;
            t->sw = sw;
            t->sh = sh;
        }
        for (auto& d : depths_) {
            if (d->tex) glDeleteTextures(1, &d->tex);
            if (d->colour) glDeleteTextures(1, &d->colour);
            if (d->fbo) glDeleteFramebuffers(1, &d->fbo);
        }
        depths_.clear();
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        kx_ = kx;
        ky_ = ky;
        TexturesMoved();
        Dirty();
    }
    // the target filtered down to the console's size, in a scratch framebuffer bound for reading
    GLuint Shrunk(Target& t) {
        Dirty();
        Scratch& sc = ScratchFor(t.w, t.h);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, t.read_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, sc.fbo);
        glDisable(GL_SCISSOR_TEST);
        glColorMask(1, 1, 1, 1);
        glBlitFramebuffer(0, 0, int(t.sw), int(t.sh), 0, 0, int(t.w), int(t.h), GL_COLOR_BUFFER_BIT, GL_LINEAR);
        return sc.fbo;
    }

    // one stripe of the target's guest memory: the 8 bytes at offset 8k of every 64 (a sampled hash alone missed
    // a few changed bytes for good)
    static uint64_t GuestStripe(const Target& t, uint32_t k) {
        const uint8_t* p = static_cast<const uint8_t*>(kernel::HostPtr(t.data));
        const size_t n = t.end - t.data;
        uint64_t h = 1469598103934665603ull ^ k;
        for (size_t i = size_t(k) * 8; i + 8 <= n; i += 64) {
            uint64_t w;
            std::memcpy(&w, p + i, 8);
            h = (h ^ w) * 1099511628211ull;
        }
        return h ^ n;
    }
    static void Remember(Target& t) {
        for (uint32_t k = 0; k < 8; k++) t.known[k] = GuestStripe(t, k);
    }
    // one stripe a frame, as much reading as the old sampled hash: a change anywhere shows within 8 frames
    static bool GuestChanged(Target& t) {
        const uint32_t k = t.next_stripe++ & 7;
        return GuestStripe(t, k) != t.known[k];
    }
    // the title wrote the target's memory with the CPU (a movie frame decoded into it, say), with or without a lock:
    // that is newer than the GPU's copy, which takes it. Checked once a frame per target.
    void CheckGuest(Target& t) {
        // every other frame: a stripe reads every 64th word of the whole target, a few MB a frame over all of them
        if (t.checked == frame_ || (t.checked && frame_ - t.checked < 2)) return;
        t.checked = frame_;
        if (!GuestChanged(t)) return;
        UploadGuest(t);
    }
    // the title wrote the surface's memory itself: the target takes it
    void UploadGuest(Target& t) {
        Dirty();
        uint32_t w, h;
        Remember(t);
        if (!reference::ReadRgba(t.surf, w, h, upload_) || w != t.w || h != t.h) return;
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (kx_ == 1.f && ky_ == 1.f) {
            glBindTexture(GL_TEXTURE_2D, t.tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(w), int(h), GL_RGBA, GL_UNSIGNED_BYTE, upload_.data());
        } else {
            // the console sized pixels into a scratch image, stretched up into the target
            Scratch& sc = ScratchFor(w, h);
            glBindTexture(GL_TEXTURE_2D, sc.tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, int(w), int(h), GL_RGBA, GL_UNSIGNED_BYTE, upload_.data());
            glBindFramebuffer(GL_READ_FRAMEBUFFER, sc.fbo);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t.read_fbo);
            glDisable(GL_SCISSOR_TEST);
            glColorMask(1, 1, 1, 1);
            glBlitFramebuffer(0, 0, int(w), int(h), 0, 0, int(t.sw), int(t.sh), GL_COLOR_BUFFER_BIT, GL_NEAREST);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
        t.dirty = false;
        t.gen++;
    }

    // textures: one GL texture per guest texture (address, format, size), updated in place when its bytes change,
    // so a texture the title rewrites every frame does not leave a new GL texture behind each time

    static int Levels(uint32_t w, uint32_t h, bool mips) {
        int levels = 1;
        if (mips)
            for (uint32_t m = std::max(w, h); m > 1; m >>= 1) levels++;
        return levels;
    }
    void Fill(const reference::GlStage& s, bool cube, int levels) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        if (cube) {
            for (int f = 0; f < 6; f++)
                if (s.faces[f]) glTexSubImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, 0, 0, 0, GLsizei(s.w), GLsizei(s.h), GL_RGBA, GL_UNSIGNED_BYTE, s.faces[f]);
        } else {
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, GLsizei(s.w), GLsizei(s.h), GL_RGBA, GL_UNSIGNED_BYTE, s.rgba);
        }
        if (levels > 1) glGenerateMipmap(cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D);
    }
    GLuint TextureFor(const reference::GlStage& s, bool rect, bool cube) {
        const uint64_t key = s.slot ? s.slot : (s.id | 1ull << 63);
        const GLenum target = cube ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;
        const int levels = Levels(s.w, s.h, !rect && s.w > 1 && s.h > 1);
        auto it = textures_.find(key);
        if (it != textures_.end()) {
            GlTex& t = it->second;
            t.used = frame_;
            if (t.image == s.id) return t.id;
            if (t.w == s.w && t.h == s.h && t.cube == cube && t.levels == levels) {
                glBindTexture(target, t.id);
                Fill(s, cube, levels);
                t.image = s.id;
                return t.id;
            }
            glDeleteTextures(1, &t.id);
            TexturesMoved();
            textures_.erase(it);
        }
        GlTex t;
        glGenTextures(1, &t.id);
        glBindTexture(target, t.id);
        glTexStorage2D(target, levels, GL_RGBA8, GLsizei(s.w), GLsizei(s.h));
        Fill(s, cube, levels);
        t.image = s.id;
        t.w = s.w;
        t.h = s.h;
        t.cube = cube;
        t.levels = levels;
        t.used = frame_;
        textures_[key] = t;
        return t.id;
    }

    GLuint SamplerFor(const reference::GlStage& s, bool rect, bool cube) {
        auto wrap = [&](uint32_t m) { return cube ? GL_CLAMP_TO_EDGE : m == 1 ? GL_REPEAT : m == 2 ? GL_MIRRORED_REPEAT : GL_CLAMP_TO_EDGE; };
        // the phone's texture filtering setting: trilinear blends mip levels, anisotropic keeps roads sharp far ahead
        const int filter = 0;  // the game's own filtering
        uint32_t key = s.address_u | s.address_v << 4 | std::min(s.mag, 3u) << 8 | std::min(s.min, 3u) << 10 | std::min(s.mip, 3u) << 12 |
                       (rect ? 1u << 14 : 0) | (cube ? 1u << 15 : 0) | uint32_t(filter) << 16;
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        GLuint id;
        glGenSamplers(1, &id);
        glSamplerParameteri(id, GL_TEXTURE_WRAP_S, wrap(s.address_u));
        glSamplerParameteri(id, GL_TEXTURE_WRAP_T, wrap(s.address_v));
        glSamplerParameteri(id, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
        glSamplerParameteri(id, GL_TEXTURE_MAG_FILTER, s.mag >= 2 ? GL_LINEAR : GL_NEAREST);
        GLint min;
        bool lin = s.min >= 2;
        if (rect || s.mip == 0) min = lin ? GL_LINEAR : GL_NEAREST;
        else if (s.mip == 1 && !(filter >= 1 && lin)) min = lin ? GL_LINEAR_MIPMAP_NEAREST : GL_NEAREST_MIPMAP_NEAREST;
        else min = lin ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST_MIPMAP_LINEAR;
        glSamplerParameteri(id, GL_TEXTURE_MIN_FILTER, min);
        if (filter >= 2 && lin && !rect && s.mip != 0) {
            static const float max_aniso = [] {
                float m = 0;
                const char* ext = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
                if (ext && std::strstr(ext, "GL_EXT_texture_filter_anisotropic")) glGetFloatv(0x84FF, &m);  // MAX_TEXTURE_MAX_ANISOTROPY
                return m;
            }();
            if (max_aniso > 1) glSamplerParameterf(id, 0x84FE, std::min(max_aniso, filter == 2 ? 4.f : 16.f));  // TEXTURE_MAX_ANISOTROPY
        }
        samplers_[key] = id;
        return id;
    }

    void Evict() {
        for (auto it = textures_.begin(); it != textures_.end();) {
            if (frame_ - it->second.used > 120) {
                glDeleteTextures(1, &it->second.id);
                TexturesMoved();
                it = textures_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // programs

    Prog* GetProgram(const reference::DrawPacket& pk, bool atest) {
        const uint32_t variants = pk.raw ? Variants(pk) : 0;
        const uint64_t key = glshader::CacheHash(pk.key, pk.info, atest) ^
                             (pk.raw ? (pk.program_hash ^ uint64_t(variants) << 1) * 0x9E3779B97F4A7C15ull + 1 : 0);
        auto it = programs_.find(key);
        if (it != programs_.end()) return &it->second;
        Prog p;
        std::string fs = glshader::FragmentSource(pk.key, pk.info, atest);
        std::string vsrc = pk.raw ? glshader::VertexProgramSource(*pk.program, variants) : std::string(glshader::kVertexSource);
        // linked programs are kept on disk by their sources and the driver: compiling one mid race stalled a frame,
        // and now that happens once ever instead of once a session
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](const std::string& t) {
            for (unsigned char ch : t) h = (h ^ ch) * 1099511628211ull;
        };
        mix(vsrc);
        mix(fs);
        mix(driver_);
        char name[32];
        std::snprintf(name, sizeof name, "%016llx.bin", (unsigned long long)h);
        const std::string file = cache_dir_.empty() ? std::string() : cache_dir_ + "/" + name;
        p.id = glCreateProgram();
        bool linked = false;
        if (!file.empty())
            if (FILE* fp = std::fopen(file.c_str(), "rb")) {
                std::vector<uint8_t> blob;
                uint8_t buf[65536];
                for (size_t n; (n = std::fread(buf, 1, sizeof buf, fp)) > 0;) blob.insert(blob.end(), buf, buf + n);
                std::fclose(fp);
                if (blob.size() > 4) {
                    GLenum fmt;
                    std::memcpy(&fmt, blob.data(), 4);
                    glProgramBinary(p.id, fmt, blob.data() + 4, GLsizei(blob.size() - 4));
                    GLint ok = 0;
                    glGetProgramiv(p.id, GL_LINK_STATUS, &ok);
                    linked = ok != 0;
                    if (linked) cached_programs_++;
                }
            }
        if (!linked) {
            GLuint v = Compile(GL_VERTEX_SHADER, vsrc.c_str()), f = Compile(GL_FRAGMENT_SHADER, fs.c_str());
            if (v && f) {
                glAttachShader(p.id, v);
                glAttachShader(p.id, f);
                glProgramParameteri(p.id, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE);
                glLinkProgram(p.id);
                GLint ok = 0;
                glGetProgramiv(p.id, GL_LINK_STATUS, &ok);
                if (!ok) {
                    char log[1024] = {};
                    glGetProgramInfoLog(p.id, sizeof log, nullptr, log);
                } else {
                    linked = true;
                    compiled_programs_++;
                    GLint len = 0;
                    glGetProgramiv(p.id, GL_PROGRAM_BINARY_LENGTH, &len);
                    if (len > 0 && !file.empty()) {
                        std::vector<uint8_t> blob(size_t(len) + 4);
                        GLenum fmt = 0;
                        GLsizei got = 0;
                        glGetProgramBinary(p.id, len, &got, &fmt, blob.data() + 4);
                        std::memcpy(blob.data(), &fmt, 4);
                        if (got > 0)
                            if (FILE* fp = std::fopen(file.c_str(), "wb")) {
                                std::fwrite(blob.data(), 1, size_t(got) + 4, fp);
                                std::fclose(fp);
                            }
                    }
                }
            }
            if (v) glDeleteShader(v);
            if (f) glDeleteShader(f);
        }
        if (linked) {
            p.ok = true;
            p.pc = glGetUniformLocation(p.id, "u_pc");
            p.tf = glGetUniformLocation(p.id, "u_tf");
            p.fogc = glGetUniformLocation(p.id, "u_fogc");
            p.at = glGetUniformLocation(p.id, "u_at");
            p.surf = glGetUniformLocation(p.id, "u_surf");
            p.bm = glGetUniformLocation(p.id, "u_bm");
            p.bl = glGetUniformLocation(p.id, "u_bl");
            p.tsz = glGetUniformLocation(p.id, "u_tsz");
            p.c = glGetUniformLocation(p.id, "u_c");
            p.fogp = glGetUniformLocation(p.id, "u_fogp");
            p.vsmisc = glGetUniformLocation(p.id, "u_vsmisc");
            if (p.c >= 0) {
                p.c_at.resize(192);
                for (int i = 0; i < 192; i++) p.c_at[size_t(i)] = glGetUniformLocation(p.id, ("u_c[" + std::to_string(i) + "]").c_str());
            }
            glUseProgram(p.id);
            cur_program_ = p.id;
            for (int i = 0; i < 4; i++) {
                GLint loc = glGetUniformLocation(p.id, (std::string("u_s") + char('0' + i)).c_str());
                if (loc >= 0) glUniform1i(loc, i);
            }
        } else {
        }
        return &(programs_[key] = p);
    }
    GLuint Compile(GLenum type, const char* src) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char log[2048] = {};
            glGetShaderInfoLog(s, sizeof log, nullptr, log);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }

    EGLDisplay dpy_ = EGL_NO_DISPLAY;
    EGLSurface surf_ = EGL_NO_SURFACE;
    EGLContext ctx_ = EGL_NO_CONTEXT;
    std::thread::id owner_;
    GLuint vao_ = 0, bound_vao_ = ~0u;
    uint64_t raw_fmt_[16] = {};          // each register's format and binding as last set (0: unknown)
    GLuint raw_vb_buf_[16] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
    size_t raw_vb_off_[16] = {};
    GLsizei raw_vb_stride_[16] = {};
    std::vector<std::unique_ptr<Target>> targets_;
    std::vector<std::unique_ptr<DepthBuf>> depths_;
    uint32_t last_clear_color_ = 0, last_clear_flags_ = 0, last_clear_n_ = 0, last_fog_color_ = 0;
    float last_clear_z_ = 0;
    std::unordered_map<uint64_t, GLuint> fbos_;
    std::unordered_map<uint64_t, GlTex> textures_;
    std::unordered_map<uint32_t, GLuint> samplers_;
    std::unordered_map<uint64_t, Prog> programs_;
    reference::DrawPacket pk_;
    unsigned long long frame_ = 0;
    std::vector<uint8_t> pack_;
    std::vector<uint32_t> readback_, upload_, staging_;
    StreamPool vpool_[kPoolFrames], ipool_[kPoolFrames];
    GLuint ring_buf_[2] = {};
    uint8_t* ring_map_[2] = {};
    size_t ring_head_[2] = {};
    GLsync ring_fence_[kPoolFrames] = {};
    bool ring_tried_ = false, ring_ok_ = false;
    uint64_t ring_overflow_ = 0;
    size_t idx_off_ = 0;
    uint8_t gamma_[768] = {};
    bool gamma_identity_ = true, gamma_dirty_ = false;
    GLuint gamma_prog_ = 0, gamma_lut_ = 0, gamma_vao_ = 0;
    GLuint ring_bound_array_ = 0, ring_bound_elem_ = 0;  // what Put last bound (0: unknown, bind again)
    std::vector<uint16_t> index16_;
    GLenum index_type_ = GL_UNSIGNED_INT;
    GLenum draw_mode_ = GL_TRIANGLES;
    GLint base_vertex_ = 0;
    bool draw_arrays_ = false;
    GLuint pbo_[2] = {};
    size_t pbo_cap_[2] = {};
    uint32_t pbo_w_[2] = {}, pbo_h_[2] = {};
    bool pbo_full_[2] = {};
    int pbo_next_ = 0;
    std::vector<std::shared_ptr<Counted>> counted_;
    std::deque<Report> reports_;
    std::vector<GLuint> free_queries_;
    std::unordered_map<uint32_t, Cube> cubes_;
    GLuint cube_fbo_ = 0;
    bool power_registered_ = false;  // a cube face as a draw target, to scale a face's target into it
    std::unordered_map<uint64_t, Scratch> scratch_;
    // render scale, across and down: the console's pixels times these are the GPU's (the screen's own pixels at
    // "native", whatever the picture's shape)
    float kx_ = 1, ky_ = 1;
    GLuint present_fbo_[platform::PresentChain::kBuffers] = {}, present_tex_[platform::PresentChain::kBuffers] = {};
    void* present_img_[platform::PresentChain::kBuffers] = {};
    void* present_egl_img_[3] = {};
    PFNEGLDUPNATIVEFENCEFDANDROIDPROC p_dup_fd_ = nullptr;
    PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC p_client_buffer_ = nullptr;
    PFNEGLCREATEIMAGEKHRPROC p_create_image_ = nullptr;
    PFNEGLDESTROYIMAGEKHRPROC p_destroy_image_ = nullptr;
    PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_image_target_ = nullptr;
    PFNEGLCREATESYNCKHRPROC p_create_sync_ = nullptr;
    PFNEGLWAITSYNCKHRPROC p_wait_sync_ = nullptr;
    PFNEGLDESTROYSYNCKHRPROC p_destroy_sync_ = nullptr;
    uint32_t SX(int x) const { return uint32_t(std::lround(float(x) * kx_)); }
    uint32_t SY(int y) const { return uint32_t(std::lround(float(y) * ky_)); }
    GlState gs_;
    uint64_t unit_tex_[4] = {~0ull, ~0ull, ~0ull, ~0ull};
    GLuint unit_smp_[4] = {~0u, ~0u, ~0u, ~0u};
    int attr_on_[16] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
    std::array<float, 4> attr_def_[16] = {};
    GLuint cur_program_ = 0;
    std::string driver_, cache_dir_;
    uint64_t cached_programs_ = 0, compiled_programs_ = 0;
    std::set<uint32_t> frame_buffers_;  // the memory of every target presented
    std::unique_ptr<PendingClear> pending_clear_;
    Target* shown_ = nullptr;   // the target the last present showed
    bool applying_clear_ = false;
    unsigned long long present_draws_ = 0;
    GLuint depth_prog_ = 0, depth_vao_ = 0;
    unsigned long long draws_seen_ = 0;
};

bool IsTargetThunk(uint32_t data, uint32_t format, uint32_t w, uint32_t h) {
    return g_self && g_self->IsTarget(data, format, w, h);
}

}  // namespace

std::unique_ptr<Backend> CreateGlBackend() {
    auto b = std::make_unique<GlBackend>();
    if (!b->Setup()) return nullptr;
    return b;
}

}  // namespace gpu
