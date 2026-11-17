// GPU renderer on Vulkan 1.3. The same work as gl_backend.cpp, which it replaces: the CPU reference renderer prepares
// each draw (input assembly, texture decoding, the pixel stage's key), the GPU runs the console's vertex programs and
// pixel stages as shaders from gl_shader.cpp (turned into Vulkan GLSL and SPIR-V by vk_glsl.cpp). Render targets live on
// the GPU and go back to guest memory only when the title reads that memory. Frames reach the window as Android
// hardware buffers (platform::PresentChain), with sync files both ways.
//
// Conventions kept from the GL renderer so its shaders and memory layouts stay as they are: the viewport is the target's
// whole image with y as GL has it (rows in memory are the same either way), clip z goes from GL's -w..w to 0..w in the
// vertex shader, and the winding the GL renderer called clockwise is counter clockwise here (framebuffer y runs down).
#include <android/hardware_buffer.h>
#include <unistd.h>
#define VK_USE_PLATFORM_ANDROID_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "frame.h"
#include "gl_shader.h"
#include "gpu/backend.h"
#include "gpu/texture.h"
#include "gpu/vertex.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "power.h"
#include "reference_renderer.h"
#include "vk_glsl.h"
#include <dxgiformat.h>

namespace gpu {
namespace {

constexpr float kDepthScale = 0.99995f;
constexpr VkFormat kColour = VK_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t kFrames = 3;                 // frames in flight
constexpr size_t kRingBytes = 40u << 20;        // a frame's vertices, indices, uniforms and texture uploads
constexpr uint32_t kQueries = 4096;             // visibility tests a frame
constexpr uint32_t kConstBinding = 16;          // the vertex binding of constant inputs (registers a draw leaves out)
constexpr size_t kUniformBytes = 8u << 20;      // a frame's uniform blocks
constexpr uint32_t kUniformRange = 4096;        // the most a draw's block takes (the vertex program's 192 constants and more)

bool TargetColourFormat(uint32_t f) {
    switch (f) {
        case 0x05: case 0x11: case 0x06: case 0x12: case 0x07: case 0x1E: return true;
        default: return false;
    }
}
bool OpaqueFormat(uint32_t f) { return f == 0x05 || f == 0x11 || f == 0x07 || f == 0x1E; }

// GL enums the console's registers hold, as Vulkan's
VkBlendFactor BlendFactor(uint32_t g) {
    switch (g) {
        case 0: return VK_BLEND_FACTOR_ZERO;
        case 1: return VK_BLEND_FACTOR_ONE;
        case 0x300: return VK_BLEND_FACTOR_SRC_COLOR;
        case 0x301: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        case 0x302: return VK_BLEND_FACTOR_SRC_ALPHA;
        case 0x303: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        case 0x304: return VK_BLEND_FACTOR_DST_ALPHA;
        case 0x305: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
        case 0x306: return VK_BLEND_FACTOR_DST_COLOR;
        case 0x307: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
        case 0x308: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
        case 0x8001: return VK_BLEND_FACTOR_CONSTANT_COLOR;
        case 0x8002: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
        case 0x8003: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
        case 0x8004: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
        default: return VK_BLEND_FACTOR_ONE;
    }
}
VkBlendOp BlendOp(uint32_t g) {
    switch (g) {
        case 0x8007: return VK_BLEND_OP_MIN;
        case 0x8008: return VK_BLEND_OP_MAX;
        case 0x800A: return VK_BLEND_OP_SUBTRACT;
        case 0x800B: return VK_BLEND_OP_REVERSE_SUBTRACT;
        default: return VK_BLEND_OP_ADD;
    }
}
VkCompareOp CompareOp(uint32_t g) { return g >= 0x200 && g <= 0x207 ? VkCompareOp(g - 0x200) : VK_COMPARE_OP_ALWAYS; }
VkStencilOp StencilOp(uint32_t g) {
    switch (g) {
        case 0: return VK_STENCIL_OP_ZERO;
        case 0x1E01: return VK_STENCIL_OP_REPLACE;
        case 0x1E02: return VK_STENCIL_OP_INCREMENT_AND_CLAMP;
        case 0x1E03: return VK_STENCIL_OP_DECREMENT_AND_CLAMP;
        case 0x150A: return VK_STENCIL_OP_INVERT;
        case 0x8507: return VK_STENCIL_OP_INCREMENT_AND_WRAP;
        case 0x8508: return VK_STENCIL_OP_DECREMENT_AND_WRAP;
        default: return VK_STENCIL_OP_KEEP;
    }
}

// an image with its memory and views
struct Image {
    VkImage img = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;      // sampled (and attachment) view
    VkImageView view_one = VK_NULL_HANDLE;  // alpha read as 1
    VkImageView depth_view = VK_NULL_HANDLE;  // a depth image's depth aspect alone, for sampling
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkFormat format = kColour;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    uint32_t w = 0, h = 0, layers = 1, levels = 1;
    bool cube = false;
    explicit operator bool() const { return img != VK_NULL_HANDLE; }
};

struct Target {
    uint32_t data = 0, end = 0, w = 0, h = 0, format = 0;
    uint64_t last_world = 0;
    uint32_t sw = 0, sh = 0;
    Surface surf;
    Image img;
    Image copy;  // the target as it was, for a draw that reads the target it draws to
    uint64_t copy_gen = 0;  // gen the copy was made at: a draw reading the target again before it changed reuses it
    uint64_t gen = 1;
    unsigned long long last_draw = 0;
    uint32_t frame_draws = 0;
    uint64_t known[8] = {};
    uint32_t next_stripe = 0;
    unsigned long long checked = ~0ull;
    bool dirty = false;
    bool pend = false;         // a whole clear not done yet: the next pass begins with it (no load of what was there)
    VkClearColorValue pend_value{};
    uint32_t cube = 0;         // a face of the cube over this guest address (the car reflections), else 0
    uint32_t face = 0;
};
struct DepthBuf {
    uint32_t data = 0, w = 0, h = 0, sw = 0, sh = 0;
    Image img;
    Image colour;  // the depth as the console's memory holds it, read as A8R8G8B8
    bool initialised = false;
    bool written = true;
    bool pend = false;         // a whole clear of depth and stencil not done yet, as the target's
    VkClearDepthStencilValue pend_value{1.f, 0};
};
struct Tex {
    Image img;
    uint64_t image = 0;
    uint64_t used = 0;
};
// a program's uniform members as offsets in its stage's block (-1: not in this program)
struct Prog {
    VkShaderModule vs = VK_NULL_HANDLE, fs = VK_NULL_HANDLE;
    uint32_t vs_bytes = 0, fs_bytes = 0;
    int c = -1, surf = -1, fogp = -1, vsmisc = -1;
    uint32_t c_lo = 0, c_hi = 192;  // the vertex constants the program reads (all of them when it indexes by a0)
    int pc = -1, tf = -1, fogc = -1, at = -1, bm = -1, bl = -1, tsz = -1;
    uint32_t samplers = 0;
    uint64_t id = 0;
    bool ok = false;
};
struct Counted {
    uint32_t slot = 0, query = ~0u;  // frame slot and query in its pool; ~0: not counted on the GPU
    uint64_t frame = 0;
    double area = 0;
    int passed = -1;
};
struct Report {
    uint32_t va = 0;
    std::vector<std::shared_ptr<Counted>> counted;
    unsigned long long frame = 0;
};

// a vertex attribute as the pipeline takes it
struct Attr {
    uint32_t location, binding, offset;
    VkFormat format;
};

class VkBackend;
VkBackend* g_self = nullptr;
bool IsTargetThunk(uint32_t data, uint32_t format, uint32_t w, uint32_t h);

class VkBackend final : public Backend {
public:
    ~VkBackend() override {
        if (g_self == this) g_self = nullptr;
        reference::SetTargetTest(nullptr);
        if (dev_) vkDeviceWaitIdle(dev_);
        SavePipelineCache();
    }

    // instance, device and everything frames need; false where this phone lacks what the renderer uses
    bool Setup() {
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        app.pApplicationName = "nfsmw";
        app.apiVersion = VK_API_VERSION_1_3;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &app;
        if (vkCreateInstance(&ici, nullptr, &inst_) != VK_SUCCESS) return Fail("no Vulkan instance");
        uint32_t n = 0;
        vkEnumeratePhysicalDevices(inst_, &n, nullptr);
        if (!n) return Fail("no Vulkan device");
        std::vector<VkPhysicalDevice> pds(n);
        vkEnumeratePhysicalDevices(inst_, &n, pds.data());
        phys_ = pds[0];
        vkGetPhysicalDeviceProperties(phys_, &props_);
        if (props_.apiVersion < VK_API_VERSION_1_3) return Fail("Vulkan 1.3 needed");
        if (props_.limits.maxVertexInputBindings < kConstBinding + 1) return Fail("vertex bindings");
        vkGetPhysicalDeviceMemoryProperties(phys_, &memprops_);
        // a queue that draws and copies
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, nullptr);
        std::vector<VkQueueFamilyProperties> qf(n);
        vkGetPhysicalDeviceQueueFamilyProperties(phys_, &n, qf.data());
        qfam_ = ~0u;
        for (uint32_t i = 0; i < n; i++)
            if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) { qfam_ = i; break; }
        if (qfam_ == ~0u) return Fail("no graphics queue");
        // device extensions: hardware buffers for the window, sync files, push descriptors
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, nullptr);
        std::vector<VkExtensionProperties> exts(n);
        vkEnumerateDeviceExtensionProperties(phys_, nullptr, &n, exts.data());
        auto has = [&](const char* e) {
            for (auto& x : exts)
                if (!std::strcmp(x.extensionName, e)) return true;
            return false;
        };
        std::vector<const char*> want = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                         VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME};
        for (const char* e : want)
            if (!has(e)) return Fail(e);
        if (has(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME)) {
            want.push_back(VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
            foreign_ = true;
        }
        VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        f11.pNext = &f13;
        VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        f2.pNext = &f11;
        auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(vkGetInstanceProcAddr(inst_, "vkGetPhysicalDeviceFeatures2"));
        if (!features2) return Fail("features query");
        features2(phys_, &f2);
        if (!f13.dynamicRendering) return Fail("dynamic rendering");
        VkPhysicalDeviceVulkan13Features e13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
        e13.dynamicRendering = VK_TRUE;
        e13.synchronization2 = f13.synchronization2;
        VkPhysicalDeviceVulkan11Features e11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
        e11.pNext = &e13;
        VkPhysicalDeviceFeatures2 ef{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        ef.pNext = &e11;
        ef.features.samplerAnisotropy = f2.features.samplerAnisotropy;
        ef.features.occlusionQueryPrecise = VK_FALSE;
        const float prio = 1.f;
        VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qci.queueFamilyIndex = qfam_;
        qci.queueCount = 1;
        qci.pQueuePriorities = &prio;
        VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        dci.pNext = &ef;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = uint32_t(want.size());
        dci.ppEnabledExtensionNames = want.data();
        if (vkCreateDevice(phys_, &dci, nullptr, &dev_) != VK_SUCCESS) return Fail("device creation");
        vkGetDeviceQueue(dev_, qfam_, 0, &queue_);
#define LOAD(name) name##_ = reinterpret_cast<PFN_##name>(vkGetDeviceProcAddr(dev_, #name))
        LOAD(vkCmdBeginRendering);
        LOAD(vkCmdEndRendering);
        LOAD(vkCmdSetCullMode);
        LOAD(vkCmdSetFrontFace);
        LOAD(vkCmdSetPrimitiveTopology);
        LOAD(vkCmdSetDepthTestEnable);
        LOAD(vkCmdSetDepthWriteEnable);
        LOAD(vkCmdSetDepthCompareOp);
        LOAD(vkCmdSetStencilTestEnable);
        LOAD(vkCmdSetStencilOp);
        LOAD(vkCmdBindVertexBuffers2);
        LOAD(vkCmdPushDescriptorSetKHR);
        LOAD(vkGetSemaphoreFdKHR);
        LOAD(vkImportSemaphoreFdKHR);
        LOAD(vkGetAndroidHardwareBufferPropertiesANDROID);
#undef LOAD
        if (!vkCmdBeginRendering_ || !vkCmdPushDescriptorSetKHR_ || !vkCmdBindVertexBuffers2_ || !vkGetSemaphoreFdKHR_ ||
            !vkImportSemaphoreFdKHR_ || !vkGetAndroidHardwareBufferPropertiesANDROID_ || !vkCmdSetCullMode_)
            return Fail("device functions");
        // the depth format: D24S8 where it can be drawn and sampled, else D32S8
        depth_format_ = VK_FORMAT_D24_UNORM_S8_UINT;
        VkFormatProperties fp;
        vkGetPhysicalDeviceFormatProperties(phys_, depth_format_, &fp);
        const VkFormatFeatureFlags need = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
        if ((fp.optimalTilingFeatures & need) != need) depth_format_ = VK_FORMAT_D32_SFLOAT_S8_UINT;
        // short vertex inputs read as integers turned float (SSCALED) where the phone has them
        vkGetPhysicalDeviceFormatProperties(phys_, VK_FORMAT_R16G16B16A16_SSCALED, &fp);
        sscaled_ = (fp.bufferFeatures & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT) != 0;
        if (!MakeLayout()) return Fail("pipeline layout");
        // frames: a command pool, a fence, a ring of host visible memory and a query pool each
        for (uint32_t i = 0; i < kFrames; i++) {
            Frame& f = frames_[i];
            VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pci.queueFamilyIndex = qfam_;
            vkCreateCommandPool(dev_, &pci, nullptr, &f.pool);
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
            vkCreateFence(dev_, &fci, nullptr, &f.fence);
            if (!MakeBuffer(kRingBytes, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            f.ring, f.ring_mem, f.ring_map))
                return Fail("ring buffer");
            if (!MakeUniforms(kUniformBytes, f.uniforms)) return Fail("uniform ring");
            VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
            qpi.queryType = VK_QUERY_TYPE_OCCLUSION;
            qpi.queryCount = kQueries;
            vkCreateQueryPool(dev_, &qpi, nullptr, &f.queries);
        }
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VkExportSemaphoreCreateInfo esci{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
        esci.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        sci.pNext = &esci;
        vkCreateSemaphore(dev_, &sci, nullptr, &present_done_);
        VkSemaphoreCreateInfo wci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        vkCreateSemaphore(dev_, &wci, nullptr, &window_done_);
        // the constant inputs (0, 0, 0, 1) and (1, 1, 1, 1), and a 1 x 1 white texture for unbound samplers
        {
            VkDeviceMemory m;
            void* map = nullptr;
            if (!MakeBuffer(64, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, consts_, m, map)) return Fail("constants");
            const float c[8] = {0, 0, 0, 1, 1, 1, 1, 1};
            std::memcpy(map, c, sizeof c);
        }
        auto& chain = platform::GetPresentChain();
        chain.active = chain.window_ready.load();
        g_self = this;
        reference::SetTargetTest(&IsTargetThunk);
        return true;
    }

    bool Init(uint32_t w, uint32_t h) override {
        kx_ = std::clamp(platform::RenderScaleX().load(), 0.5f, 6.f);
        ky_ = std::clamp(platform::RenderScaleY().load(), 0.5f, 6.f);
        if (!reference::g_dump_dir.empty()) {
            cache_dir_ = (std::filesystem::path(reference::g_dump_dir).parent_path() / "vkcache").string();
            std::error_code ec;
            std::filesystem::create_directories(cache_dir_, ec);
        }
        LoadPipelineCache();
        reference::SetGpuMode(true);
        reference::BeginFrame();
        BeginFrame();
        return true;
    }

    bool IsTarget(uint32_t data, uint32_t format, uint32_t w, uint32_t h) const {
        if (!TargetColourFormat(format)) return false;
        for (const auto& t : targets_)
            if (t->data == data && t->w == w && t->h == h)
                return TargetColourFormat(t->format) && Format(t->format).bits == Format(format).bits;
        if (Format(format).bits == 32)
            for (const auto& d : depths_)
                if (d->data == data && d->w == w && d->h == h) return true;
        return false;
    }

    void Clear(const State& s, uint32_t n, const uint8_t* rects, uint32_t flags, uint32_t color, float z, uint32_t stencil) override {
        Target* rt = FindTarget(s.rt, true);
        if (!rt) return;
        CheckGuest(*rt);
        DepthBuf* zs = (flags & 3) ? FindDepth(s.zs, *rt) : nullptr;
        const uint32_t mask = (flags & 0x10 ? 1u : 0u) | (flags & 0x20 ? 2u : 0u) | (flags & 0x40 ? 4u : 0u) | (flags & 0x80 ? 8u : 0u);  // R G B A
        const bool colour = (flags & 0xF0) && mask;
        const bool depth = (flags & 1) && zs, sten = (flags & 2) && zs;
        if (zs && (depth || sten)) zs->written = true;
        if (!colour && !depth && !sten) return;
        bool colour_done = false, depth_done = false;
        // the rects in the GPU's pixels
        std::vector<VkClearRect> rs;
        auto add = [&](int x0, int y0, int x1, int y1) {
            x0 = PX(*rt, std::max(x0, 0));
            y0 = PY(*rt, std::max(y0, 0));
            x1 = PX(*rt, std::min(x1, int(rt->w)));
            y1 = PY(*rt, std::min(y1, int(rt->h)));
            x1 = std::min(x1, int(rt->sw));
            y1 = std::min(y1, int(rt->sh));
            if (x1 > x0 && y1 > y0) rs.push_back({{{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}}, 0, 1});
        };
        if (n && rects) {
            for (uint32_t i = 0; i < n; i++) {
                int32_t r[4];
                std::memcpy(r, rects + i * 16, 16);
                add(r[0], r[1], r[2], r[3]);
            }
        } else {
            const int x = int(s.viewport.x), y = int(s.viewport.y);
            add(x, y, x + int(s.viewport.w), y + int(s.viewport.h));
        }
        if (rs.empty()) return;
        // a whole clear with no pass open on rt: kept for the next pass to begin with (a tiled GPU then skips loading
        // what was there)
        const bool whole = rs.size() == 1 && rs[0].rect.offset.x == 0 && rs[0].rect.offset.y == 0 && rs[0].rect.extent.width == rt->sw &&
                           rs[0].rect.extent.height == rt->sh;
        if (whole && !(rendering_ && cur_rt_ == rt)) {
            if (colour && mask == 0xF) {
                rt->pend = true;
                rt->pend_value = {{float((color >> 16) & 255) / 255.f, float((color >> 8) & 255) / 255.f, float(color & 255) / 255.f,
                                   float(color >> 24) / 255.f}};
                colour_done = true;
            }
            if (depth && sten && zs->sw == rt->sw && zs->sh == rt->sh && !(rendering_ && cur_zs_ == zs)) {
                zs->pend = true;
                zs->initialised = true;
                zs->pend_value = {((2 * z - 1) * kDepthScale + 1) * 0.5f, stencil & 0xFF};
                depth_done = true;
            }
        }
        // the open pass will do when it draws into rt (and has the depth buffer where that is cleared)
        auto bind = [&](bool need_zs) {
            if (rendering_ && cur_rt_ == rt && (!need_zs || cur_zs_ == zs)) return;
            Bind(rt, need_zs ? zs : nullptr);
        };
        // some channels only: drawn with the mask, as a clear writes every channel
        if (colour && mask != 0xF) {
            bind(false);
            MaskedClear(rs, color, mask);
        }
        std::vector<VkClearAttachment> at;
        if (colour && mask == 0xF && !colour_done) {
            VkClearAttachment a{VK_IMAGE_ASPECT_COLOR_BIT, 0, {}};
            a.clearValue.color = {{float((color >> 16) & 255) / 255.f, float((color >> 8) & 255) / 255.f, float(color & 255) / 255.f,
                                   float(color >> 24) / 255.f}};
            at.push_back(a);
        }
        if ((depth || sten) && !depth_done) {
            VkClearAttachment a{VkImageAspectFlags((depth ? VK_IMAGE_ASPECT_DEPTH_BIT : 0) | (sten ? VK_IMAGE_ASPECT_STENCIL_BIT : 0)), 0, {}};
            a.clearValue.depthStencil = {((2 * z - 1) * kDepthScale + 1) * 0.5f, stencil & 0xFF};
            at.push_back(a);
        }
        if (!at.empty()) {
            bind((depth || sten) && !depth_done);
            vkCmdClearAttachments(cmd_, uint32_t(at.size()), at.data(), uint32_t(rs.size()), rs.data());
        }
        if (colour) {
            rt->dirty = true;
            rt->gen++;
            rt->last_draw = ++draws_seen_;
        }
    }

    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override {
        draws_seen_++;
        for (uint32_t th : s.textures) {
            TextureDesc d;
            if (!th || !Describe(th, d)) continue;
            if (!d.cube && d.depth == 1 && IsTarget(d.data, d.format, d.width, d.height)) continue;
            if (d.cube && CubeOfTargets(d)) continue;
            uint32_t lo, hi, w, h;
            if (reference::HeaderRange(th, lo, hi, w, h)) SyncRange(lo, hi);
        }
        const bool counted = s.simple[kMCountEnable >> 2] != 0;
        if (!reference::Prepare(s, prim, start, count, indices, pk_, !counted)) return;
        Target* rt = FrameBufferFor(FindTarget(s.rt, true));
        if (!rt) return;
        CheckGuest(*rt);
        const bool want_zs = s.rs[kRsZEnable] || s.rs[kRsStencilEnable];
        DepthBuf* zs = want_zs ? FindDepth(s.zs, *rt) : nullptr;
        const bool atest = s.simple[kMAlphaTestEnable >> 2] != 0;

        // vertex inputs: the guest's buffers as they are, or the CPU's packed vertices
        Attr attrs[17];
        uint32_t nattr = 0, w1 = 0;
        VkBuffer vbufs[17] = {};
        VkDeviceSize voffs[17] = {}, vstrides[17] = {};
        uint32_t nbind = 0;
        VkDeviceSize ix_off = 0;
        VkIndexType ix_type = VK_INDEX_TYPE_UINT16;
        uint32_t draw_count = 0;
        int32_t base_vertex = 0;
        bool indexed = true, strip = false;
        uint32_t variants = 0;
        int32_t first_vertex = 0;  // where a single stream's vertices start in its buffer, bound at offset 0
        if (pk_.raw) {
            const gpu::Assembled& a = *pk_.as;
            if (a.buffer_count == 1 && a.buffers[0].stride && a.buffers[0].stride % 4 == 0) {
                vbufs[0] = PutVertices(a.buffers[0].src, a.buffers[0].bytes, a.buffers[0].stride, first_vertex);
                vstrides[0] = a.buffers[0].stride;
            } else {
                for (uint32_t b = 0; b < a.buffer_count && b < 16; b++) {
                    vbufs[b] = Put(a.buffers[b].src, std::max<size_t>(a.buffers[b].bytes, 16), 16, voffs[b]);
                    vstrides[b] = a.buffers[b].stride;
                }
            }
            nbind = std::min<uint32_t>(a.buffer_count, 16);
            for (uint32_t r = 0; r < 16; r++) {
                if (pk_.regs >> r & 1) {
                    const gpu::VertexElement& e = a.elements[r];
                    uint32_t variant = 0;
                    bool four = false;
                    const VkFormat f = RawFormat(e.format, uint32_t(a.decode >> (r * 4)) & 15, variant, four);
                    variants |= variant << (r * 2);
                    if (four) w1 |= 1u << r;
                    attrs[nattr++] = {r, e.buffer, e.offset, f};
                } else {
                    attrs[nattr++] = {r, kConstBinding, r == 3 ? 16u : 0u, VK_FORMAT_R32G32B32A32_SFLOAT};
                }
            }
            strip = a.strip;
            if (a.index_count && a.index16) {
                VkBuffer ib = Put(a.index_src, size_t(a.index_count) * 2, 4, ix_off);
                ix_buf_ = ib;
                ix_type = VK_INDEX_TYPE_UINT16;
                base_vertex = a.base_vertex;
                draw_count = a.index_count;
            } else if (a.index_count) {
                const uint32_t* ix = static_cast<const uint32_t*>(a.index_src);
                if (a.vertex_count <= 0xFFFF) {
                    index16_.resize(a.index_count);
                    for (uint32_t i = 0; i < a.index_count; i++) index16_[i] = uint16_t(ix[i]);
                    ix_buf_ = Put(index16_.data(), index16_.size() * 2, 4, ix_off);
                    ix_type = VK_INDEX_TYPE_UINT16;
                } else {
                    ix_buf_ = Put(ix, size_t(a.index_count) * 4, 4, ix_off);
                    ix_type = VK_INDEX_TYPE_UINT32;
                }
                draw_count = a.index_count;
            } else {
                indexed = false;
                draw_count = pk_.raw_count;
            }
        } else {
            const uint32_t mask = UsedTc(pk_);
            const int ntc = __builtin_popcount(mask);
            const size_t stride = 28 + 16 * size_t(ntc), bytes = stride * pk_.verts.size();
            pack_.resize(bytes);
            uint8_t* out = pack_.data();
            auto b8 = [](float v) { return uint8_t(v <= 0 ? 0 : v >= 1 ? 255 : int(v * 255.f + 0.5f)); };
            for (const reference::GlVertex& v : pk_.verts) {
                std::memcpy(out, v.clip, 16);
                for (int j = 0; j < 4; j++) {
                    out[16 + j] = b8(v.v0[j]);
                    out[20 + j] = b8(v.v1[j]);
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
            vbufs[0] = PutVertices(pack_.data(), bytes, uint32_t(stride), first_vertex);
            vstrides[0] = stride;
            nbind = 1;
            attrs[nattr++] = {0, 0, 0, VK_FORMAT_R32G32B32A32_SFLOAT};
            attrs[nattr++] = {1, 0, 16, VK_FORMAT_R8G8B8A8_UNORM};
            attrs[nattr++] = {2, 0, 20, VK_FORMAT_R8G8B8A8_UNORM};
            attrs[nattr++] = {3, 0, 24, VK_FORMAT_R32_SFLOAT};
            uint32_t at = 28;
            for (uint32_t i = 0; i < 4; i++) {
                if (mask >> i & 1) {
                    attrs[nattr++] = {4 + i, 0, at, VK_FORMAT_R32G32B32A32_SFLOAT};
                    at += 16;
                } else {
                    attrs[nattr++] = {4 + i, kConstBinding, 0, VK_FORMAT_R32G32B32A32_SFLOAT};
                }
            }
            if (pk_.verts.size() <= 0xFFFF) {
                index16_.resize(pk_.indices.size());
                for (size_t i = 0; i < pk_.indices.size(); i++) index16_[i] = uint16_t(pk_.indices[i]);
                ix_buf_ = Put(index16_.data(), std::max<size_t>(index16_.size() * 2, 4), 4, ix_off);
                ix_type = VK_INDEX_TYPE_UINT16;
            } else {
                ix_buf_ = Put(pk_.indices.data(), pk_.indices.size() * 4, 4, ix_off);
                ix_type = VK_INDEX_TYPE_UINT32;
            }
            draw_count = uint32_t(pk_.indices.size());
        }
        if (!draw_count) return;
        Prog* prog = GetProgram(pk_, atest, variants, w1);
        if (!prog || !prog->ok) return;

        // the stages' images: sampled targets made readable before the draw's pass begins
        VkDescriptorImageInfo img[4] = {};
        for (int i = 0; i < 4; i++) {
            if (!(prog->samplers >> i & 1)) continue;
            const reference::GlStage& stg = pk_.stage[i];
            const bool cube = pk_.info[i].kind == kTexCube;
            VkImageView view = VK_NULL_HANDLE;
            if (pk_.info[i].kind && stg.target && cube) view = CubeTexture(stg);
            else if (pk_.info[i].kind && stg.target) view = TargetTexture(stg, rt);
            else if (pk_.info[i].kind) view = TextureFor(stg, pk_.info[i].rect, cube);
            if (!view) view = cube ? Dummy(true) : Dummy(false);
            img[i] = {SamplerFor(stg, pk_.info[i].rect, cube), view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        }

        // no depth test or write: the depth buffer bound for this target is attached all the same, so a pass is not
        // ended and begun again each time the title turns its depth test off and on (each restart stores and loads the
        // whole target on a tiled GPU). Only an existing buffer of the target's size; a pass the draw's own image
        // reads ended above, in which case the next pass begins with it anyway
        if (!zs && s.zs.header) {
            if (rendering_ && cur_rt_ == rt && cur_zs_) zs = cur_zs_;
            else zs = ExistingDepth(s.zs, *rt);
        }

        // fixed function state, the pipeline's part
        PipeKey key{};
        key.prog = prog->id;
        const uint32_t cw = s.simple[kMColorWriteEnable >> 2];
        key.cmask = (cw >> 16 & 1 ? 1u : 0u) | (cw >> 8 & 1 ? 2u : 0u) | (cw & 1 ? 4u : 0u) | (cw >> 24 & 1 ? 8u : 0u);
        key.depth = zs != nullptr;
        if (s.simple[kMBlendEnable >> 2]) {
            uint32_t op = s.simple[kMBlendOp >> 2];
            op = op == 0x800A || op == 0x800B || op == 0x8007 || op == 0x8008 ? op : 0x8006;
            key.blend = 1;
            key.src = s.simple[kMSrcBlend >> 2];
            key.dst = s.simple[kMDestBlend >> 2];
            key.op = op;
        }
        key.nattr = nattr;
        for (uint32_t i = 0; i < nattr; i++) key.attr[i] = attrs[i];
        key.nbind = nbind;
        for (uint32_t b = 0; b < nbind; b++) key.stride[b] = uint16_t(vstrides[b]);
        // the rest of the fixed function state goes into the pipeline too: state set per draw cost the driver more
        // than the pipelines it takes
        const bool ztest = zs && s.rs[kRsZEnable], zwrite = ztest && s.simple[kMZWriteEnable >> 2];
        const bool sten = zs && s.rs[kRsStencilEnable];
        {
            const bool cull = pk_.raw && (pk_.cull == 0x900 || pk_.cull == 0x901);
            uint32_t st = (strip ? 1u : 0u) | (cull ? (pk_.cull == 0x900 ? 1u : 2u) : 0u) << 1 | (ztest ? 1u : 0u) << 3 | (zwrite ? 1u : 0u) << 4;
            if (ztest) st |= uint32_t(CompareOp(s.simple[kMZFunc >> 2])) << 5;
            if (sten) {
                st |= 1u << 8;
                st |= uint32_t(StencilOp(s.rs[kRsStencilFail])) << 9 | uint32_t(StencilOp(s.simple[kMStencilPass >> 2])) << 12 |
                      uint32_t(StencilOp(s.simple[kMStencilZFail >> 2])) << 15 | uint32_t(CompareOp(s.simple[kMStencilFunc >> 2])) << 18;
            }
            key.state = st;
        }
        VkPipeline pipe = GetPipeline(key, *prog);
        if (!pipe) return;

        Bind(rt, zs);
        if (bound_pipe_ != pipe) {
            vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
            bound_pipe_ = pipe;
        }
        // dynamic state, sent only where it changed (each change makes the driver redo the draw's state)
        int left = int(s.viewport.x), top = int(s.viewport.y), right = left + int(s.viewport.w), bottom = top + int(s.viewport.h);
        if (s.scissor_count) {
            left = std::max(left, int(s.scissors[0]));
            top = std::max(top, int(s.scissors[1]));
            right = std::min(right, int(s.scissors[2]));
            bottom = std::min(bottom, int(s.scissors[3]));
        }
        {
            int x0 = PX(*rt, std::max(left, 0)), y0 = PY(*rt, std::max(top, 0));
            int x1 = std::min(PX(*rt, std::min(right, int(rt->w))), int(rt->sw)), y1 = std::min(PY(*rt, std::min(bottom, int(rt->h))), int(rt->sh));
            VkRect2D sc{{x0, y0}, {uint32_t(std::max(x1 - x0, 0)), uint32_t(std::max(y1 - y0, 0))}};
            if (std::memcmp(&sc, &dyn_.scissor, sizeof sc) || !dyn_.scissor_set) {
                vkCmdSetScissor(cmd_, 0, 1, &sc);
                dyn_.scissor = sc;
                dyn_.scissor_set = true;
            }
        }
        if (zs && (zwrite || s.rs[kRsStencilEnable])) zs->written = true;
        if (sten) {
            const uint32_t sref = s.simple[kMStencilRef >> 2] & 0xFF, smask = s.simple[kMStencilMask >> 2] & 0xFF;
            const uint32_t wm = s.simple[kMStencilWriteMask >> 2] & 0xFF;
            if (dyn_.smask != int(smask)) vkCmdSetStencilCompareMask(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, uint32_t(dyn_.smask = int(smask)));
            if (dyn_.wmask != int(wm)) vkCmdSetStencilWriteMask(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, uint32_t(dyn_.wmask = int(wm)));
            if (dyn_.sref != int(sref)) vkCmdSetStencilReference(cmd_, VK_STENCIL_FACE_FRONT_AND_BACK, uint32_t(dyn_.sref = int(sref)));
        }
        if (key.blend) {
            const uint32_t bc = s.simple[kMBlendColor >> 2];
            if (!dyn_.bc_set || dyn_.bc != bc) {
                const float c[4] = {float((bc >> 16) & 255) / 255.f, float((bc >> 8) & 255) / 255.f, float(bc & 255) / 255.f, float(bc >> 24) / 255.f};
                vkCmdSetBlendConstants(cmd_, c);
                dyn_.bc = bc;
                dyn_.bc_set = true;
            }
        }

        // uniforms: both stages' blocks in the frame's uniform ring, bound by their offsets
        {
            auto put = [&](uint8_t* blk, int off, const void* src, size_t n) {
                if (off >= 0) std::memcpy(blk + off, src, n);
            };
            // the vertex block (the program's 192 constants and more) is written again only when it changed: draws of one
            // object's parts share it
            VsBlock now;
            now.module = prog->vs;
            now.serial = pk_.raw ? s.vsc_serial : 0;
            now.frame = frame_;
            now.v[0] = 2.f / float(rt->w), now.v[1] = 2.f / float(rt->h), now.v[2] = 2.f / s.zscale, now.v[3] = 0.f;
            if (pk_.raw) {
                for (int k = 0; k < 4; k++) now.v[4 + k] = pk_.fogp[k];
                now.v[8] = pk_.specular ? 1.f : 0.f;
            }
            const bool reuse = vs_last_.set && std::memcmp(&now, &vs_last_, offsetof(VsBlock, set)) == 0;
            // the fragment block made here first: a draw whose blocks are both the last draw's binds nothing
            if (prog->fs_bytes > fs_now_.size()) return;
            const uint32_t fs_bytes = prog->fs_bytes;
            uint8_t* fb = fs_now_.data();
            std::memset(fb, 0, fs_bytes);
            put(fb, prog->pc, &pk_.pc[0][0], 18 * 16);
            put(fb, prog->tf, pk_.tfactor, 16);
            put(fb, prog->fogc, pk_.fog_color, 16);
            const float at[4] = {float(s.simple[kMAlphaFunc >> 2]), float(s.simple[kMAlphaRef >> 2] & 0xFF), 0, 0};
            put(fb, prog->at, at, 16);
            if (prog->bm >= 0 || prog->bl >= 0) {
                float bm[16] = {}, bl[16] = {};
                auto f = [](uint32_t b) { float v; std::memcpy(&v, &b, 4); return v; };
                for (int i = 0; i < 4; i++) {
                    const auto& ts = s.tss[size_t(i)];
                    bm[i * 4] = f(ts[kTssBumpEnvMat00]);
                    bm[i * 4 + 1] = f(ts[kTssBumpEnvMat01]);
                    bm[i * 4 + 2] = f(ts[kTssBumpEnvMat10]);
                    bm[i * 4 + 3] = f(ts[kTssBumpEnvMat11]);
                    bl[i * 4] = f(ts[kTssBumpEnvLScale]);
                    bl[i * 4 + 1] = f(ts[kTssBumpEnvLOffset]);
                }
                put(fb, prog->bm, bm, 64);
                put(fb, prog->bl, bl, 64);
            }
            if (prog->tsz >= 0) {
                float tsz[16] = {};
                for (int i = 0; i < 4; i++) {
                    tsz[i * 4] = float(std::max(pk_.stage[i].w, 1u));
                    tsz[i * 4 + 1] = float(std::max(pk_.stage[i].h, 1u));
                }
                put(fb, prog->tsz, tsz, 64);
            }
            const bool fs_same = reuse && fs_last_.set == vs_last_.set && fs_last_.bytes == fs_bytes && fs_last_.frame == frame_ &&
                                 std::memcmp(fs_last_.data.data(), fb, fs_bytes) == 0;
            if (!fs_same) {
                uint32_t voff = vs_last_.off, foff = 0;
                uint8_t* base = nullptr;
                VkDescriptorSet set = reuse ? UniformRoom(0, fs_bytes, voff, foff, base, vs_last_.set) : VK_NULL_HANDLE;
                const bool wrote_vs = !set;
                if (!set) set = UniformRoom(prog->vs_bytes, fs_bytes, voff, foff, base, VK_NULL_HANDLE);
                if (!set) return;
                if (wrote_vs) {
                    uint8_t* vb = base + voff;
                    put(vb, prog->surf, now.v, 16);
                    if (pk_.raw) {
                        if (prog->c >= 0 && prog->c_hi > prog->c_lo)
                            std::memcpy(vb + prog->c + prog->c_lo * 16, &s.vsc[prog->c_lo][0], (prog->c_hi - prog->c_lo) * 16);
                        put(vb, prog->fogp, now.v + 4, 16);
                        const float misc[4] = {now.v[8], 0, 0, 0};
                        put(vb, prog->vsmisc, misc, 16);
                    }
                    vs_last_ = now;
                    vs_last_.set = set;
                    vs_last_.off = voff;
                }
                std::memcpy(base + foff, fb, fs_bytes);
                std::memcpy(fs_last_.data.data(), fb, fs_bytes);
                fs_last_.set = set;
                fs_last_.off = foff;
                fs_last_.bytes = fs_bytes;
                fs_last_.frame = frame_;
            }
            if (!dyn_.ubo_set || dyn_.ubo != fs_last_.set || dyn_.ubo_off[0] != vs_last_.off || dyn_.ubo_off[1] != fs_last_.off) {
                const uint32_t offs[2] = {vs_last_.off, fs_last_.off};
                vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 0, 1, &fs_last_.set, 2, offs);
                dyn_.ubo = fs_last_.set;
                dyn_.ubo_off[0] = offs[0];
                dyn_.ubo_off[1] = offs[1];
                dyn_.ubo_set = true;
            }
        }
        // samplers: pushed when they differ from the last draw's
        {
            bool same = dyn_.tex_set && dyn_.tex_mask == prog->samplers;
            for (int i = 0; i < 4 && same; i++)
                if ((prog->samplers >> i & 1) && (img[i].imageView != dyn_.tex[i].imageView || img[i].sampler != dyn_.tex[i].sampler)) same = false;
            if (!same && prog->samplers) {
                VkWriteDescriptorSet w[4];
                uint32_t nw = 0;
                for (int i = 0; i < 4; i++) {
                    if (!(prog->samplers >> i & 1)) continue;
                    VkWriteDescriptorSet x{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
                    x.dstBinding = vkglsl::kFirstSampler + uint32_t(i);
                    x.descriptorCount = 1;
                    x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    x.pImageInfo = &img[i];
                    w[nw++] = x;
                }
                vkCmdPushDescriptorSetKHR_(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, nw, w);
                for (int i = 0; i < 4; i++) dyn_.tex[i] = img[i];
                dyn_.tex_mask = prog->samplers;
                dyn_.tex_set = true;
            }
        }

        // buffers
        vbufs[kConstBinding] = consts_;
        voffs[kConstBinding] = 0;
        vstrides[kConstBinding] = 0;
        for (uint32_t b = nbind; b < kConstBinding; b++) {
            vbufs[b] = consts_;
            voffs[b] = 0;
            vstrides[b] = 0;
        }
        // only the bindings that differ from the bound ones (the constant bindings mostly stay)
        {
            uint32_t lo = 0, hi = kConstBinding + 1;
            if (dyn_.vb_set) {
                while (lo < hi && dyn_.vb[lo] == vbufs[lo] && dyn_.voff[lo] == voffs[lo]) lo++;
                while (hi > lo && dyn_.vb[hi - 1] == vbufs[hi - 1] && dyn_.voff[hi - 1] == voffs[hi - 1]) hi--;
            }
            if (lo < hi) vkCmdBindVertexBuffers(cmd_, lo, hi - lo, vbufs + lo, voffs + lo);
            std::memcpy(dyn_.vb, vbufs, sizeof dyn_.vb);
            std::memcpy(dyn_.voff, voffs, sizeof dyn_.voff);
            dyn_.vb_set = true;
        }
        // the index buffer bound at offset 0 of the frame's buffer: the draw's indices are found by firstIndex
        uint32_t first_index = 0;
        if (indexed) {
            const VkDeviceSize unit = ix_type == VK_INDEX_TYPE_UINT16 ? 2 : 4;
            VkDeviceSize bind_off = 0;
            if (ix_off % unit == 0) first_index = uint32_t(ix_off / unit);
            else bind_off = ix_off;
            if (!dyn_.ib_set || dyn_.ib != ix_buf_ || dyn_.ib_off != bind_off || dyn_.ib_type != ix_type) {
                vkCmdBindIndexBuffer(cmd_, ix_buf_, bind_off, ix_type);
                dyn_.ib = ix_buf_;
                dyn_.ib_off = bind_off;
                dyn_.ib_type = ix_type;
                dyn_.ib_set = true;
            }
        }

        std::shared_ptr<Counted> c;
        if (counted) {
            c = std::make_shared<Counted>();
            c->area = CoveredPixels(pk_, left, top, right, bottom);
            c->frame = frame_;
            c->slot = uint32_t(frame_ % kFrames);
            if (frames_[c->slot].next_query < kQueries) {
                c->query = frames_[c->slot].next_query++;
                vkCmdBeginQuery(cmd_, frames_[c->slot].queries, c->query, 0);
            }
        }
        if (indexed) vkCmdDrawIndexed(cmd_, draw_count, 1, first_index, base_vertex + first_vertex, 0);
        else vkCmdDraw(cmd_, draw_count, 1, uint32_t(first_vertex), 0);
        if (c) {
            if (c->query != ~0u) vkCmdEndQuery(cmd_, frames_[c->slot].queries, c->query);
            counted_.push_back(std::move(c));
        }
        if (key.cmask) {
            rt->dirty = true;
            rt->gen++;
            rt->last_draw = draws_seen_;
            rt->frame_draws++;
        }
    }

    void Copy(const Surface& src, const Surface& dst, uint32_t n, uint32_t rects, uint32_t points, float, float) override {
        uint32_t lo, hi, w, h;
        if (src.header && reference::HeaderRange(src.header, lo, hi, w, h)) SyncRange(lo, hi);
        if (dst.header && reference::HeaderRange(dst.header, lo, hi, w, h)) {
            SyncRange(lo, hi);
            reference::Copy(src, dst, n, rects, points);
            for (auto& t : targets_)
                if (t->data < hi && lo < t->end) UploadGuest(*t);
            return;
        }
        reference::Copy(src, dst, n, rects, points);
    }

    void Present(const Surface& front) override {
        if (!power_registered_) {
            power_registered_ = true;
            power::RegisterThread(power::kRender);
        }
        // the render thread's work this frame for Android's performance hints, beside the game thread's
        power::RenderFrame(g_render_busy_ns.exchange(0, std::memory_order_relaxed));
        {
            const float kx = std::clamp(platform::RenderScaleX().load(), 0.5f, 6.f), ky = std::clamp(platform::RenderScaleY().load(), 0.5f, 6.f);
            if (std::fabs(kx - kx_) > 0.001f || std::fabs(ky - ky_) > 0.001f) Rescale(kx, ky);
        }
        Target* t = nullptr;
        uint32_t lo = 0, hi = 0, w = 0, h = 0;
        if (SurfaceRange(front, lo, hi, w, h))
            for (auto& x : targets_)
                if (x->data == lo && x->w == w && x->h == h) t = x.get();
        if (t) frame_buffers_.insert(t->data);
        if (t) {
            Target* most = t;
            for (auto& x : targets_)
                if (x->w == t->w && x->h == t->h && x->frame_draws > most->frame_draws) most = x.get();
            t = most;
        }
        for (auto& x : targets_) {
            if (x->frame_draws > 64) x->last_world = frame_;
            x->frame_draws = 0;
        }
        EndRendering();
        bool shown = false;
        if (t) {
            CheckGuest(*t);
            Settle(*t);
            shown = PresentDirect(*t);
            if (!shown) shown = PresentThroughMemory(*t);
        } else if (SurfaceRange(front, lo, hi, w, h)) {
            SyncRange(lo, hi);
            auto& f = platform::GetFrame();
            std::lock_guard lock(f.lock);
            if (reference::ReadRgba(front, f.w, f.h, f.rgba)) f.serial++;
        }
        if (!shown) Submit(nullptr, nullptr, true);
        frame_++;
        BeginFrame();
        DeliverReports();
        if (frame_ % 60 == 0) Evict();
        if (frame_ % 1800 == 0) SavePipelineCache();
        reference::BeginFrame();
    }

    void Sync(uint32_t data_va, uint32_t size) override { SyncRange(data_va, data_va + std::max(size, 1u)); }
    void SyncLate(uint32_t data_va) override { SyncRange(data_va, data_va + 1); }
    void Flush() override {}
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
    void ClearReport() override { counted_.clear(); }
    void GetReport(uint32_t report) override {
        if (report) reports_.push_back({report, counted_, frame_});
    }

private:
    bool Fail(const char* why) {
        return false;
    }

    // ---- memory, buffers, images ----

    uint32_t MemType(uint32_t bits, VkMemoryPropertyFlags want) const {
        for (uint32_t i = 0; i < memprops_.memoryTypeCount; i++)
            if ((bits >> i & 1) && (memprops_.memoryTypes[i].propertyFlags & want) == want) return i;
        return ~0u;
    }
    bool MakeBuffer(size_t bytes, VkBufferUsageFlags usage, VkBuffer& buf, VkDeviceMemory& mem, void*& map) {
        VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size = bytes;
        bci.usage = usage;
        if (vkCreateBuffer(dev_, &bci, nullptr, &buf) != VK_SUCCESS) return false;
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(dev_, buf, &req);
        uint32_t t = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        if (t == ~0u) t = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (t == ~0u) return false;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = t;
        if (vkAllocateMemory(dev_, &mai, nullptr, &mem) != VK_SUCCESS) return false;
        vkBindBufferMemory(dev_, buf, mem, 0);
        return vkMapMemory(dev_, mem, 0, VK_WHOLE_SIZE, 0, &map) == VK_SUCCESS;
    }
    VkImageView View(VkImage img, VkFormat fmt, VkImageAspectFlags aspect, bool cube, uint32_t levels, bool alpha_one) {
        VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        v.image = img;
        v.viewType = cube ? VK_IMAGE_VIEW_TYPE_CUBE : VK_IMAGE_VIEW_TYPE_2D;
        v.format = fmt;
        if (alpha_one) v.components.a = VK_COMPONENT_SWIZZLE_ONE;
        v.subresourceRange = {aspect, 0, levels, 0, cube ? 6u : 1u};
        VkImageView out = VK_NULL_HANDLE;
        vkCreateImageView(dev_, &v, nullptr, &out);
        return out;
    }
    // a device local image; colour images get a second view with alpha read as 1
    Image MakeImage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, uint32_t levels = 1, bool cube = false) {
        Image im;
        const bool depth = fmt == VK_FORMAT_D24_UNORM_S8_UINT || fmt == VK_FORMAT_D32_SFLOAT_S8_UINT;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.flags = cube ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = fmt;
        ici.extent = {std::max(w, 1u), std::max(h, 1u), 1};
        ici.mipLevels = levels;
        ici.arrayLayers = cube ? 6 : 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = usage;
        ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        if (vkCreateImage(dev_, &ici, nullptr, &im.img) != VK_SUCCESS) return Image{};
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev_, im.img, &req);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = req.size;
        mai.memoryTypeIndex = MemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        if (mai.memoryTypeIndex == ~0u || vkAllocateMemory(dev_, &mai, nullptr, &im.mem) != VK_SUCCESS) {
            vkDestroyImage(dev_, im.img, nullptr);
            return Image{};
        }
        vkBindImageMemory(dev_, im.img, im.mem, 0);
        im.format = fmt;
        im.aspect = depth ? VkImageAspectFlags(VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
        im.w = std::max(w, 1u);
        im.h = std::max(h, 1u);
        im.levels = levels;
        im.layers = cube ? 6 : 1;
        im.cube = cube;
        if (depth) {
            im.view = View(im.img, fmt, im.aspect, false, 1, false);
            im.depth_view = View(im.img, fmt, VK_IMAGE_ASPECT_DEPTH_BIT, false, 1, false);
        } else {
            im.view = View(im.img, fmt, im.aspect, cube, levels, false);
            if (!cube) im.view_one = View(im.img, fmt, im.aspect, false, levels, true);
        }
        return im;
    }
    // freed once the GPU is past the frame that last used it
    void Drop(Image& im) {
        if (im.img) frames_[frame_ % kFrames].drop_images.push_back(im);
        im = Image{};
    }
    void Destroy(Image& im) {
        if (im.view) vkDestroyImageView(dev_, im.view, nullptr);
        if (im.view_one) vkDestroyImageView(dev_, im.view_one, nullptr);
        if (im.depth_view) vkDestroyImageView(dev_, im.depth_view, nullptr);
        if (im.img) vkDestroyImage(dev_, im.img, nullptr);
        if (im.mem) vkFreeMemory(dev_, im.mem, nullptr);
        im = Image{};
    }

    // a layout change of a whole image, in the main command buffer (outside a pass) or the given one
    // the stages and accesses an image's layout stands for: a barrier waits for those only (a barrier over all
    // commands drained the GPU at every change of target)
    static void Use(VkImageLayout l, VkPipelineStageFlags& stage, VkAccessFlags& access) {
        switch (l) {
            case VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL:
                stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
                access = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
                break;
            case VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL:
                stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
                access = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
                break;
            case VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL:
                stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
                access = VK_ACCESS_SHADER_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL:
                stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                access = VK_ACCESS_TRANSFER_READ_BIT;
                break;
            case VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL:
                stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
                access = VK_ACCESS_TRANSFER_WRITE_BIT;
                break;
            case VK_IMAGE_LAYOUT_UNDEFINED:
                stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
                access = 0;
                break;
            default:
                stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
                access = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
                break;
        }
    }
    void Barrier(VkCommandBuffer cb, Image& im, VkImageLayout to) {
        if (im.layout == to) return;
        VkPipelineStageFlags src_stage, dst_stage;
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        Use(im.layout, src_stage, b.srcAccessMask);
        Use(to, dst_stage, b.dstAccessMask);
        if (im.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) b.srcAccessMask = 0;  // reads need no flush, only to be over
        b.oldLayout = im.layout;
        b.newLayout = to;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = im.img;
        b.subresourceRange = {im.aspect, 0, im.levels, 0, im.layers};
        vkCmdPipelineBarrier(cb, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &b);
        im.layout = to;
    }
    void Layout(Image& im, VkImageLayout to) {
        if (im.layout == to) return;
        EndRendering();
        Barrier(cmd_, im, to);
    }

    // ---- frames and command buffers ----

    struct Frame {
        VkCommandPool pool = VK_NULL_HANDLE;
        std::vector<VkCommandBuffer> cbs;
        uint32_t used = 0;
        VkFence fence = VK_NULL_HANDLE;
        VkBuffer ring = VK_NULL_HANDLE;
        VkDeviceMemory ring_mem = VK_NULL_HANDLE;
        void* ring_map = nullptr;
        size_t head = 0;
        struct Extra { VkBuffer buf; VkDeviceMemory mem; void* map; size_t size, head; };
        std::vector<Extra> extra;  // more room when the ring runs out
        VkQueryPool queries = VK_NULL_HANDLE;
        uint32_t next_query = 0;
        std::vector<Image> drop_images;
        bool submitted = false;
        struct Uniforms {
            VkBuffer buf = VK_NULL_HANDLE;
            VkDeviceMemory mem = VK_NULL_HANDLE;
            uint8_t* map = nullptr;
            size_t size = 0, head = 0;
            VkDescriptorSet set = VK_NULL_HANDLE;
        };
        Uniforms uniforms;
        std::vector<Uniforms> uextra;
    };
    VkCommandBuffer NewCmd() {
        Frame& f = frames_[frame_ % kFrames];
        if (f.used == f.cbs.size()) {
            VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            ai.commandPool = f.pool;
            ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ai.commandBufferCount = 1;
            VkCommandBuffer cb;
            vkAllocateCommandBuffers(dev_, &ai, &cb);
            f.cbs.push_back(cb);
        }
        VkCommandBuffer cb = f.cbs[f.used++];
        dyn_ = Dyn{};  // a new command buffer has no state
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cb, &bi);
        return cb;
    }
    // the frame slot is free again: its fence passed, its command buffers, ring and dropped images are reused
    void BeginFrame() {
        Frame& f = frames_[frame_ % kFrames];
        if (f.submitted) vkWaitForFences(dev_, 1, &f.fence, VK_TRUE, UINT64_MAX);
        f.submitted = false;
        DeliverReportsOf(uint32_t(frame_ % kFrames));
        vkResetCommandPool(dev_, f.pool, 0);
        f.used = 0;
        f.head = 0;
        for (auto& e : f.extra) {
            vkUnmapMemory(dev_, e.mem);
            vkDestroyBuffer(dev_, e.buf, nullptr);
            vkFreeMemory(dev_, e.mem, nullptr);
        }
        f.extra.clear();
        f.uniforms.head = 0;
        for (auto& u : f.uextra) FreeUniforms(u);
        f.uextra.clear();
        for (auto& im : f.drop_images) Destroy(im);
        f.drop_images.clear();
        f.next_query = 0;
        cmd_ = NewCmd();
        upload_ = VK_NULL_HANDLE;
        vkCmdResetQueryPool(cmd_, f.queries, 0, kQueries);
        bound_pipe_ = VK_NULL_HANDLE;
        rendering_ = false;
    }
    // the upload command buffer runs before the main one of the same submit: new images get their bytes there
    VkCommandBuffer Upload() {
        if (!upload_) upload_ = NewCmd();
        return upload_;
    }
    // what is recorded so far to the GPU; wait/signal: semaphores for the present; then a new main command buffer
    void Submit(VkSemaphore wait, VkSemaphore signal, bool fence) {
        EndRendering();
        vkEndCommandBuffer(cmd_);
        VkCommandBuffer cbs[2];
        uint32_t ncb = 0;
        if (upload_) {
            vkEndCommandBuffer(upload_);
            cbs[ncb++] = upload_;
        }
        cbs[ncb++] = cmd_;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
        if (wait) {
            si.waitSemaphoreCount = 1;
            si.pWaitSemaphores = &wait;
            si.pWaitDstStageMask = &stage;
        }
        if (signal) {
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &signal;
        }
        si.commandBufferCount = ncb;
        si.pCommandBuffers = cbs;
        Frame& f = frames_[frame_ % kFrames];
        if (fence) {
            vkResetFences(dev_, 1, &f.fence);
            f.submitted = true;
        }
        vkQueueSubmit(queue_, 1, &si, fence ? f.fence : VK_NULL_HANDLE);
        upload_ = VK_NULL_HANDLE;
        if (!fence) cmd_ = NewCmd();
        bound_pipe_ = VK_NULL_HANDLE;
    }
    // everything so far done on the GPU (a read back of a target)
    void Finish() {
        Submit(nullptr, nullptr, false);
        vkQueueWaitIdle(queue_);
    }

    // data into the frame's ring: the buffer and offset
    VkBuffer Put(const void* data, size_t bytes, size_t align, VkDeviceSize& off) {
        uint8_t* dst = nullptr;
        VkBuffer b = Room(bytes, align, off, dst);
        if (dst) std::memcpy(dst, data, bytes);
        return b;
    }
    // vertices of one stream at a multiple of their stride: the buffer stays bound at offset 0 and the draw starts at
    // first (a binding changed for every draw made the driver set the vertex fetch up again each time)
    VkBuffer PutVertices(const void* data, size_t bytes, uint32_t stride, int32_t& first) {
        uint8_t* dst = nullptr;
        VkDeviceSize at = 0;
        VkBuffer b = Room(std::max<size_t>(bytes, 16) + stride, 4, at, dst);
        const VkDeviceSize pad = (stride - at % stride) % stride;
        first = int32_t((at + pad) / stride);
        if (dst) std::memcpy(dst + pad, data, bytes);
        return b;
    }
    VkBuffer PutUniform(uint32_t bytes, VkDeviceSize& off, uint8_t*& dst) {
        if (!bytes) {
            off = 0;
            dst = nullptr;
            return frames_[frame_ % kFrames].ring;
        }
        VkBuffer b = Room(bytes, std::max<size_t>(props_.limits.minUniformBufferOffsetAlignment, 16), off, dst);
        return b;
    }
    VkBuffer Room(size_t bytes, size_t align, VkDeviceSize& off, uint8_t*& dst) {
        Frame& f = frames_[frame_ % kFrames];
        size_t at = (f.head + align - 1) & ~(align - 1);
        if (at + bytes <= kRingBytes) {
            f.head = at + bytes;
            off = at;
            dst = static_cast<uint8_t*>(f.ring_map) + at;
            return f.ring;
        }
        for (auto& e : f.extra) {
            at = (e.head + align - 1) & ~(align - 1);
            if (at + bytes <= e.size) {
                e.head = at + bytes;
                off = at;
                dst = static_cast<uint8_t*>(e.map) + at;
                return e.buf;
            }
        }
        Frame::Extra e{};
        e.size = std::max<size_t>(bytes + align, 16u << 20);
        if (!MakeBuffer(e.size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        e.buf, e.mem, e.map)) {
            off = 0;
            dst = nullptr;
            return f.ring;
        }
        e.head = bytes;
        off = 0;
        dst = static_cast<uint8_t*>(e.map);
        f.extra.push_back(e);
        return e.buf;
    }

    // ---- passes ----

    // drawing into rt (and zs): the pass begins unless it is the one open
    void Bind(Target* rt, DepthBuf* zs) {
        if (rendering_ && rt == cur_rt_ && zs == cur_zs_) return;
        EndRendering();
        Barrier(cmd_, rt->img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        if (zs) {
            if (!zs->initialised) {
                // a new depth buffer starts at the far plane
                zs->initialised = true;
                zs->pend = true;
                zs->pend_value = {1.f, 0};
            }
            Barrier(cmd_, zs->img, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        }
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = rt->img.view;
        ca.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        ca.loadOp = rt->pend ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca.clearValue.color = rt->pend_value;
        rt->pend = false;
        VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (zs) {
            da.imageView = zs->img.view;
            da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
            da.loadOp = zs->pend ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD;
            da.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            da.clearValue.depthStencil = zs->pend_value;
            zs->pend = false;
        }
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {rt->sw, rt->sh}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        if (zs) {
            ri.pDepthAttachment = &da;
            ri.pStencilAttachment = &da;
        }
        vkCmdBeginRendering_(cmd_, &ri);
        rendering_ = true;
        cur_rt_ = rt;
        cur_zs_ = zs;
        // the viewport is the whole target, as GL had it
        VkViewport vp{0, 0, float(rt->sw), float(rt->sh), 0.f, 1.f};
        vkCmdSetViewport(cmd_, 0, 1, &vp);
        bound_pipe_ = VK_NULL_HANDLE;
    }
    // a pending whole clear done now: the image is about to be read or copied outside a pass of its own
    void Settle(Target& t) {
        if (!t.pend) return;
        t.pend = false;
        Layout(t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cmd_, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &t.pend_value, 1, &r);
    }
    void Settle(DepthBuf& d) {
        if (!d.pend) return;
        d.pend = false;
        Layout(d.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageSubresourceRange r{d.img.aspect, 0, 1, 0, 1};
        vkCmdClearDepthStencilImage(cmd_, d.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &d.pend_value, 1, &r);
    }
    void EndRendering() {
        if (!rendering_) return;
        vkCmdEndRendering_(cmd_);
        rendering_ = false;
        Target* ended = cur_rt_;
        cur_rt_ = nullptr;
        cur_zs_ = nullptr;
        // a cube face just drawn: copied into its cube now, while no pass is open (when the cube is read, a pass into
        // the frame's main target would have to stop for it)
        if (ended && ended->cube) {
            auto it = cubes_.find(ended->cube);
            if (it != cubes_.end() && it->second.img && it->second.gen[ended->face] != ended->gen && !ended->pend)
                UpdateFace(it->second, ended->face, *ended);
        }
        // made ready to be read at once: a draw that samples it later would otherwise end the pass open then, and on a
        // tiled GPU each ended pass stores the whole target (and its depth) and the next loads it again; at the screen's
        // resolution that was most of the GPU's memory traffic in a race (about 10 such ends a frame)
        if (ended && ended->img.layout == VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL) Barrier(cmd_, ended->img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // ---- targets ----

    static bool SurfaceRange(const Surface& s, uint32_t& lo, uint32_t& hi, uint32_t& w, uint32_t& h) {
        TextureDesc d;
        if (!s.header || !DescribeSurface(s, d)) return false;
        lo = d.data;
        hi = d.data + TotalSize(d);
        w = d.width;
        h = d.height;
        return true;
    }
    bool Frozen(const Target& t) const { return t.last_world && t.last_world + 1 < frame_ && frame_ - t.last_world < 1800; }
    Target* FrameBufferFor(Target* rt) {
        if (!rt || !frame_buffers_.count(rt->data) || rt->frame_draws > 64) return rt;
        if (Frozen(*rt)) return rt;
        for (auto& x : targets_)
            if (x.get() != rt && x->w == rt->w && x->h == rt->h && frame_buffers_.count(x->data) && x->frame_draws > 64 &&
                x->frame_draws > rt->frame_draws * 4)
                return x.get();
        return rt;
    }
    static constexpr VkImageUsageFlags kTargetUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
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
        t->data = lo;
        t->end = hi;
        t->w = w;
        t->h = h;
        t->sw = TW(w, h);
        t->sh = TH(w, h);
        t->surf = s;
        TextureDesc d;
        if (Describe(s.header, d)) t->format = d.format;
        t->img = MakeImage(t->sw, t->sh, kColour, kTargetUsage);
        if (!t->img) return nullptr;
        ClearImage(t->img);
        targets_.push_back(std::move(t));
        Target* made = targets_.back().get();
        UploadGuest(*made);
        made->checked = frame_;
        return made;
    }
    void ClearImage(Image& im) {
        Layout(im, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkClearColorValue c{};
        VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, im.levels, 0, im.layers};
        vkCmdClearColorImage(cmd_, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
    }
    Target* TargetAt(uint32_t data, uint32_t w, uint32_t h) {
        for (auto& t : targets_)
            if (t->data == data && t->w == w && t->h == h) return t.get();
        return nullptr;
    }
    DepthBuf* ExistingDepth(const Surface& s, const Target& rt) {
        uint32_t lo, hi, w, h;
        if (!SurfaceRange(s, lo, hi, w, h) || w < rt.w || h < rt.h) return nullptr;
        for (auto& d : depths_)
            if (d->data == lo && d->w == rt.w && d->h == rt.h) return d.get();
        return nullptr;
    }
    DepthBuf* FindDepth(const Surface& s, const Target& rt) {
        uint32_t lo, hi, w, h;
        if (!SurfaceRange(s, lo, hi, w, h) || w < rt.w || h < rt.h) return nullptr;
        w = rt.w;
        h = rt.h;
        for (auto& d : depths_)
            if (d->data == lo && d->w == w && d->h == h) return d.get();
        auto d = std::make_unique<DepthBuf>();
        d->data = lo;
        d->w = w;
        d->h = h;
        d->sw = TW(w, h);
        d->sh = TH(w, h);
        d->img = MakeImage(d->sw, d->sh, depth_format_,
                           VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (!d->img) return nullptr;
        depths_.push_back(std::move(d));
        return depths_.back().get();
    }

    // a stage reading a target: its image, a copy when the draw writes it, alpha 1 for formats without
    VkImageView TargetTexture(const reference::GlStage& stg, Target* draw_rt) {
        Target* t = TargetAt(stg.target, stg.w, stg.h);
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
            return VK_NULL_HANDLE;
        }
        CheckGuest(*t);
        Settle(*t);
        const bool one = OpaqueFormat(stg.format);
        if (t != draw_rt) {
            Layout(t->img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            return one ? t->img.view_one : t->img.view;
        }
        // the console reads the pixels it is about to overwrite: the draw gets a copy
        if (t->copy && t->copy_gen == t->gen && t->copy.layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) return one ? t->copy.view_one : t->copy.view;
        if (!t->copy) t->copy = MakeImage(t->sw, t->sh, kColour, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        t->copy_gen = t->gen;
        Layout(t->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Barrier(cmd_, t->copy, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageCopy c{};
        c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.extent = {t->sw, t->sh, 1};
        vkCmdCopyImage(cmd_, t->img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t->copy.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        Barrier(cmd_, t->copy, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return one ? t->copy.view_one : t->copy.view;
    }

    bool CubeOfTargets(const TextureDesc& d) const {
        for (uint32_t f = 0; f < 6; f++)
            if (!IsTarget(d.data + f * FaceSize(d), d.format, d.width, d.height)) return false;
        return true;
    }
    struct Cube {
        Image img;
        uint32_t size = 0;
        uint64_t gen[6] = {};
    };
    // a cube whose faces the title renders as six targets: each face scaled into the square cube on the GPU, again only
    // when that target was drawn since
    VkImageView CubeTexture(const reference::GlStage& stg) {
        Cube& c = cubes_[stg.target];
        const uint32_t size = std::max(1u, uint32_t(std::lround(float(stg.w) * std::max(ScaleOf(stg.w, stg.h, kx_), ScaleOf(stg.w, stg.h, ky_)))));
        if (!c.img || c.size != size) {
            Drop(c.img);
            c = Cube{};
            c.img = MakeImage(size, size, kColour, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, true);
            c.size = size;
            if (!c.img) return VK_NULL_HANDLE;
            ClearImage(c.img);
        }
        for (uint32_t f = 0; f < 6; f++) {
            Target* t = TargetAt(stg.target + f * stg.face_stride, stg.w, stg.h);
            if (!t) continue;
            t->cube = stg.target;  // from now on its face is copied as soon as a pass into it ends (EndRendering)
            t->face = f;
            CheckGuest(*t);
            Settle(*t);
            if (c.gen[f] == t->gen) continue;
            EndRendering();
            UpdateFace(c, f, *t);
        }
        Layout(c.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return c.img.view;
    }
    // a face's target scaled into the square cube (outside a pass), the cube left readable
    void UpdateFace(Cube& c, uint32_t f, Target& t) {
        c.gen[f] = t.gen;
        Barrier(cmd_, t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        Barrier(cmd_, c.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        VkImageBlit b{};
        b.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        b.srcOffsets[1] = {int32_t(t.sw), int32_t(t.sh), 1};
        b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, f, 1};
        b.dstOffsets[1] = {int32_t(c.size), int32_t(c.size), 1};
        vkCmdBlitImage(cmd_, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, c.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
        Barrier(cmd_, c.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    // the render scale changed (the window still turning as the game started): every target redrawn at the new scale
    // from what it holds, the depth buffers made again
    void Rescale(float kx, float ky) {
        EndRendering();
        for (auto& t : targets_) {
            Settle(*t);
            const uint32_t sw = TW(t->w, t->h, kx), sh = TH(t->w, t->h, ky);
            Image img = MakeImage(sw, sh, kColour, kTargetUsage);
            if (!img) continue;
            Layout(t->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            Barrier(cmd_, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageBlit b{};
            b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.srcOffsets[1] = {int32_t(t->sw), int32_t(t->sh), 1};
            b.dstOffsets[1] = {int32_t(sw), int32_t(sh), 1};
            vkCmdBlitImage(cmd_, t->img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
            Drop(t->img);
            Drop(t->copy);
            t->img = img;
            t->sw = sw;
            t->sh = sh;
        }
        for (auto& d : depths_) {
            Drop(d->img);
            Drop(d->colour);
        }
        depths_.clear();
        kx_ = kx;
        ky_ = ky;
    }

    // ---- guest memory ----

    void SyncRange(uint32_t lo, uint32_t hi) {
        for (auto& t : targets_)
            if (t->dirty && t->data < hi && lo < t->end) Readback(*t);
    }
    // the target into guest memory, at the console's size: filtered down on the GPU, copied out, waited for
    void Readback(Target& t) {
        Settle(t);
        VkDeviceSize off = 0;
        uint8_t* dst = nullptr;
        const size_t bytes = size_t(t.w) * t.h * 4;
        VkBuffer buf = Room(bytes, 16, off, dst);
        if (!dst) return;
        Image* src = &t.img;
        if (t.sw != t.w || t.sh != t.h) {
            Image& sc = ScratchFor(t.w, t.h);
            Layout(t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            Barrier(cmd_, sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageBlit b{};
            b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.srcOffsets[1] = {int32_t(t.sw), int32_t(t.sh), 1};
            b.dstOffsets[1] = {int32_t(t.w), int32_t(t.h), 1};
            vkCmdBlitImage(cmd_, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, sc.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                           VK_FILTER_LINEAR);
            src = &sc;
        }
        Layout(*src, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {t.w, t.h, 1};
        vkCmdCopyImageToBuffer(cmd_, src->img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
        Finish();
        readback_.resize(size_t(t.w) * t.h);
        std::memcpy(readback_.data(), dst, bytes);
        reference::WriteRgba(t.surf, readback_.data(), t.w, t.h);
        t.dirty = false;
        for (auto& o : targets_)
            if (o->data < t.end && t.data < o->end) Remember(*o);
    }
    Image& ScratchFor(uint32_t w, uint32_t h) {
        Image& sc = scratch_[uint64_t(w) << 32 | h];
        if (!sc) sc = MakeImage(w, h, kColour, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        return sc;
    }
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
    static bool GuestChanged(Target& t) {
        const uint32_t k = t.next_stripe++ & 7;
        return GuestStripe(t, k) != t.known[k];
    }
    void CheckGuest(Target& t) {
        if (t.checked == frame_ || (t.checked && frame_ - t.checked < 2)) return;
        t.checked = frame_;
        if (!GuestChanged(t)) return;
        UploadGuest(t);
    }
    // the title wrote the surface's memory itself: the target takes it (stretched to the GPU's size)
    void UploadGuest(Target& t) {
        uint32_t w, h;
        Remember(t);
        Settle(t);
        if (!reference::ReadRgba(t.surf, w, h, upload_px_) || w != t.w || h != t.h) return;
        VkDeviceSize off = 0;
        VkBuffer buf = Put(upload_px_.data(), upload_px_.size() * 4, 16, off);
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {w, h, 1};
        if (t.sw == w && t.sh == h) {
            Layout(t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdCopyBufferToImage(cmd_, buf, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        } else {
            Image& sc = ScratchFor(w, h);
            Layout(sc, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            vkCmdCopyBufferToImage(cmd_, buf, sc.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            Barrier(cmd_, sc, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            Barrier(cmd_, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkImageBlit b{};
            b.srcSubresource = b.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            b.srcOffsets[1] = {int32_t(w), int32_t(h), 1};
            b.dstOffsets[1] = {int32_t(t.sw), int32_t(t.sh), 1};
            vkCmdBlitImage(cmd_, sc.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b,
                           VK_FILTER_NEAREST);
        }
        t.dirty = false;
        t.gen++;
    }

    // ---- textures and samplers ----

    static uint32_t Levels(uint32_t w, uint32_t h, bool mips) {
        uint32_t levels = 1;
        if (mips)
            for (uint32_t m = std::max(w, h); m > 1; m >>= 1) levels++;
        return levels;
    }
    // a guest texture's GPU image: made again when its bytes change (draws before the change still read the old one),
    // the bytes and the mip levels recorded in the upload command buffer
    VkImageView TextureFor(const reference::GlStage& s, bool rect, bool cube) {
        if (!s.rgba) return VK_NULL_HANDLE;
        const uint64_t key = s.slot ? s.slot : (s.id | 1ull << 63);
        auto it = textures_.find(key);
        if (it != textures_.end()) {
            it->second.used = frame_;
            if (it->second.image == s.id) return it->second.img.view;
            Drop(it->second.img);
            textures_.erase(it);
        }
        const uint32_t levels = Levels(s.w, s.h, !rect && s.w > 1 && s.h > 1);
        Tex t;
        t.img = MakeImage(s.w, s.h, kColour, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                          levels, cube);
        if (!t.img) return VK_NULL_HANDLE;
        VkCommandBuffer cb = Upload();
        Barrier(cb, t.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        const size_t face = size_t(s.w) * s.h * 4;
        for (uint32_t f = 0; f < (cube ? 6u : 1u); f++) {
            const uint8_t* px = cube ? s.faces[f] : s.rgba;
            if (!px) continue;
            VkDeviceSize off = 0;
            VkBuffer buf = Put(px, face, 16, off);
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, f, 1};
            r.imageExtent = {s.w, s.h, 1};
            vkCmdCopyBufferToImage(cb, buf, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
        }
        if (levels > 1) {
            // each level filtered from the one above
            for (uint32_t l = 1; l < levels; l++) {
                VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
                b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
                b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
                b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = t.img.img;
                b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 1, 0, t.img.layers};
                vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
                VkImageBlit bl{};
                bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l - 1, 0, t.img.layers};
                bl.srcOffsets[1] = {int32_t(std::max(s.w >> (l - 1), 1u)), int32_t(std::max(s.h >> (l - 1), 1u)), 1};
                bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, t.img.layers};
                bl.dstOffsets[1] = {int32_t(std::max(s.w >> l, 1u)), int32_t(std::max(s.h >> l, 1u)), 1};
                vkCmdBlitImage(cb, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl,
                               VK_FILTER_LINEAR);
            }
            // levels 0..n-2 are transfer sources now, the last a destination: all to shader reads
            VkImageMemoryBarrier b[2] = {{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}, {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER}};
            for (auto& x : b) {
                x.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
                x.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                x.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
                x.srcQueueFamilyIndex = x.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                x.image = t.img.img;
            }
            b[0].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b[0].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, levels - 1, 0, t.img.layers};
            b[1].oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b[1].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, levels - 1, 1, 0, t.img.layers};
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 2, b);
            t.img.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        } else {
            Barrier(cb, t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        t.image = s.id;
        t.used = frame_;
        VkImageView v = t.img.view;
        textures_[key] = t;
        return v;
    }
    // a 1 x 1 white image (or cube) for a sampler the shader declares and the draw leaves empty
    VkImageView Dummy(bool cube) {
        Image& d = cube ? dummy_cube_ : dummy_;
        if (!d) {
            d = MakeImage(1, 1, kColour, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, 1, cube);
            VkCommandBuffer cb = Upload();
            Barrier(cb, d, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkClearColorValue c{{1, 1, 1, 1}};
            VkImageSubresourceRange r{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, d.layers};
            vkCmdClearColorImage(cb, d.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &c, 1, &r);
            Barrier(cb, d, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        return d.view;
    }
    VkSampler SamplerFor(const reference::GlStage& s, bool rect, bool cube) {
        const uint32_t key = s.address_u | s.address_v << 4 | std::min(s.mag, 3u) << 8 | std::min(s.min, 3u) << 10 | std::min(s.mip, 3u) << 12 |
                             (rect ? 1u << 14 : 0) | (cube ? 1u << 15 : 0);
        auto it = samplers_.find(key);
        if (it != samplers_.end()) return it->second;
        auto wrap = [&](uint32_t m) {
            return cube ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE
                        : m == 1 ? VK_SAMPLER_ADDRESS_MODE_REPEAT : m == 2 ? VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT : VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        };
        VkSamplerCreateInfo c{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        c.magFilter = s.mag >= 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        c.minFilter = s.min >= 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
        c.addressModeU = wrap(s.address_u);
        c.addressModeV = wrap(s.address_v);
        c.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        if (rect || s.mip == 0) {
            c.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
            c.maxLod = 0.f;  // level 0 only, as GL without a mipmap filter
        } else {
            c.mipmapMode = s.mip == 1 ? VK_SAMPLER_MIPMAP_MODE_NEAREST : VK_SAMPLER_MIPMAP_MODE_LINEAR;
            c.maxLod = VK_LOD_CLAMP_NONE;
        }
        VkSampler smp = VK_NULL_HANDLE;
        vkCreateSampler(dev_, &c, nullptr, &smp);
        samplers_[key] = smp;
        return smp;
    }
    VkSampler Nearest() {
        if (!nearest_) {
            VkSamplerCreateInfo c{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
            c.magFilter = c.minFilter = VK_FILTER_NEAREST;
            c.addressModeU = c.addressModeV = c.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            vkCreateSampler(dev_, &c, nullptr, &nearest_);
        }
        return nearest_;
    }
    void Evict() {
        for (auto it = textures_.begin(); it != textures_.end();) {
            if (frame_ - it->second.used > 120) {
                Drop(it->second.img);
                it = textures_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // ---- shaders and pipelines ----

    bool MakeLayout() {
        // set 0: the two uniform blocks, at dynamic offsets in the frame's uniform ring
        VkDescriptorSetLayoutBinding u[2] = {};
        for (uint32_t i = 0; i < 2; i++) {
            u[i].binding = i;
            u[i].descriptorCount = 1;
            u[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            u[i].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo ul{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        ul.bindingCount = 2;
        ul.pBindings = u;
        if (vkCreateDescriptorSetLayout(dev_, &ul, nullptr, &ubo_layout_) != VK_SUCCESS) return false;
        // set 1: the four samplers, pushed when they change
        VkDescriptorSetLayoutBinding b[4] = {};
        for (uint32_t i = 0; i < 4; i++) {
            b[i].binding = vkglsl::kFirstSampler + i;
            b[i].descriptorCount = 1;
            b[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            b[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dl{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dl.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
        dl.bindingCount = 4;
        dl.pBindings = b;
        if (vkCreateDescriptorSetLayout(dev_, &dl, nullptr, &set_layout_) != VK_SUCCESS) return false;
        VkDescriptorSetLayout sets[2] = {ubo_layout_, set_layout_};
        VkPushConstantRange pr{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16};
        VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pl.setLayoutCount = 2;
        pl.pSetLayouts = sets;
        pl.pushConstantRangeCount = 1;
        pl.pPushConstantRanges = &pr;
        if (vkCreatePipelineLayout(dev_, &pl, nullptr, &layout_) != VK_SUCCESS) return false;
        VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 2 * 256};
        VkDescriptorPoolCreateInfo pc{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pc.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        pc.maxSets = 256;
        pc.poolSizeCount = 1;
        pc.pPoolSizes = &ps;
        return vkCreateDescriptorPool(dev_, &pc, nullptr, &pool_) == VK_SUCCESS;
    }
    // a uniform buffer of the given size and the set over it
    bool MakeUniforms(size_t bytes, Frame::Uniforms& u) {
        VkDeviceMemory mem;
        void* map = nullptr;
        if (!MakeBuffer(bytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, u.buf, mem, map)) return false;
        u.mem = mem;
        u.map = static_cast<uint8_t*>(map);
        u.size = bytes;
        u.head = 0;
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = pool_;
        ai.descriptorSetCount = 1;
        ai.pSetLayouts = &ubo_layout_;
        if (vkAllocateDescriptorSets(dev_, &ai, &u.set) != VK_SUCCESS) return false;
        VkDescriptorBufferInfo bi[2] = {{u.buf, 0, kUniformRange}, {u.buf, 0, kUniformRange}};
        VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        for (uint32_t i = 0; i < 2; i++) {
            w[i].dstSet = u.set;
            w[i].dstBinding = i;
            w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
            w[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(dev_, 2, w, 0, nullptr);
        return true;
    }
    void FreeUniforms(Frame::Uniforms& u) {
        if (u.set) vkFreeDescriptorSets(dev_, pool_, 1, &u.set);
        if (u.map) vkUnmapMemory(dev_, u.mem);
        if (u.buf) vkDestroyBuffer(dev_, u.buf, nullptr);
        if (u.mem) vkFreeMemory(dev_, u.mem, nullptr);
        u = Frame::Uniforms{};
    }
    // room for a draw's blocks in one uniform buffer (they share the set): vs 0 when the vertex block is reused from
    // earlier in the buffer of set want (then only that buffer will do); offsets, the set, where to write
    VkDescriptorSet UniformRoom(uint32_t vs, uint32_t fs, uint32_t& vs_off, uint32_t& fs_off, uint8_t*& base, VkDescriptorSet want) {
        Frame& f = frames_[frame_ % kFrames];
        const uint32_t align = uint32_t(std::max<VkDeviceSize>(props_.limits.minUniformBufferOffsetAlignment, 16));
        const uint32_t vsa = vs ? (std::max(vs, 16u) + align - 1) & ~(align - 1) : 0;
        const uint32_t need = vsa + std::max(fs, 16u) + kUniformRange;  // the range read past each offset stays inside
        auto take = [&](Frame::Uniforms& u) -> bool {
            if (want && u.set != want) return false;
            const size_t at = (u.head + align - 1) & ~size_t(align - 1);
            if (at + need > u.size) return false;
            if (vs) vs_off = uint32_t(at);
            fs_off = uint32_t(at + vsa);
            u.head = at + vsa + std::max(fs, 16u);
            base = u.map;
            return true;
        };
        if (take(f.uniforms)) return f.uniforms.set;
        for (auto& u : f.uextra)
            if (take(u)) return u.set;
        if (want) return VK_NULL_HANDLE;  // the reused block's buffer is full: the caller writes both again
        Frame::Uniforms u;
        if (!MakeUniforms(kUniformBytes, u)) {
            base = nullptr;
            return VK_NULL_HANDLE;
        }
        f.uextra.push_back(u);
        take(f.uextra.back());
        return f.uextra.back().set;
    }
    // SPIR-V of a GLSL source: from the disk cache by the source's hash, else compiled and kept
    VkShaderModule Module(const std::string& src, bool vertex) {
        uint64_t h = 1469598103934665603ull ^ (vertex ? 1 : 2);
        for (unsigned char ch : src) h = (h ^ ch) * 1099511628211ull;
        auto it = modules_.find(h);
        if (it != modules_.end()) return it->second;
        std::vector<uint32_t> spv;
        char name[40];
        std::snprintf(name, sizeof name, "/%016llx.spv", (unsigned long long)h);
        const std::string file = cache_dir_.empty() ? std::string() : cache_dir_ + name;
        if (!file.empty())
            if (FILE* f = std::fopen(file.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long n = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (n > 0 && n % 4 == 0) {
                    spv.resize(size_t(n) / 4);
                    if (std::fread(spv.data(), 1, size_t(n), f) != size_t(n)) spv.clear();
                }
                std::fclose(f);
            }
        if (spv.empty()) {
            std::string log;
            spv = vkglsl::Compile(src, vertex, &log);
            if (spv.empty()) {
                return modules_[h] = VK_NULL_HANDLE;
            }
            if (!file.empty())
                if (FILE* f = std::fopen(file.c_str(), "wb")) {
                    std::fwrite(spv.data(), 4, spv.size(), f);
                    std::fclose(f);
                }
        }
        VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        ci.codeSize = spv.size() * 4;
        ci.pCode = spv.data();
        VkShaderModule m = VK_NULL_HANDLE;
        vkCreateShaderModule(dev_, &ci, nullptr, &m);
        return modules_[h] = m;
    }
    static int Offset(const vkglsl::Converted& c, const char* name) {
        for (const auto& m : c.block)
            if (m.name == name) return int(m.offset);
        return -1;
    }
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
    // the constants a generated vertex program reads: u_c[N] by number, c(N + a0) anywhere (then all of them)
    static void ConstantRange(const std::string& src, uint32_t& lo, uint32_t& hi) {
        lo = 0;
        hi = 192;
        if (src.find(" + a0)") != std::string::npos) return;
        uint32_t a = 192, b = 0;
        for (size_t at = src.find("u_c["); at != std::string::npos; at = src.find("u_c[", at + 4)) {
            const char* p = src.c_str() + at + 4;
            if (*p < '0' || *p > '9') continue;  // the helper's u_c[clamp(...)]
            const uint32_t n = uint32_t(std::strtoul(p, nullptr, 10));
            if (n >= 192) continue;  // the declaration
            a = std::min(a, n);
            b = std::max(b, n + 1);
        }
        if (a < b) {
            lo = a;
            hi = b;
        } else {
            lo = hi = 0;
        }
    }
    Prog* GetProgram(const reference::DrawPacket& pk, bool atest, uint32_t variants, uint32_t w1) {
        const uint64_t key = glshader::CacheHash(pk.key, pk.info, atest) ^
                             (pk.raw ? (pk.program_hash ^ uint64_t(variants) << 1 ^ uint64_t(w1) << 33) * 0x9E3779B97F4A7C15ull + 1 : 0);
        auto it = programs_.find(key);
        if (it != programs_.end()) return &it->second;
        Prog p;
        p.id = key;
        const std::string vsrc = pk.raw ? glshader::VertexProgramSource(*pk.program, variants, w1) : std::string(glshader::kVertexSource);
        ConstantRange(vsrc, p.c_lo, p.c_hi);
        const vkglsl::Converted vs = vkglsl::ToVulkan(vsrc, true);
        const vkglsl::Converted fs = vkglsl::ToVulkan(glshader::FragmentSource(pk.key, pk.info, atest), false);
        p.vs = Module(vs.source, true);
        p.fs = Module(fs.source, false);
        p.vs_bytes = vs.block_bytes;
        p.fs_bytes = fs.block_bytes;
        p.c = Offset(vs, "u_c");
        p.surf = Offset(vs, "u_surf");
        p.fogp = Offset(vs, "u_fogp");
        p.vsmisc = Offset(vs, "u_vsmisc");
        p.pc = Offset(fs, "u_pc");
        p.tf = Offset(fs, "u_tf");
        p.fogc = Offset(fs, "u_fogc");
        p.at = Offset(fs, "u_at");
        p.bm = Offset(fs, "u_bm");
        p.bl = Offset(fs, "u_bl");
        p.tsz = Offset(fs, "u_tsz");
        p.samplers = fs.samplers;
        p.ok = p.vs && p.fs;
        return &(programs_[key] = p);
    }

    // a raw draw's input register as Vulkan reads it; variant as glshader takes it; four: three components of a byte or
    // short format read as four, w made 1 in the shader
    VkFormat RawFormat(uint8_t dxgi, uint32_t decode, uint32_t& variant, bool& four) const {
        const bool w1 = decode == kDecodeW1 || decode == kDecodeIntW1;
        variant = 0;
        four = false;
        switch (dxgi) {
            case DXGI_FORMAT_R32_FLOAT: return VK_FORMAT_R32_SFLOAT;
            case DXGI_FORMAT_R32G32_FLOAT: return VK_FORMAT_R32G32_SFLOAT;
            case DXGI_FORMAT_R32G32B32_FLOAT:
                variant = decode == kDecodeFloat2H ? 3u : 0u;
                return VK_FORMAT_R32G32B32_SFLOAT;
            case DXGI_FORMAT_R32G32B32A32_FLOAT: return VK_FORMAT_R32G32B32A32_SFLOAT;
            case DXGI_FORMAT_B8G8R8A8_UNORM: variant = 1; return VK_FORMAT_R8G8B8A8_UNORM;
            case DXGI_FORMAT_R16_SNORM: return VK_FORMAT_R16_SNORM;
            case DXGI_FORMAT_R16G16_SNORM: return VK_FORMAT_R16G16_SNORM;
            case DXGI_FORMAT_R16G16B16A16_SNORM: four = w1; return VK_FORMAT_R16G16B16A16_SNORM;
            case DXGI_FORMAT_R8_UNORM: return VK_FORMAT_R8_UNORM;
            case DXGI_FORMAT_R8G8_UNORM: return VK_FORMAT_R8G8_UNORM;
            case DXGI_FORMAT_R8G8B8A8_UNORM: four = w1; return VK_FORMAT_R8G8B8A8_UNORM;
            case DXGI_FORMAT_R16_SINT: return sscaled_ ? VK_FORMAT_R16_SSCALED : VK_FORMAT_R16_SNORM;
            case DXGI_FORMAT_R16G16_SINT: return sscaled_ ? VK_FORMAT_R16G16_SSCALED : VK_FORMAT_R16G16_SNORM;
            case DXGI_FORMAT_R16G16B16A16_SINT: four = w1; return sscaled_ ? VK_FORMAT_R16G16B16A16_SSCALED : VK_FORMAT_R16G16B16A16_SNORM;
            case DXGI_FORMAT_R32_UINT: variant = 2; return VK_FORMAT_R32_UINT;
            default: return VK_FORMAT_R32G32B32A32_SFLOAT;
        }
    }

    struct PipeKey {
        uint64_t prog = 0;
        uint32_t cmask = 0, depth = 0, blend = 0, src = 0, dst = 0, op = 0, nattr = 0, nbind = 0;
        // topology 1, cull 2 (0 none, 1 back with GL's clockwise front, 2 back with counter clockwise), depth test 1,
        // depth write 1, depth function 3, stencil test 1, stencil fail, pass, depth fail and function 3 each
        uint32_t state = 0;
        uint16_t stride[17] = {};
        Attr attr[17];
        uint64_t Hash() const {
            uint64_t h = prog * 0x9E3779B97F4A7C15ull;
            auto mix = [&](uint64_t v) { h = (h ^ v) * 1099511628211ull; };
            mix(cmask | depth << 4 | blend << 5 | uint64_t(nattr) << 8 | uint64_t(nbind) << 16);
            mix(uint64_t(src) << 32 | dst);
            mix(op);
            mix(state);
            for (uint32_t b = 0; b < nbind; b++) mix(stride[b]);
            for (uint32_t i = 0; i < nattr; i++) mix(uint64_t(attr[i].location) | uint64_t(attr[i].binding) << 8 | uint64_t(attr[i].format) << 16 | uint64_t(attr[i].offset) << 40);
            return h;
        }
    };
    VkPipeline GetPipeline(const PipeKey& k, const Prog& prog) {
        const uint64_t h = k.Hash();
        auto it = pipes_.find(h);
        if (it != pipes_.end()) return it->second;
        VkVertexInputBindingDescription binds[17];
        uint32_t nb = 0;
        for (uint32_t b = 0; b <= kConstBinding; b++) binds[nb++] = {b, b < k.nbind ? uint32_t(k.stride[b]) : 0u, VK_VERTEX_INPUT_RATE_VERTEX};
        VkVertexInputAttributeDescription attrs[17];
        for (uint32_t i = 0; i < k.nattr; i++) attrs[i] = {k.attr[i].location, k.attr[i].binding, k.attr[i].format, k.attr[i].offset};
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        vi.vertexBindingDescriptionCount = nb;
        vi.pVertexBindingDescriptions = binds;
        vi.vertexAttributeDescriptionCount = k.nattr;
        vi.pVertexAttributeDescriptions = attrs;
        VkPipelineColorBlendAttachmentState cb{};
        cb.colorWriteMask = k.cmask;
        if (k.blend) {
            cb.blendEnable = VK_TRUE;
            cb.srcColorBlendFactor = cb.srcAlphaBlendFactor = BlendFactor(k.src);
            cb.dstColorBlendFactor = cb.dstAlphaBlendFactor = BlendFactor(k.dst);
            cb.colorBlendOp = cb.alphaBlendOp = BlendOp(k.op);
        }
        VkPipeline p = MakePipeline(prog.vs, prog.fs, vi, cb, k.depth != 0, &k);
        return pipes_[h] = p;
    }
    // the pipeline state every draw shares; full: the draws' dynamic depth, stencil, cull and topology states
    VkPipeline MakePipeline(VkShaderModule vs, VkShaderModule fs, const VkPipelineVertexInputStateCreateInfo& vi,
                            const VkPipelineColorBlendAttachmentState& cb, bool depth, const PipeKey* key) {
        const bool full = key != nullptr;
        VkPipelineShaderStageCreateInfo st[2] = {{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}, {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
        st[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        st[0].module = vs;
        st[0].pName = "main";
        st[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        st[1].module = fs;
        st[1].pName = "main";
        VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        const uint32_t fx = key ? key->state : 0;
        ia.topology = (fx & 1) ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP : VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        vp.viewportCount = 1;
        vp.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        rs.polygonMode = VK_POLYGON_MODE_FILL;
        rs.cullMode = (fx >> 1 & 3) ? VK_CULL_MODE_BACK_BIT : VK_CULL_MODE_NONE;
        // GL's clockwise front is counter clockwise in Vulkan's downward framebuffer y
        rs.frontFace = (fx >> 1 & 3) == 1 ? VK_FRONT_FACE_COUNTER_CLOCKWISE : VK_FRONT_FACE_CLOCKWISE;
        rs.lineWidth = 1.f;
        VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        ds.depthTestEnable = (fx >> 3) & 1;
        ds.depthWriteEnable = (fx >> 4) & 1;
        ds.depthCompareOp = ds.depthTestEnable ? VkCompareOp(fx >> 5 & 7) : VK_COMPARE_OP_ALWAYS;
        ds.stencilTestEnable = (fx >> 8) & 1;
        ds.front.failOp = VkStencilOp(fx >> 9 & 7);
        ds.front.passOp = VkStencilOp(fx >> 12 & 7);
        ds.front.depthFailOp = VkStencilOp(fx >> 15 & 7);
        ds.front.compareOp = VkCompareOp(fx >> 18 & 7);
        ds.back = ds.front;
        VkPipelineColorBlendStateCreateInfo bs{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        bs.attachmentCount = 1;
        bs.pAttachments = &cb;
        std::vector<VkDynamicState> dyn = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        if (full) {
            dyn.insert(dyn.end(), {VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
                                   VK_DYNAMIC_STATE_STENCIL_REFERENCE});
        }
        VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dy.dynamicStateCount = uint32_t(dyn.size());
        dy.pDynamicStates = dyn.data();
        const VkFormat colour = kColour;
        VkPipelineRenderingCreateInfo rci{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        rci.colorAttachmentCount = 1;
        rci.pColorAttachmentFormats = &colour;
        if (depth) {
            rci.depthAttachmentFormat = depth_format_;
            rci.stencilAttachmentFormat = depth_format_;
        }
        VkGraphicsPipelineCreateInfo pc{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        pc.pNext = &rci;
        pc.stageCount = 2;
        pc.pStages = st;
        pc.pVertexInputState = &vi;
        pc.pInputAssemblyState = &ia;
        pc.pViewportState = &vp;
        pc.pRasterizationState = &rs;
        pc.pMultisampleState = &ms;
        pc.pDepthStencilState = &ds;
        pc.pColorBlendState = &bs;
        pc.pDynamicState = &dy;
        pc.layout = layout_;
        VkPipeline p = VK_NULL_HANDLE;
        if (vkCreateGraphicsPipelines(dev_, pcache_, 1, &pc, nullptr, &p) != VK_SUCCESS) p = VK_NULL_HANDLE;
        pcache_dirty_ = true;
        return p;
    }
    void LoadPipelineCache() {
        std::vector<uint8_t> blob;
        if (!cache_dir_.empty())
            if (FILE* f = std::fopen((cache_dir_ + "/pipelines.bin").c_str(), "rb")) {
                uint8_t buf[65536];
                for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) blob.insert(blob.end(), buf, buf + n);
                std::fclose(f);
            }
        VkPipelineCacheCreateInfo ci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        ci.initialDataSize = blob.size();
        ci.pInitialData = blob.empty() ? nullptr : blob.data();
        if (vkCreatePipelineCache(dev_, &ci, nullptr, &pcache_) != VK_SUCCESS) {
            ci.initialDataSize = 0;
            ci.pInitialData = nullptr;
            vkCreatePipelineCache(dev_, &ci, nullptr, &pcache_);
        }
    }
    void SavePipelineCache() {
        if (!pcache_ || !pcache_dirty_ || cache_dir_.empty()) return;
        size_t n = 0;
        vkGetPipelineCacheData(dev_, pcache_, &n, nullptr);
        std::vector<uint8_t> blob(n);
        if (n && vkGetPipelineCacheData(dev_, pcache_, &n, blob.data()) == VK_SUCCESS)
            if (FILE* f = std::fopen((cache_dir_ + "/pipelines.bin").c_str(), "wb")) {
                std::fwrite(blob.data(), 1, n, f);
                std::fclose(f);
            }
        pcache_dirty_ = false;
    }

    // ---- full screen passes: masked clears, depth as colour, gamma ----

    // a full screen triangle's vertex shader, and fragment shaders that draw a colour, a depth as the console's word,
    // and the gamma ramp
    VkShaderModule FullScreenVs() {
        if (!fs_vs_)
            fs_vs_ = Module("#version 450\nlayout(location = 0) out vec2 v_uv;\n"
                            "void main() { vec2 p = vec2(float((gl_VertexIndex & 1) << 2) - 1.0, float((gl_VertexIndex & 2) << 1) - 1.0);\n"
                            "  v_uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }\n",
                            true);
        return fs_vs_;
    }
    VkPipeline Utility(int kind, uint32_t cmask, bool depth = false) {
        const uint64_t key = uint64_t(kind) << 8 | cmask | (depth ? 1ull << 16 : 0);
        auto it = utility_.find(key);
        if (it != utility_.end()) return it->second;
        static const char* kFs[3] = {
            // 0: a colour from the push constants
            "#version 450\nlayout(push_constant) uniform P { vec4 c; } pc;\nlayout(location = 0) out vec4 o;\nvoid main() { o = pc.c; }\n",
            // 1: the depth as the console's D24S8 word read as A8R8G8B8
            "#version 450\nlayout(set = 1, binding = 2) uniform highp sampler2D u_depth;\nlayout(location = 0) out vec4 o;\n"
            "void main() {\n  float d = texelFetch(u_depth, ivec2(gl_FragCoord.xy), 0).r;\n"
            "  float z = clamp(((2.0 * d - 1.0) / 0.99995 + 1.0) * 0.5, 0.0, 1.0);\n  uint q = uint(z * 16777215.0 + 0.5);\n"
            "  o = vec4(float((q >> 8) & 255u), float(q & 255u), 0.0, float((q >> 16) & 255u)) / 255.0;\n}\n",
            // 2: each channel looked up in the gamma ramp
            "#version 450\nlayout(set = 1, binding = 2) uniform highp sampler2D u_src;\nlayout(set = 1, binding = 3) uniform highp sampler2D u_lut;\n"
            "layout(location = 0) out vec4 o;\nvoid main() {\n  vec4 c = texelFetch(u_src, ivec2(gl_FragCoord.xy), 0);\n"
            "  ivec3 i = ivec3(clamp(c.rgb * 255.0 + 0.5, 0.0, 255.0));\n"
            "  o = vec4(texelFetch(u_lut, ivec2(i.r, 0), 0).r, texelFetch(u_lut, ivec2(i.g, 0), 0).g, texelFetch(u_lut, ivec2(i.b, 0), 0).b, c.a);\n}\n"};
        VkShaderModule fs = Module(kFs[kind], false);
        VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineColorBlendAttachmentState cb{};
        cb.colorWriteMask = cmask;
        VkPipeline p = MakePipeline(FullScreenVs(), fs, vi, cb, depth, nullptr);
        return utility_[key] = p;
    }
    // some channels of a clear: a full screen colour drawn through the rects with the mask
    void MaskedClear(const std::vector<VkClearRect>& rs, uint32_t color, uint32_t mask) {
        VkPipeline p = Utility(0, mask, cur_zs_ != nullptr);
        if (!p) return;
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        bound_pipe_ = p;
        const float c[4] = {float((color >> 16) & 255) / 255.f, float((color >> 8) & 255) / 255.f, float(color & 255) / 255.f, float(color >> 24) / 255.f};
        vkCmdPushConstants(cmd_, layout_, VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, c);
        for (const auto& r : rs) {
            vkCmdSetScissor(cmd_, 0, 1, &r.rect);
            vkCmdDraw(cmd_, 3, 1, 0, 0);
        }
        dyn_ = Dyn{};
    }
    // the depth buffer as the console's memory holds it, through a colour image
    VkImageView DepthAsColour(DepthBuf& d) {
        Settle(d);
        if (!d.written && d.colour) return d.colour.view;
        d.written = false;
        if (!d.colour) d.colour = MakeImage(d.sw, d.sh, kColour, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        if (!d.colour) return VK_NULL_HANDLE;
        VkPipeline p = Utility(1, 0xF);
        if (!p) return VK_NULL_HANDLE;
        EndRendering();
        Barrier(cmd_, d.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        Barrier(cmd_, d.colour, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        FullScreenPass(d.colour.view, d.sw, d.sh, p, {d.img.depth_view, VK_NULL_HANDLE}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        Barrier(cmd_, d.colour, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return d.colour.view;
    }
    // a full screen triangle into view, its samplers from images (binding 2 and 3)
    void FullScreenPass(VkImageView view, uint32_t w, uint32_t h, VkPipeline p, std::array<VkImageView, 2> images, VkImageLayout layout) {
        VkRenderingAttachmentInfo ca{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca.imageView = view;
        ca.imageLayout = layout;
        ca.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        ca.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, {w, h}};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &ca;
        vkCmdBeginRendering_(cmd_, &ri);
        vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
        VkViewport vp{0, 0, float(w), float(h), 0.f, 1.f};
        vkCmdSetViewport(cmd_, 0, 1, &vp);
        VkRect2D sc{{0, 0}, {w, h}};
        vkCmdSetScissor(cmd_, 0, 1, &sc);
        VkDescriptorImageInfo ii[2];
        VkWriteDescriptorSet wr[2];
        uint32_t n = 0;
        for (uint32_t i = 0; i < 2; i++) {
            if (!images[i]) continue;
            ii[n] = {Nearest(), images[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            wr[n] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[n].dstBinding = vkglsl::kFirstSampler + i;
            wr[n].descriptorCount = 1;
            wr[n].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            wr[n].pImageInfo = &ii[n];
            n++;
        }
        if (n) vkCmdPushDescriptorSetKHR_(cmd_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_, 1, n, wr);
        vkCmdDraw(cmd_, 3, 1, 0, 0);
        vkCmdEndRendering_(cmd_);
        bound_pipe_ = VK_NULL_HANDLE;
        dyn_ = Dyn{};  // the utility pipeline's static state and samplers replaced the draws' ones
    }

    // ---- the frame to the window ----

    struct Shown {
        void* ahb = nullptr;
        Image img;
    };
    // the window's hardware buffer k as an image of ours (made again when the window made a new buffer)
    Image* WindowImage(int k, void* ahb) {
        Shown& s = shown_[k];
        if (s.ahb == ahb && s.img) return &s.img;
        if (s.img) {
            vkDeviceWaitIdle(dev_);
            Destroy(s.img);
        }
        s.ahb = ahb;
        auto* buffer = static_cast<AHardwareBuffer*>(ahb);
        VkAndroidHardwareBufferFormatPropertiesANDROID fmt{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
        VkAndroidHardwareBufferPropertiesANDROID props{VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
        props.pNext = &fmt;
        if (vkGetAndroidHardwareBufferPropertiesANDROID_(dev_, buffer, &props) != VK_SUCCESS) return nullptr;
        AHardwareBuffer_Desc desc{};
        AHardwareBuffer_describe(buffer, &desc);
        VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
        ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
        VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ici.pNext = &ext;
        ici.imageType = VK_IMAGE_TYPE_2D;
        ici.format = kColour;
        ici.extent = {desc.width, desc.height, 1};
        ici.mipLevels = 1;
        ici.arrayLayers = 1;
        ici.samples = VK_SAMPLE_COUNT_1_BIT;
        ici.tiling = VK_IMAGE_TILING_OPTIMAL;
        ici.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        Image im;
        if (vkCreateImage(dev_, &ici, nullptr, &im.img) != VK_SUCCESS) return nullptr;
        VkImportAndroidHardwareBufferInfoANDROID imp{VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
        imp.buffer = buffer;
        VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
        ded.pNext = &imp;
        ded.image = im.img;
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.pNext = &ded;
        mai.allocationSize = props.allocationSize;
        mai.memoryTypeIndex = MemType(props.memoryTypeBits, 0);
        if (vkAllocateMemory(dev_, &mai, nullptr, &im.mem) != VK_SUCCESS) {
            vkDestroyImage(dev_, im.img, nullptr);
            return nullptr;
        }
        vkBindImageMemory(dev_, im.img, im.mem, 0);
        im.w = desc.width;
        im.h = desc.height;
        im.view = View(im.img, kColour, VK_IMAGE_ASPECT_COLOR_BIT, false, 1, false);
        s.img = im;
        return &s.img;
    }
    // the frame copied into the window's buffer it is not showing, after the window's last draw of it, and a sync file
    // for the window to wait on
    bool PresentDirect(Target& t) {
        auto& pc = platform::GetPresentChain();
        if (!pc.active) return false;
        std::unique_lock lock(pc.lock);
        pc.want_w = t.sw;
        pc.want_h = t.sh;
        if (pc.w != t.sw || pc.h != t.sh || !pc.buffer[0]) {
            Submit(nullptr, nullptr, true);
            return true;  // the window makes buffers of this size first
        }
        const int k = pc.Free(lock);
        if (k < 0 || !pc.buffer[k]) {
            Submit(nullptr, nullptr, true);
            return true;
        }
        Image* win = WindowImage(k, pc.buffer[k]);
        if (!win) {
            Submit(nullptr, nullptr, true);
            return true;
        }
        if (pc.write_fd[k] >= 0) {
            close(pc.write_fd[k]);
            pc.write_fd[k] = -1;
        }
        // the frame's drawing goes now; only the copy below waits for the window
        Submit(nullptr, nullptr, false);
        // the window's buffer comes from (and goes back to) the other GL context: owned by a foreign queue
        VkImageMemoryBarrier acq{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        acq.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        acq.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        acq.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        acq.srcQueueFamilyIndex = foreign_ ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_IGNORED;
        acq.dstQueueFamilyIndex = foreign_ ? qfam_ : VK_QUEUE_FAMILY_IGNORED;
        acq.image = win->img;
        acq.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &acq);
        if (!gamma_identity_) {
            Layout(t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            FullScreenPass(win->view, t.sw, t.sh, Utility(2, 0xF), {t.img.view, GammaLut()}, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        } else {
            // a copy needs the transfer layout; the attachment layout above is changed again here
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = win->img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
            Layout(t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkImageCopy c{};
            c.srcSubresource = c.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.extent = {t.sw, t.sh, 1};
            vkCmdCopyImage(cmd_, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, win->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
            VkImageMemoryBarrier back = b;
            back.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            back.dstAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
            back.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            back.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &back);
            // readable again now: the next frame's effects read the frame before (motion blur) while another pass is
            // open, which a layout change then would end
            Barrier(cmd_, t.img, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        }
        VkImageMemoryBarrier rel{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        rel.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        rel.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        rel.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        rel.srcQueueFamilyIndex = foreign_ ? qfam_ : VK_QUEUE_FAMILY_IGNORED;
        rel.dstQueueFamilyIndex = foreign_ ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_IGNORED;
        rel.image = win->img;
        rel.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdPipelineBarrier(cmd_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &rel);
        // the window's draw of this buffer finished before we write it
        VkSemaphore wait = VK_NULL_HANDLE;
        if (pc.read_fd[k] >= 0) {
            VkImportSemaphoreFdInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR};
            imp.semaphore = window_done_;
            imp.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
            imp.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
            imp.fd = pc.read_fd[k];
            if (vkImportSemaphoreFdKHR_(dev_, &imp) == VK_SUCCESS) wait = window_done_;
            else close(pc.read_fd[k]);
            pc.read_fd[k] = -1;
        }
        Submit(wait, present_done_, true);
        int fd = -1;
        VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
        gi.semaphore = present_done_;
        gi.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_SYNC_FD_BIT;
        if (vkGetSemaphoreFdKHR_(dev_, &gi, &fd) != VK_SUCCESS) fd = -1;
        pc.write_fd[k] = fd;
        pc.Push(k);
        pc.serial++;
        return true;
    }
    // without the window's buffers: the frame read back and handed over in memory
    bool PresentThroughMemory(Target& t) {
        VkDeviceSize off = 0;
        uint8_t* dst = nullptr;
        const size_t bytes = size_t(t.sw) * t.sh * 4;
        VkBuffer buf = Room(bytes, 16, off, dst);
        if (!dst) return false;
        Layout(t.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        VkBufferImageCopy r{};
        r.bufferOffset = off;
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {t.sw, t.sh, 1};
        vkCmdCopyImageToBuffer(cmd_, t.img.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buf, 1, &r);
        Submit(nullptr, nullptr, true);
        Frame& f = frames_[frame_ % kFrames];
        vkWaitForFences(dev_, 1, &f.fence, VK_TRUE, UINT64_MAX);
        auto& fr = platform::GetFrame();
        std::lock_guard lock(fr.lock);
        fr.rgba.resize(size_t(t.sw) * t.sh);
        std::memcpy(fr.rgba.data(), dst, bytes);
        fr.w = t.sw;
        fr.h = t.sh;
        fr.serial++;
        return true;
    }
    VkImageView GammaLut() {
        if (!gamma_lut_) gamma_lut_ = MakeImage(256, 1, kColour, VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        if (gamma_dirty_) {
            uint8_t lut[256 * 4];
            for (int i = 0; i < 256; i++) {
                lut[i * 4] = gamma_[i];
                lut[i * 4 + 1] = gamma_[256 + i];
                lut[i * 4 + 2] = gamma_[512 + i];
                lut[i * 4 + 3] = 255;
            }
            VkDeviceSize off = 0;
            VkBuffer buf = Put(lut, sizeof lut, 16, off);
            Layout(gamma_lut_, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy r{};
            r.bufferOffset = off;
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            r.imageExtent = {256, 1, 1};
            vkCmdCopyBufferToImage(cmd_, buf, gamma_lut_.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
            gamma_dirty_ = false;
        }
        Layout(gamma_lut_, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        return gamma_lut_.view;
    }

    // ---- visibility tests ----

    static double CoveredPixels(const reference::DrawPacket& pk, int left, int top, int right, int bottom) {
        double total = 0;
        const double limit = double(std::max(right - left, 0)) * double(std::max(bottom - top, 0));
        for (size_t i = 0; i + 2 < pk.indices.size(); i += 3) {
            const reference::GlVertex* v[3] = {&pk.verts[pk.indices[i]], &pk.verts[pk.indices[i + 1]], &pk.verts[pk.indices[i + 2]]};
            float x[3], y[3];
            for (int j = 0; j < 3; j++) {
                const float w = v[j]->clip[3];
                if (!(w > 0)) return limit;
                x[j] = std::clamp(v[j]->clip[0] / w, float(left), float(right));
                y[j] = std::clamp(v[j]->clip[1] / w, float(top), float(bottom));
            }
            total += std::fabs(double((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))) * 0.5;
        }
        return std::min(total, limit);
    }
    // the results of a frame slot whose fence has passed
    void DeliverReportsOf(uint32_t slot) {
        Frame& f = frames_[slot];
        for (auto& r : reports_)
            for (auto& c : r.counted)
                if (c->passed < 0 && c->slot == slot && c->query != ~0u) {
                    uint64_t v = 0;
                    vkGetQueryPoolResults(dev_, f.queries, c->query, 1, sizeof v, &v, sizeof v, VK_QUERY_RESULT_64_BIT);
                    c->passed = v ? 1 : 0;
                }
    }
    // reports whose tests have all been read, written as the console's GPU writes them: time, count, then 0 for done
    void DeliverReports() {
        constexpr uint32_t kTime = 0, kTimeHigh = 4, kCount = 8, kStatus = 12;
        while (!reports_.empty()) {
            Report& r = reports_.front();
            // two presents old: the GPU is done with it (its slot's fence is waited for at the latest then)
            if (frame_ < r.frame + 2) return;
            for (auto& c : r.counted)
                if (c->passed < 0) {
                    if (c->query == ~0u) { c->passed = 0; continue; }
                    Frame& f = frames_[c->slot];
                    if (f.submitted) vkWaitForFences(dev_, 1, &f.fence, VK_TRUE, UINT64_MAX);
                    DeliverReportsOf(c->slot);
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

    uint32_t SX(int x) const { return uint32_t(std::lround(float(x) * kx_)); }
    uint32_t SY(int y) const { return uint32_t(std::lround(float(y) * ky_)); }
    // A target's scale. The picture's targets (the title's 640x480 ones) are drawn at the screen's resolution; the
    // smaller ones the effects draw into (the car reflection's cube faces, blur and glow, shadows) at most 1.5 times
    // the console's size, as on a 720 line screen: at the screen's own resolution they were most of the GPU's work
    // and heat for no difference one can see
    static constexpr float kEffectScale = 1.5f;
    static float ScaleOf(uint32_t w, uint32_t h, float k) { return w < 512 || h < 384 ? std::min(k, kEffectScale) : k; }
    uint32_t TW(uint32_t w, uint32_t h, float k) const { return std::max(1u, uint32_t(std::lround(float(w) * ScaleOf(w, h, k)))); }
    uint32_t TW(uint32_t w, uint32_t h) const { return TW(w, h, kx_); }
    uint32_t TH(uint32_t w, uint32_t h, float k) const { return std::max(1u, uint32_t(std::lround(float(h) * ScaleOf(w, h, k)))); }
    uint32_t TH(uint32_t w, uint32_t h) const { return TH(w, h, ky_); }
    // a console pixel position in a target's own pixels
    static int PX(const Target& t, int x) { return int(std::lround(double(x) * t.sw / t.w)); }
    static int PY(const Target& t, int y) { return int(std::lround(double(y) * t.sh / t.h)); }

    // device
    VkInstance inst_ = VK_NULL_HANDLE;
    VkPhysicalDevice phys_ = VK_NULL_HANDLE;
    VkDevice dev_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t qfam_ = 0;
    VkPhysicalDeviceProperties props_{};
    VkPhysicalDeviceMemoryProperties memprops_{};
    VkFormat depth_format_ = VK_FORMAT_D24_UNORM_S8_UINT;
    bool foreign_ = false, sscaled_ = false;
    PFN_vkCmdBeginRendering vkCmdBeginRendering_ = nullptr;
    PFN_vkCmdEndRendering vkCmdEndRendering_ = nullptr;
    PFN_vkCmdSetCullMode vkCmdSetCullMode_ = nullptr;
    PFN_vkCmdSetFrontFace vkCmdSetFrontFace_ = nullptr;
    PFN_vkCmdSetPrimitiveTopology vkCmdSetPrimitiveTopology_ = nullptr;
    PFN_vkCmdSetDepthTestEnable vkCmdSetDepthTestEnable_ = nullptr;
    PFN_vkCmdSetDepthWriteEnable vkCmdSetDepthWriteEnable_ = nullptr;
    PFN_vkCmdSetDepthCompareOp vkCmdSetDepthCompareOp_ = nullptr;
    PFN_vkCmdSetStencilTestEnable vkCmdSetStencilTestEnable_ = nullptr;
    PFN_vkCmdSetStencilOp vkCmdSetStencilOp_ = nullptr;
    PFN_vkCmdBindVertexBuffers2 vkCmdBindVertexBuffers2_ = nullptr;
    PFN_vkCmdPushDescriptorSetKHR vkCmdPushDescriptorSetKHR_ = nullptr;
    PFN_vkGetSemaphoreFdKHR vkGetSemaphoreFdKHR_ = nullptr;
    PFN_vkImportSemaphoreFdKHR vkImportSemaphoreFdKHR_ = nullptr;
    PFN_vkGetAndroidHardwareBufferPropertiesANDROID vkGetAndroidHardwareBufferPropertiesANDROID_ = nullptr;

    // frames and recording
    Frame frames_[kFrames];
    VkCommandBuffer cmd_ = VK_NULL_HANDLE, upload_ = VK_NULL_HANDLE;
    bool rendering_ = false;
    Target* cur_rt_ = nullptr;
    DepthBuf* cur_zs_ = nullptr;
    VkPipeline bound_pipe_ = VK_NULL_HANDLE;
    // dynamic state as last set in the command buffer (a utility pipeline or a new command buffer forgets it)
    struct Dyn {
        int cull = -1, front = -1, topo = -1, ztest = -1, zwrite = -1, zfunc = -1, sten = -1, smask = -1, wmask = -1, sref = -1;
        uint32_t sops[4] = {}, bc = 0, tex_mask = 0;
        bool sops_set = false, bc_set = false, scissor_set = false, tex_set = false;
        VkRect2D scissor{};
        VkDescriptorImageInfo tex[4] = {};
        VkDescriptorSet ubo = VK_NULL_HANDLE;
        uint32_t ubo_off[2] = {};
        bool ubo_set = false;
        VkBuffer vb[17] = {};
        VkDeviceSize voff[17] = {};
        bool vb_set = false;
        VkBuffer ib = VK_NULL_HANDLE;
        VkDeviceSize ib_off = 0;
        VkIndexType ib_type = VK_INDEX_TYPE_UINT16;
        bool ib_set = false;
    } dyn_;
    VkDescriptorSetLayout ubo_layout_ = VK_NULL_HANDLE;
    // the vertex block last written: what it holds, and where
    struct VsBlock {
        VkShaderModule module = VK_NULL_HANDLE;
        uint64_t serial = 0;
        unsigned long long frame = 0;
        float v[12] = {};  // surface, fog parameters, specular
        VkDescriptorSet set = VK_NULL_HANDLE;
        uint32_t off = 0;
    } vs_last_;
    // the fragment block last written, and the uniform set and offsets bound
    struct FsBlock {
        std::array<uint8_t, 2048> data{};
        uint32_t bytes = 0, off = 0;
        unsigned long long frame = ~0ull;
        VkDescriptorSet set = VK_NULL_HANDLE;
    } fs_last_;
    std::array<uint8_t, 2048> fs_now_{};
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkBuffer ix_buf_ = VK_NULL_HANDLE, consts_ = VK_NULL_HANDLE;
    VkSemaphore present_done_ = VK_NULL_HANDLE, window_done_ = VK_NULL_HANDLE;
    unsigned long long frame_ = 0, draws_seen_ = 0;

    // pipelines and shaders
    VkDescriptorSetLayout set_layout_ = VK_NULL_HANDLE;
    VkPipelineLayout layout_ = VK_NULL_HANDLE;
    VkPipelineCache pcache_ = VK_NULL_HANDLE;
    bool pcache_dirty_ = false;
    std::unordered_map<uint64_t, VkShaderModule> modules_;
    std::unordered_map<uint64_t, Prog> programs_;
    std::unordered_map<uint64_t, VkPipeline> pipes_;
    std::unordered_map<uint64_t, VkPipeline> utility_;
    VkShaderModule fs_vs_ = VK_NULL_HANDLE;
    std::string cache_dir_;

    // resources
    std::vector<std::unique_ptr<Target>> targets_;
    std::vector<std::unique_ptr<DepthBuf>> depths_;
    std::unordered_map<uint64_t, Tex> textures_;
    std::unordered_map<uint32_t, VkSampler> samplers_;
    VkSampler nearest_ = VK_NULL_HANDLE;
    std::unordered_map<uint32_t, Cube> cubes_;
    std::unordered_map<uint64_t, Image> scratch_;
    Image dummy_, dummy_cube_, gamma_lut_;
    Shown shown_[platform::PresentChain::kBuffers];
    std::set<uint32_t> frame_buffers_;
    reference::DrawPacket pk_;
    std::vector<uint8_t> pack_;
    std::vector<uint16_t> index16_;
    std::vector<uint32_t> readback_, upload_px_;
    std::vector<std::shared_ptr<Counted>> counted_;
    std::deque<Report> reports_;
    uint8_t gamma_[768] = {};
    bool gamma_identity_ = true, gamma_dirty_ = false;
    bool power_registered_ = false;
    float kx_ = 1, ky_ = 1;
};

bool IsTargetThunk(uint32_t data, uint32_t format, uint32_t w, uint32_t h) { return g_self && g_self->IsTarget(data, format, w, h); }

}  // namespace

std::unique_ptr<Backend> CreateVkBackend() {
    auto b = std::make_unique<VkBackend>();
    if (!b->Setup()) return nullptr;
    return b;
}

}  // namespace gpu
