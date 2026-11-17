// Stage 1 renderer: the SDK's CPU reference renderer draws into guest memory, each Present copies the front buffer
// to the window thread. Correct but slow; the Vulkan backend replaces it.
#include <chrono>
#include <filesystem>
#include <cstdio>
#include <algorithm>
#include <string>
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "app/app.h"
#include "frame.h"
#include "gpu/backend.h"
#include "gpu/texture.h"
#include "kernel/kernel.h"
#include "kernel/memory.h"
#include "reference_renderer.h"

namespace platform {
PresentChain& GetPresentChain() {
    static PresentChain p;
    return p;
}
Frame& GetFrame() {
    static Frame f;
    return f;
}
Stats& GetStats() {
    static Stats s;
    return s;
}
int& RenderScale() {
    static int k = 1;
    return k;
}
float& ScreenAspect() {
    static float a = 0;
    return a;
}
std::atomic<float>& RenderScaleX() {
    static std::atomic<float> k{2.f};
    return k;
}
std::atomic<float>& RenderScaleY() {
    static std::atomic<float> k{2.f};
    return k;
}
Placement& PicturePlacement() {
    static Placement p;
    return p;
}
}  // namespace platform

namespace gpu {
namespace {

class SwBackend final : public Backend {
public:
    bool Init(uint32_t w, uint32_t h) override {
        reference::BeginFrame();
        return true;
    }
    void Clear(const State& s, uint32_t n, const uint8_t* rects, uint32_t flags, uint32_t color, float z,
               uint32_t stencil) override {
        if (!render_) return;
        platform::GetStats().clears++;
        reference::Clear(s, n, rects, flags, color, z, stencil);
    }
    void Draw(const State& s, uint32_t prim, uint32_t start, uint32_t count, const uint8_t* indices) override {
        if (!render_) return;
        auto t0 = std::chrono::steady_clock::now();
        reference::Draw(s, prim, start, count, indices);
        auto us = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count());
        auto& st = platform::GetStats();
        st.draws++;
        st.draw_us += us;
        frame_us_ += us;
    }
    void Copy(const Surface& src, const Surface& dst, uint32_t n, uint32_t rects, uint32_t points, float,
              float) override {
        if (!render_) return;
        platform::GetStats().copies++;
        reference::Copy(src, dst, n, rects, points);
    }
    void Present(const Surface& front) override {
        auto& st = platform::GetStats();
        if (render_) {
            auto& f = platform::GetFrame();
            std::lock_guard lock(f.lock);
            if (reference::ReadRgba(front, f.w, f.h, f.rgba)) {
                f.serial++;
                uint64_t lit = 0;
                for (size_t i = 0; i < f.rgba.size(); i += 16) lit += (f.rgba[i] & 0xFFFFFF) != 0;
                st.lit_pixels = lit;
                st.total_pixels = f.rgba.size() / 16;
            } else {
            }
            // a frame that takes long to draw is drawn one in several, so the game keeps its own speed
            double ms = double(frame_us_) / 1000.0;
            ema_ms_ = ema_ms_ == 0 ? ms : ema_ms_ * 0.6 + ms * 0.4;
            skip_ = std::min(5, int(ema_ms_ / 16.0));
            to_skip_ = skip_;
        }
        frame_us_ = 0;
        st.presents++;
        reference::BeginFrame();
        render_ = to_skip_ == 0;
        if (to_skip_ > 0) to_skip_--;
        ++frames_;
    }
    void Sync(uint32_t, uint32_t) override {}
    void SyncLate(uint32_t) override {}
    void Flush() override {}
    void SetGamma(const uint8_t*) override {}

private:
    bool render_ = true;
    int skip_ = 0, to_skip_ = 0;
    double ema_ms_ = 0;
    uint64_t frame_us_ = 0;
    unsigned long long frames_ = 0;
};

}  // namespace

std::unique_ptr<Backend> CreateGlBackend();
std::unique_ptr<Backend> CreateVkBackend();

// the GPU renderer, the CPU one only where neither Vulkan nor OpenGL ES can start
std::unique_ptr<Backend> CreateBackend() {
    {
        // Vulkan where the phone has what it needs, else OpenGL ES
        if (auto v = CreateVkBackend()) return v;
        if (auto g = CreateGlBackend()) return g;
    }
    return std::make_unique<SwBackend>();
}

}  // namespace gpu
