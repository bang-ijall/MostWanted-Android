// the latest presented frame, handed from the game's render path to the window thread
#pragma once
#include <cstdint>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>

namespace platform {
struct Frame {
    std::mutex lock;
    std::vector<uint32_t> rgba;  // R in the low byte
    uint32_t w = 0, h = 0;
    uint64_t serial = 0;
};
Frame& GetFrame();

// the renderer's finished frame handed to the window on the GPU, with no copy through memory: the window makes three
// images (Android hardware buffers as EGLImages, which both GL contexts can bind without sharing), the renderer copies
// each frame into one it is not showing and leaves an EGL fence the window waits on (EGL objects as plain pointers
// here, to keep EGL out of this header)
struct PresentChain {
    std::mutex lock;
    std::atomic<bool> window_ready{false};  // the window can show images (SDL draws with GL ES and has the EGL calls)
    std::atomic<bool> active{false};        // the renderer has the EGL calls too: frames come this way
    uint32_t want_w = 0, want_h = 0;        // the frame size the renderer has
    uint32_t w = 0, h = 0;                  // the size of the buffers the window made
    // the buffers (AHardwareBuffer): each side makes its own EGLImage of them, and the fences go across as Android
    // native fence files
    // Frames are shown in the order made, each once: one buffer on screen, up to two waiting, one being written. The
    // window took only the latest before, and when two frames came between two refreshes of the screen the first was
    // never shown and the next refresh showed the same picture again (120 made, about 100 seen, unsteady)
    static constexpr int kBuffers = 4, kMaxQueued = 2;
    void* buffer[kBuffers] = {};
    int write_fd[kBuffers] = {-1, -1, -1, -1};  // the renderer's copy into it done, for the window to wait on
    int read_fd[kBuffers] = {-1, -1, -1, -1};   // the window's draw of it done, for the renderer before writing again
    int latest = -1, in_use = -1;
    int queue[kBuffers] = {};  // made, not yet shown, oldest first
    int queued = 0;
    std::condition_variable taken;  // the window took a frame
    uint64_t serial = 0;

    bool Queued(int k) const {
        for (int i = 0; i < queued; i++)
            if (queue[i] == k) return true;
        return false;
    }
    // renderer, holding the lock: a buffer to write the next frame into. With the queue full it waits for the window
    // to take one (a refresh at most); if the window stalls, the oldest waiting frame is dropped instead
    int Free(std::unique_lock<std::mutex>& lock) {
        if (queued >= kMaxQueued && !taken.wait_for(lock, std::chrono::milliseconds(50), [&] { return queued < kMaxQueued; })) {
            const int old = queue[0];
            for (int i = 1; i < queued; i++) queue[i - 1] = queue[i];
            queued--;
            (void)old;  // its write fence is replaced when it is written again
        }
        for (int k = 0; k < kBuffers; k++)
            if (k != in_use && !Queued(k)) return k;
        return -1;
    }
    // renderer, holding the lock: frame k is ready
    void Push(int k) {
        if (queued < kBuffers) queue[queued++] = k;
        latest = k;
    }
    // window, holding the lock: the next frame to show, -1 for none new (call taken.notify_all after unlocking)
    int Pop() {
        if (!queued) return -1;
        const int k = queue[0];
        for (int i = 1; i < queued; i++) queue[i - 1] = queue[i];
        queued--;
        return k;
    }
};
PresentChain& GetPresentChain();

// renderer activity, logged by the window thread every few seconds
struct Stats {
    std::atomic<uint64_t> draws{0}, clears{0}, copies{0}, presents{0}, draw_us{0}, lit_pixels{0}, total_pixels{0}, nonblack{0};
};
Stats& GetStats();
// how many times the console's size the GPU draws at, set by the app before the game starts
int& RenderScale();
// the screen's height / width in landscape (0 until the window is up)
float& ScreenAspect();
// the render scale across and down (the console's pixels to the GPU's), set by the window thread from the resolution
// setting and the picture's size on the screen; the renderer follows at its next present
std::atomic<float>& RenderScaleX();
std::atomic<float>& RenderScaleY();

// where the console picture sits on the window: window pixel = dst + (picture 0..1 - src) / src size * dst size
struct Placement {
    float src_x = 0, src_y = 0, src_w = 1, src_h = 1;  // the shown part of the picture, 0..1
    float dst_x = 0, dst_y = 0, dst_w = 0, dst_h = 0;  // window pixels
};
Placement& PicturePlacement();  // window thread only

}  // namespace platform
